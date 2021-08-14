/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2021, The Souffle Developers. All rights reserved
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */
#pragma once

#include "souffle/utility/ParallelUtil.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

namespace souffle {
namespace details {

constexpr std::pair<unsigned, unsigned> ToPrime[] = {
        // https://primes.utm.edu/lists/2small/0bit.html
        // ((2^n) - k) is prime
        // {n, k}
        {4, 3},  // 2^4 - 3 = 13
        {8, 5},  // 8^5 - 5 = 251
        {9, 3}, {10, 3}, {11, 9}, {12, 3}, {13, 1}, {14, 3}, {15, 19}, {16, 15}, {17, 1}, {18, 5}, {19, 1},
        {20, 3}, {21, 9}, {22, 3}, {23, 15}, {24, 3}, {25, 39}, {26, 5}, {27, 39}, {28, 57}, {29, 3},
        {30, 35}, {31, 1}, {32, 5}, {33, 9}, {34, 41}, {35, 31}, {36, 5}, {37, 25}, {38, 45}, {39, 7},
        {40, 87}, {41, 21}, {42, 11}, {43, 57}, {44, 17}, {45, 55}, {46, 21}, {47, 115}, {48, 59}, {49, 81},
        {50, 27}, {51, 129}, {52, 47}, {53, 111}, {54, 33}, {55, 55}, {56, 5}, {57, 13}, {58, 27}, {59, 55},
        {60, 93}, {61, 1}, {62, 57}, {63, 25}};

// (2^64)-59 is the largest prime that fits in uint64_t
constexpr uint64_t LargestPrime64 = 18446744073709551557UL;

// Return a prime greater or equal to the lower bound.
// Return 0 if the next prime would not fit in 64 bits.
inline uint64_t GreaterOrEqualPrime(const uint64_t LowerBound) {
    if (LowerBound > LargestPrime64) {
        return 0;
    }

    for (auto&& [N, K] : ToPrime) {
        const uint64_t Prime = (1UL << uint64_t(N)) - K;
        if (Prime >= LowerBound) {
            return Prime;
        }
    }
    return LargestPrime64;
}

template <typename T>
struct Factory {
    template <class... Args>
    T& replace(T& Place, Args&&... Xs) {
        Place = T{std::forward<Args>(Xs)...};
        return Place;
    }
};

}  // namespace details

/**
 * A concurrent, almost lock-free associative hash-map that can only grow.
 * Elements cannot be removed, the hash-map can only grow.
 *
 * The datastructures enables a configurable number of concurrent access lanes.
 * Access to the datastructure is lock-free between different lanes.
 * Concurrent accesses through the same lane is sequential.
 *
 * Growing the datastructure requires to temporarily lock all lanes to let a
 * single lane perform the growing operation. The global lock is amortized
 * thanks to an exponential growth strategy.
 */
template <template <typename> class LanesPolicy, class Key, class T, class Hash = std::hash<Key>,
        class KeyEqual = std::equal_to<Key>, class KeyFactory = details::Factory<Key>>
class ConcurrentInsertOnlyHashMap {
    static size_t primeBucketSizing(size_t n_required) {
        auto n = details::GreaterOrEqualPrime(n_required);
        // Assert never expected to trigger. If we need more than `LargestPrime64` buckets we
        // couldn't allocate it in a 64 bit addr space b/c a bucket requires > 1 byte.
        assert(n != 0 && "absurd # of buckets required");
        return n;
    }

public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, const T>;
    using size_type = std::size_t;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using self_type = ConcurrentInsertOnlyHashMap<LanesPolicy, Key, T, Hash, KeyEqual, KeyFactory>;
    using lane_id = typename LanesPolicy<void>::lane_id;

private:
    // Each bucket of the hash-map is a linked list.
    struct BucketList {
        ~BucketList() {
            // If this triggers either:
            // a) Client code mutated data structure internal state. (Their problem.)
            // b) The cleanup code should have acquired and released this node.
            assert(!Next && "internal invariants violated");
        }

        // Stores the couple of a key and its associated value.
        value_type Value;

        // Points to next element of the map that falls into the same bucket.
        BucketList* Next;
    };

    using BucketVector = std::vector<std::atomic<BucketList*>>;

    template <typename F>
    void foreachEntry(F&& go) {
        for (auto& B : Buckets) {
            auto* L = B.load(std::memory_order_relaxed);
            while (L) {
                auto* E = L;
                L = L->Next;
                go(E);
            }
        }
    }

public:
    using node_type = std::unique_ptr<BucketList>;

    /**
     * @brief Construct a hash-map with at least the given number of buckets.
     *
     * Load-factor is initialized to 1.0.
     */
    ConcurrentInsertOnlyHashMap(const std::size_t /*LaneCount*/, std::size_t Bucket_Count, Hash hash = {},
            KeyEqual key_equal = {}, KeyFactory key_factory = {})
            : Hasher(std::move(hash)), EqualTo(std::move(key_equal)), Factory(std::move(key_factory)) {
        Size = 0;
        LoadFactor = 1.0;  //< TODO: has this been tune?
        Buckets = BucketVector(primeBucketSizing(Bucket_Count));
        MaxSizeBeforeGrow = std::ceil(LoadFactor * Buckets.size());
    }

    ConcurrentInsertOnlyHashMap(Hash hash = {}, KeyEqual key_equal = {}, KeyFactory key_factory = {})
            : ConcurrentInsertOnlyHashMap(8, std::move(hash), std::move(key_equal), std::move(key_factory)) {}

    ~ConcurrentInsertOnlyHashMap() {
        foreachEntry([](auto* BL) {
            BL->Next = nullptr;
            delete BL;
        });
    }

    void setNumLanes(std::size_t) {
        /* ignore. we're not using lanes for now, but the internal impl might change in the future */
    }

    /** @brief Create a fresh node initialized with the given value and a
     * default-constructed key.
     *
     * The ownership of the returned node given to the caller.
     */
    node_type node(const T& V) {
        return {std::make_unique<BucketList>(BucketList{{Key{}, V}, nullptr})};
    }

    /** @brief Checks if the map contains an element with the given key.
     *
     * The search is done concurrently with possible insertion of the
     * searched key. If return true, then there is definitely an element
     * with the specified key, if return false then there was no such
     * element when the search began.
     */
    template <class K>
    bool weakContains(const lane_id, const K& X) const {
        const size_t HashValue = Hasher(X);
        const auto Guard = RW_Bucket.read_lock();
        const size_t Bucket = HashValue % Buckets.size();

        BucketList* L = Buckets[Bucket].load(std::memory_order_consume);
        while (L != nullptr) {
            if (EqualTo(L->Value.first, X)) {
                // found the key
                return true;
            }
            L = L->Next;
        }
        return false;
    }

    /**
     * @brief Inserts in-place if the key is not mapped, does nothing if the key already exists.
     *
     * @param H is the access lane.
     *
     * @param N is a node initialized with the mapped value to insert.
     *
     * @param Xs are arguments to forward to the hasher, the comparator and and
     * the constructor of the key.
     *
     *
     * Be Careful: the inserted node becomes available to concurrent lanes as
     * soon as it is inserted, thus concurrent lanes may access the inserted
     * value even before the inserting lane returns from this function.
     * This is the reason why the inserting lane must prepare the inserted
     * node's mapped value prior to calling this function.
     *
     * Be Careful: the given node remains the ownership of the caller unless
     * the returned couple second member is true.
     *
     * Be Careful: the given node may not be inserted if the key already
     * exists.  The caller is in charge of handling that case and either
     * dispose of the node or save it for the next insertion operation.
     *
     * Be Careful: Once the given node is actually inserted, its ownership is
     * transfered to the hash-map. However it remains valid.
     *
     * If the key that compares equal to arguments Xs exists, then nothing is
     * inserted. The returned value is the couple of the pointer to the
     * existing value and the false boolean value.
     *
     * If the key that compares equal to arguments Xs does not exist, then the
     * node N is updated with the key constructed from Xs, and inserted in the
     * hash-map. The returned value is the couple of the pointer to the
     * inserted value and the true boolean value.
     *
     */
    template <class... Args>
    std::pair<const value_type*, bool> get(const lane_id, node_type& Node, Args&&... Xs) {
        assert(Node && "invalid handle");
        assert(!Node->Next && "node is already part of a chain");

        // At any time a concurrent lane may insert the key before this lane.
        //
        // The synchronisation point is the atomic compare-and-exchange of the
        // head of the bucket list that must contain the inserted node.
        //
        // The insertion algorithm is as follow:
        //
        // 1) Compute the key hash from Xs.
        //
        // 2) Lock the lane, that also prevent concurrent lanes from growing of
        // the datastructure.
        //
        // 3) Determine the bucket where the element must be inserted.
        //
        // 4) Read the "last known head" of the bucket list. Other lanes
        // inserting in the same bucket may update the bucket head
        // concurrently.
        //
        // 5) Search the bucket list for the key by comparing with Xs starting
        // from the last known head. If it is not the first round of search,
        // then stop searching where the previous round of search started.
        //
        // 6) If the key is found return the couple of the value pointer and
        // false (to indicate that this lane did not insert the node N).
        //
        // 7) It the key is not found prepare N for insertion by updating its
        // key with Xs and chaining the last known head.
        //
        // 8) Try to exchange to last known head with N at the bucket head. The
        // atomic compare and exchange operation guarantees that it only
        // succeed if not other node was inserted in the bucket since we
        // searched it, otherwise it fails when another lane has concurrently
        // inserted a node in the same bucket.
        //
        // 9) If the atomic compare and exchange succeeded, the node has just
        // been inserted by this lane. From now-on other lanes can also see
        // the node. Return the couple of a pointer to the inserted value and
        // the true boolean.
        //
        // 10) If the atomic compare and exchange failed, another node has been
        // inserted by a concurrent lane in the same bucket. A new round of
        // search is required -> restart from step 4.
        //
        //
        // The datastructure is optionaly grown after step 9) before returning.

        // 1)
        const size_t HashValue = Hasher(std::forward<Args>(Xs)...);

        // 2) prevent the datastructure from growing. this can cause us to temporarily exceed our load factor
        auto read_lock = RW_Bucket.read_lock();

        // 3)
        const size_t Bucket = HashValue % Buckets.size();

        // 4)
        // the head of the bucket's list last time we checked
        BucketList* LastKnownHead = Buckets[Bucket].load(std::memory_order_relaxed);
        // the head of the bucket's list we already searched from
        BucketList* SearchedFrom = nullptr;

        // Loop until either the node is inserted or the key is found in the bucket.
        // Assuming bucket collisions are rare this loop is not executed more than once.
        do {
            // 5)
            // search the key in the bucket, stop where we already search at a
            // previous iteration.
            BucketList* L = LastKnownHead;
            while (L != SearchedFrom) {
                if (EqualTo(L->Value.first, std::forward<Args>(Xs)...)) {
                    // 6)
                    // found the key
                    Node->Next = nullptr;  // drop any refs to the chain that may have been collected
                    return {&L->Value, false};
                }
                L = L->Next;
            }
            SearchedFrom = LastKnownHead;

            // 7)
            // Not found in bucket, prepare node chaining.
            Node->Next = LastKnownHead;
            // The factory step could be done only once, but assuming bucket collisions are
            // rare this whole loop is not executed more than once.
            Factory.replace(const_cast<key_type&>(Node->Value.first), std::forward<Args>(Xs)...);

            // 8)
            // Try to insert the key in front of the bucket's list.
            // This operation also performs step 4) because LastKnownHead is
            // updated in the process.
            //
            // 19) Exchange failed? Concurrent insertion detected in this bucket, new round required.
        } while (!Buckets[Bucket].compare_exchange_strong(
                LastKnownHead, Node.get(), std::memory_order_release, std::memory_order_relaxed));

        // 9) inserted, now to see if we need to resize to satisfy our load-factor
        auto NewSize = ++Size;
        auto Value = &(Node->Value);
        Node.release();

        if (NewSize > MaxSizeBeforeGrow) {
            read_lock = {};  // release our read lock early to avoid deadlock
            grow();
        }

        return {Value, true};
    }

private:
    // The concurrent lanes manager.
    mutable ReadWriteLock RW_Bucket;

    /// Hash function.
    Hash Hasher;

    /// Atomic pointer to head bucket linked-list head.
    BucketVector Buckets;

    /// The Equal-to function.
    KeyEqual EqualTo;

    KeyFactory Factory;

    /// Current number of elements stored in the map.
    std::atomic<std::size_t> Size;

    /// Maximum size before the map should grow.
    std::size_t MaxSizeBeforeGrow;

    /// The load-factor of the map.
    double LoadFactor;

    // Grow the datastructure.
    // Must be called while owning the write lock
    void grow() {
        auto _ = RW_Bucket.write_lock();
        assert(MaxSizeBeforeGrow <= Size);

        // Compute the new number of buckets:
        // Chose a prime number of buckets that ensures the desired load factor
        // given the current number of elements in the map.
        const std::size_t CurrentSize = Size;
        const std::size_t NeededBucketCount = std::ceil(CurrentSize / LoadFactor);
        const std::size_t NewBucketCount = primeBucketSizing(NeededBucketCount);

        // Rehash, this operation is costly because it requires to scan
        // the existing elements, compute its hash to find its new bucket
        // and insert in the new bucket.
        //
        // Maybe concurrent lanes could help using some job-stealing algorithm.
        BucketVector NewBuckets(NewBucketCount);
        foreachEntry([&](BucketList* const E) {
            auto&& [K, _] = E->Value;
            auto& Bucket = NewBuckets[Hasher(K) % NewBuckets.size()];

            E->Next = Bucket.load(std::memory_order_relaxed);
            Bucket.store(E, std::memory_order_relaxed);
        });

        Buckets = std::move(NewBuckets);
        MaxSizeBeforeGrow = std::ceil(Buckets.size() * LoadFactor);
    }
};

}  // namespace souffle
