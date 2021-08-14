/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2013, 2014, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */

/************************************************************************
 *
 * @file RecordTable.h
 *
 * Data container implementing a map between records and their references.
 * Records are separated by arity, i.e., stored in different RecordMaps.
 *
 ***********************************************************************/

#pragma once

#include "souffle/RamTypes.h"
#include "souffle/datastructure/ConcurrentFlyweight.h"
#include "souffle/utility/span.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace souffle {

namespace details {

// Helper to unroll for loop
template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F&& f) {
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        constexpr_for<Start + Inc, End, Inc>(f);
    }
}

/// @brief The data-type of RamDomain records of any size.
using GenericRecord = std::vector<RamDomain>;

/// @brief The data-type of RamDomain records of specialized size.
template <std::size_t Arity>
using SpecializedRecord = std::array<RamDomain, Arity>;

/// @brief Hash function object for a RamDomain record.
struct GenericRecordHash {
    explicit GenericRecordHash(const std::size_t Arity) : Arity(Arity) {}

    const std::size_t Arity;
    std::hash<RamDomain> domainHash;

    template <typename T>
    std::size_t operator()(const T& Record) const {
        std::size_t Seed = 0;
        for (std::size_t I = 0; I < Arity; ++I) {
            Seed ^= domainHash(Record[I]) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2);
        }
        return Seed;
    }
};

template <std::size_t Arity>
struct SpecializedRecordHash {
    std::hash<RamDomain> DomainHash;

    template <typename T>
    std::size_t operator()(const T& Record) const {
        std::size_t Seed = 0;
        constexpr_for<0, Arity, 1>(
                [&](auto I) { Seed ^= DomainHash(Record[I]) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2); });
        return Seed;
    }
};

/// @brief Equality function object for RamDomain records.
struct GenericRecordEqual {
    explicit GenericRecordEqual(const std::size_t Arity) : Arity(Arity) {}

    const std::size_t Arity;

    template <typename T, typename U>
    bool operator()(const T& A, const U& B) const {
        return (std::memcmp(A.data(), B.data(), Arity * sizeof(RamDomain)) == 0);
    }
};

template <std::size_t Arity>
struct SpecializedRecordEqual {
    template <typename T, typename U>
    bool operator()(const T& A, const U& B) const {
        constexpr auto Len = Arity * sizeof(RamDomain);
        return (std::memcmp(A.data(), B.data(), Len) == 0);
    }
};

/// @brief Compare function object for RamDomain records.
struct GenericRecordCmp {
    explicit GenericRecordCmp(std::size_t Arity) : Arity(Arity) {}

    const std::size_t Arity;

    template <typename T, typename U>
    int operator()(const T& A, const U& B) const {
        return std::memcmp(A.data(), B.data(), Arity * sizeof(RamDomain));
    }
};

template <std::size_t Arity>
struct SpecializedRecordCmp {
    template <typename T, typename U>
    bool operator()(const T& A, const U& B) const {
        constexpr std::size_t Len = Arity * sizeof(RamDomain);
        return std::memcmp(A.data(), B.data(), Len) < 0;
    }
};

/// @brief Factory of RamDomain record.
struct GenericRecordFactory {
    using value_type = GenericRecord;
    using pointer = GenericRecord*;
    using reference = GenericRecord&;

    explicit GenericRecordFactory(const std::size_t Arity) : Arity(Arity) {}

    const std::size_t Arity;

    reference replace(reference Place, std::vector<RamDomain> V) {
        assert(V.size() == Arity);
        Place = std::move(V);
        return Place;
    }

    reference replace(reference Place, const span<RamDomain const>& V) {
        assert(V.size() == Arity);
        return replace(Place, V.data());
    }

    reference replace(reference Place, const RamDomain* V) {
        Place.resize(Arity);
        std::copy_n(V, Arity, Place.begin());
        return Place;
    }
};

template <std::size_t Arity>
struct SpecializedRecordFactory {
    using value_type = SpecializedRecord<Arity>;
    using pointer = SpecializedRecord<Arity>*;
    using reference = SpecializedRecord<Arity>&;

    reference replace(reference Place, const span<RamDomain const, Arity>& V) {
        return replace(Place, V.data());
    }

    reference replace(reference Place, const RamDomain* V) {
        constexpr std::size_t Len = Arity * sizeof(RamDomain);
        std::memmove(Place.data(), V, Len);
        return Place;
    }
};

}  // namespace details

/** @brief Interface of bidirectional mappping between records and record references. */
class RecordMap {
public:
    virtual ~RecordMap() = default;

    virtual void setNumLanes(const std::size_t NumLanes) = 0;
    virtual size_t arity() const = 0;
    virtual const RamDomain* unpack(RamDomain Index) const = 0;
    virtual RamDomain pack(const RamDomain* Tuple) = 0;

    virtual RamDomain pack(span<RamDomain const> Tuple) {
        assert(arity() == Tuple.size());
        return pack(Tuple.data());
    }
};

/** @brief Bidirectional mappping between records and record references, for any record arity. */
class GenericRecordMap : public RecordMap,
                         protected FlyweightImpl<details::GenericRecord, details::GenericRecordHash,
                                 details::GenericRecordEqual, details::GenericRecordFactory> {
    using Base = FlyweightImpl<details::GenericRecord, details::GenericRecordHash,
            details::GenericRecordEqual, details::GenericRecordFactory>;

    const std::size_t Arity;

public:
    explicit GenericRecordMap(const std::size_t lane_count, const std::size_t arity)
            : Base(lane_count, 8, true, details::GenericRecordHash(arity), details::GenericRecordEqual(arity),
                      details::GenericRecordFactory(arity)),
              Arity(arity) {}

    void setNumLanes(const std::size_t NumLanes) override {
        Base::setNumLanes(NumLanes);
    }

    size_t arity() const override {
        return Arity;
    }

    using RecordMap::pack;

    /** @brief converts record to a record reference */
    RamDomain pack(const RamDomain* Tuple) override {
        return findOrInsert(span<RamDomain const>{Tuple, Tuple + Arity}).first;
    }

    /** @brief convert record reference to a record pointer */
    const RamDomain* unpack(RamDomain Index) const override {
        return fetch(Index).data();
    }
};

/** @brief Bidirectional mappping between records and record references, specialized for a record arity. */
template <std::size_t Arity>
class SpecializedRecordMap
        : public RecordMap,
          protected FlyweightImpl<details::SpecializedRecord<Arity>, details::SpecializedRecordHash<Arity>,
                  details::SpecializedRecordEqual<Arity>, details::SpecializedRecordFactory<Arity>> {
    using Record = details::SpecializedRecord<Arity>;
    using RecordView = span<RamDomain const, Arity>;
    using RecordHash = details::SpecializedRecordHash<Arity>;
    using RecordEqual = details::SpecializedRecordEqual<Arity>;
    using RecordFactory = details::SpecializedRecordFactory<Arity>;
    using Base = FlyweightImpl<Record, RecordHash, RecordEqual, RecordFactory>;

public:
    SpecializedRecordMap(const std::size_t LaneCount)
            : Base(LaneCount, 8, true, RecordHash(), RecordEqual(), RecordFactory()) {}

    void setNumLanes(const std::size_t NumLanes) override {
        Base::setNumLanes(NumLanes);
    }

    size_t arity() const override {
        return Arity;
    }

    using RecordMap::pack;

    RamDomain pack(const RecordView& Tuple) {
        return Base::findOrInsert(Tuple).first;
    }

    /** @brief converts record to a record reference */
    RamDomain pack(const RamDomain* Tuple) override {
        return Base::findOrInsert(RecordView{Tuple, Tuple + Arity}).first;
    }

    /** @brief convert record reference to a record pointer */
    const RamDomain* unpack(RamDomain Index) const override {
        return Base::fetch(Index).data();
    }
};

/** Record map specialized for arity 0 */
template <>
class SpecializedRecordMap<0> : public RecordMap {
    // The empty record always at index 1
    // The index 0 of each map is reserved.
    static constexpr RamDomain EmptyRecordIndex = 1;

    // To comply with previous behavior, the empty record
    // has no data:
    const RamDomain* EmptyRecordData = nullptr;

public:
    SpecializedRecordMap(const std::size_t /* LaneCount */) {}

    void setNumLanes(const std::size_t) override {}

    size_t arity() const override {
        return 0;
    }

    using RecordMap::pack;

    RamDomain pack(span<RamDomain const, 0>) {
        return pack(EmptyRecordData);
    }

    /** @brief converts record to a record reference */
    RamDomain pack(const RamDomain*) override {
        return EmptyRecordIndex;
    }

    /** @brief convert record reference to a record pointer */
    const RamDomain* unpack(RamDomain Index) const override {
        assert(Index == EmptyRecordIndex);
        return EmptyRecordData;
    }
};

/** The interface of any Record Table. */
class RecordTableInterface {
public:
    virtual ~RecordTableInterface() = default;

    virtual void setNumLanes(const std::size_t NumLanes) = 0;

    virtual RamDomain pack(const RamDomain* Tuple, const std::size_t Arity) = 0;

    virtual const RamDomain* unpack(const RamDomain Ref, const std::size_t Arity) const = 0;
};

/** A concurrent Record Table with some specialized record maps. */
template <std::size_t... SpecializedArities>
class SpecializedRecordTable : public RecordTableInterface {
    // Resizing for a new arity / changing lane count should be extremely rare.
    // Most cases we're just doing read-only (from the arity dispatcher's POV).
    mutable ReadWriteLock rw_maps;
    size_t lanes;
    std::vector<std::unique_ptr<RecordMap>> maps{std::max<size_t>({0, (SpecializedArities + 1)...})};

public:
    /** @brief Construct a record table with the number of concurrent access lanes. */
    SpecializedRecordTable(size_t lanes = 1) : lanes(lanes) {
        ((maps[SpecializedArities] = std::make_unique<SpecializedRecordMap<SpecializedArities>>(lanes)), ...);
    }

    /**
     * @brief set the number of concurrent access lanes.
     * Not thread-safe, use only when the datastructure is not being used.
     */
    void setNumLanes(const std::size_t n) override {
        lanes = n;
        for (auto& m : maps)
            m->setNumLanes(n);
    }

    /** @brief convert record to record reference */
    RamDomain pack(const RamDomain* Tuple, const std::size_t Arity) override {
        auto&& [_, m] = lookupMap(Arity);
        return m.pack(Tuple);
    }

    /** @brief convert record reference to a record */
    const RamDomain* unpack(const RamDomain Ref, const std::size_t Arity) const override {
        auto&& [_, m] = existingMap(Arity);
        return m.unpack(Ref);
    }

private:
    /** @brief lookup RecordMap for a given arity; the map for that arity must exist. */
    std::pair<ReadWriteLock::ReadLock, RecordMap&> existingMap(size_t arity) const {
        auto lock = rw_maps.read_lock();
        assert(arity < maps.size() && "Lookup for an arity while there is no record for that arity.");
        auto& m = maps[arity];
        assert(m && "Lookup for an arity while there is no record for that arity.");
        return {std::move(lock), *m};
    }

    /** @brief lookup RecordMap for a given arity; if it does not exist, create new RecordMap */
    std::pair<ReadWriteLock::ReadLock, RecordMap&> lookupMap(size_t arity) {
        auto lookup = [&]() { return arity < maps.size() ? maps[arity].get() : nullptr; };

        {
            auto lock = rw_maps.read_lock();
            if (auto* m = lookup()) return {std::move(lock), *m};
        }

        auto lock = rw_maps.write_lock();
        if (auto* m = lookup()) return {std::move(lock), *m};

        maps.resize(std::max(maps.size(), arity + 1));
        maps[arity] = std::make_unique<GenericRecordMap>(lanes, arity);
        auto& m = maps[arity];
        return {std::move(lock), *m};
    }
};

/** Default record table uses specialized record maps for arities 0 to 12. */
using RecordTable = SpecializedRecordTable<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12>;

/** @brief helper to convert tuple to record reference for the synthesiser */
template <class RecordTableT, std::size_t Arity>
RamDomain pack(RecordTableT&& recordTab, Tuple<RamDomain, Arity> const& tuple) {
    return recordTab.pack(tuple.data(), Arity);
}

/** @brief helper to convert tuple to record reference for the synthesiser */
template <class RecordTableT, std::size_t Arity>
RamDomain pack(RecordTableT&& recordTab, span<const RamDomain, Arity> tuple) {
    return recordTab.pack(tuple.data(), Arity);
}

}  // namespace souffle
