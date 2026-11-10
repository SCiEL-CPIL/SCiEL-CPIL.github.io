"""Covering design C(19,8,5) with bitset-accelerated greedy + local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145

    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {cs: i for i, cs in enumerate(all_tsubsets)}

    # For each point, list of t-subset indices containing it
    point_to_tsubsets = [[] for _ in range(v)]
    for idx, cs in enumerate(all_tsubsets):
        for p in cs:
            point_to_tsubsets[p].append(idx)

    # For each k-subset, list of t-subset indices it covers
    all_ksubsets = list(itertools.combinations(range(v), k))
    ksubset_to_tsubsets = {}
    for ks in all_ksubsets:
        tsubs = []
        for ts in itertools.combinations(ks, t):
            tsubs.append(tsubset_to_idx[ts])
        ksubset_to_tsubsets[ks] = np.array(tsubs, dtype=np.int32)

    def greedy_cover(deadline_inner):
        """Greedy: pick k-subset covering most uncovered t-subsets each step."""
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []

        while uncovered.any() and time.time() < deadline_inner:
            # Find best k-subset
            best_count = -1
            best_ks = None
            for ks in all_ksubsets:
                tsubs = ksubset_to_tsubsets[ks]
                if len(tsubs) == 0:
                    continue
                cnt = uncovered[tsubs].sum()
                if cnt > best_count:
                    best_count = cnt
                    best_ks = ks
                    if best_count == num_tsubsets:
                        break

            if best_ks is None or best_count <= 0:
                break

            blocks.append(best_ks)
            uncovered[ksubset_to_tsubsets[best_ks]] = False

        return blocks, uncovered

    def local_improve(blocks, uncovered, deadline_inner):
        """Remove redundant blocks."""
        improved = True
        while improved and time.time() < deadline_inner:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline_inner:
                    break
                tsubs = ksubset_to_tsubsets[blocks[i]]
                redundant = True
                for ts_idx in tsubs:
                    ts = all_tsubsets[ts_idx]
                    found = False
                    for j, b in enumerate(blocks):
                        if j == i:
                            continue
                        if all(p in b for p in ts):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    improved = True
                    break
        return blocks

    best_blocks = None
    best_count = float('inf')

    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Slight randomization: shuffle k-subset order
        rng = random.Random(restart * 7919 + 1)
        shuffled_ksubsets = list(all_ksubsets)
        rng.shuffle(shuffled_ksubsets)

        # Redo greedy with shuffled order
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        deadline_inner = deadline - 3

        while uncovered.any() and time.time() < deadline_inner:
            best_count_step = -1
            best_ks = None
            for ks in shuffled_ksubsets:
                tsubs = ksubset_to_tsubsets[ks]
                if len(tsubs) == 0:
                    continue
                cnt = uncovered[tsubs].sum()
                if cnt > best_count_step:
                    best_count_step = cnt
                    best_ks = ks
                    if best_count_step == num_tsubsets:
                        break
            if best_ks is None or best_count_step <= 0:
                break
            blocks.append(best_ks)
            uncovered[ksubset_to_tsubsets[best_ks]] = False

        if not blocks or uncovered.any():
            continue

        local_improve(blocks, uncovered, deadline - 1)

        if not uncovered.any() and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        if best_count <= 264:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
