import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # ---------- Precomputation ----------
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)

    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)

    # block -> list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        block_t_indices.append([t_subset_to_idx[s] for s in itertools.combinations(block, t)])

    # t-subset -> list of blocks covering it
    t_subset_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # sets for fast membership
    block_t_sets = [set(x) for x in block_t_indices]

    # ---------- Greedy Set Cover ----------
    uncovered = [True] * num_t
    gain = [len(x) for x in block_t_indices]
    active = set()

    while True:
        if time.time() > deadline:
            break
        max_gain = max(gain)
        if max_gain == 0:
            break
        best = gain.index(max_gain)
        active.add(best)
        for t_idx in block_t_indices[best]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b in t_subset_blocks[t_idx]:
                    gain[b] -= 1

    # ---------- Cover counts ----------
    cover_count = [0] * num_t
    for b in active:
        for t_idx in block_t_indices[b]:
            cover_count[t_idx] += 1

    # ---------- Helper: deletion pass ----------
    def deletion_pass():
        if time.time() > deadline:
            return False
        order = list(active)
        random.shuffle(order)
        removed = False
        for b in order:
            if b not in active:
                continue
            ok = True
            for t_idx in block_t_indices[b]:
                if cover_count[t_idx] <= 1:
                    ok = False
                    break
            if ok:
                active.remove(b)
                for t_idx in block_t_indices[b]:
                    cover_count[t_idx] -= 1
                removed = True
        return removed

    # Multiple deletion passes
    for _ in range(4):
        if not deletion_pass():
            break

    # ---------- Systematic single-block swap ----------
    def unique_count(b):
        c = 0
        for t_idx in block_t_indices[b]:
            if cover_count[t_idx] == 1:
                c += 1
        return c

    def do_swap(b_out, b_in):
        active.add(b_in)
        active.remove(b_out)
        for t_idx in block_t_indices[b_out]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[b_in]:
            cover_count[t_idx] += 1

    # Sort by most unique first
    bwu = [(unique_count(b), b) for b in active if unique_count(b) > 0]
    bwu.sort(key=lambda x: -x[0])

    for _, b in bwu:
        if time.time() > deadline:
            break
        if b not in active:
            continue
        uniq = [t_idx for t_idx in block_t_indices[b] if cover_count[t_idx] == 1]
        if not uniq:
            continue
        first = uniq[0]
        candidates = [c for c in t_subset_blocks[first] if c not in active]
        if not candidates:
            continue
        valid = [c for c in candidates if all(u in block_t_sets[c] for u in uniq)]
        if not valid:
            continue
        # pick candidate maximizing overlap with already-covered (>=2) subsets
        best_c, best_s = None, -1
        for c in valid:
            s = 0
            for t_idx in block_t_indices[c]:
                if cover_count[t_idx] >= 2:
                    s += 1
            if s > best_s:
                best_s = s
                best_c = c
        if best_c is not None:
            do_swap(b, best_c)

    for _ in range(5):
        if not deletion_pass():
            break

    # ---------- 2-exchange local search ----------
    # Try replacing up to 2 active blocks with 2 new blocks to reduce total size
    def try_2_exchange():
        if time.time() > deadline:
            return False
        # candidate blocks: those not active
        non_active = [b for b in range(num_k) if b not in active]
        if not non_active:
            return False
        # For each active block, its unique subsets
        # We try pairs of active blocks that share no responsibility
        act_list = list(active)
        random.shuffle(act_list)
        # limit number of pairs tried
        max_pairs = 400
        tried = 0
        for i in range(len(act_list)):
            if time.time() > deadline:
                return False
            b1 = act_list[i]
            for j in range(i + 1, len(act_list)):
                tried += 1
                if tried > max_pairs:
                    return False
                b2 = act_list[j]
                # union of t-subsets covered by b1 and b2
                union = block_t_sets[b1] | block_t_sets[b2]
                # subsets that would become uncovered if we remove both
                need = [t_idx for t_idx in union if cover_count[t_idx] <= 2]
                if not need:
                    continue
                # find a pair of non-active blocks covering all of need
                # use first need element to get candidates
                cands = [c for c in t_subset_blocks[need[0]] if c not in active]
                if not cands:
                    continue
                found = False
                for c1 in cands:
                    if found:
                        break
                    for c2 in cands:
                        if c2 == c1:
                            continue
                        # check both together cover all need
                        ok = True
                        for t_idx in need:
                            if t_idx not in block_t_sets[c1] and t_idx not in block_t_sets[c2]:
                                ok = False
                                break
                        if ok:
                            # perform swap
                            active.add(c1)
                            active.add(c2)
                            active.remove(b1)
                            active.remove(b2)
                            for t_idx in block_t_indices[b1]:
                                cover_count[t_idx] -= 1
                            for t_idx in block_t_indices[b2]:
                                cover_count[t_idx] -= 1
                            for t_idx in block_t_indices[c1]:
                                cover_count[t_idx] += 1
                            for t_idx in block_t_indices[c2]:
                                cover_count[t_idx] += 1
                            found = True
                            break
                if found:
                    return True
        return False

    # Iterate 2-exchange + deletion
    for _ in range(6):
        if time.time() > deadline:
            break
        improved = try_2_exchange()
        for _ in range(3):
            if not deletion_pass():
                break
        if not improved:
            # try a few more single swaps then stop
            break

    # ---------- Final exhaustive polish ----------
    # Repeatedly attempt single-block swaps in a deterministic sweep for improvement
    for _ in range(3):
        if time.time() > deadline:
            break
        changed = False
        for b in list(active):
            if time.time() > deadline:
                break
            if b not in active:
                continue
            uniq = [t_idx for t_idx in block_t_indices[b] if cover_count[t_idx] == 1]
            if not uniq:
                continue
            first = uniq[0]
            cands = [c for c in t_subset_blocks[first] if c not in active]
            if not cands:
                continue
            valid = [c for c in cands if all(u in block_t_sets[c] for u in uniq)]
            if not valid:
                continue
            best_c, best_s = None, -1
            for c in valid:
                s = 0
                for t_idx in block_t_indices[c]:
                    if cover_count[t_idx] >= 2:
                        s += 1
                if s > best_s:
                    best_s = s
                    best_c = c
            if best_c is not None:
                do_swap(b, best_c)
                changed = True
        for _ in range(3):
            if not deletion_pass():
                break
        if not changed:
            break

    result = [list(all_k_subsets[b]) for b in active]
    return result
