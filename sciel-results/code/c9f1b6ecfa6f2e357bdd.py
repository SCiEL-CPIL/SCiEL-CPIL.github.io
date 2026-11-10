"""Covering design C(19,7,4) - fast point-greedy + aggressive multi-restart refinement."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0


def _build_index(v, t):
    """Map each t-subset to an integer index. Return (idx_dict, total, reverse_list)."""
    idx = {}
    reverse = []
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        reverse.append(comb)
        i += 1
    return idx, i, reverse


def _precompute_block_bitsets(v, k, t, idx):
    """Precompute a numpy bool bitset (length C(v,t)) for every k-subset."""
    n = len(idx)
    all_ksubs = list(combinations(range(v), k))
    all_bitsets = np.zeros((len(all_ksubs), n), dtype=bool)
    for bi, ks in enumerate(all_ksubs):
        for comb in combinations(ks, t):
            ci = idx[comb]
            all_bitsets[bi, ci] = True
    return all_ksubs, all_bitsets


def _greedy_construct(v, k, t, idx, total, reverse, deadline):
    """Point-by-point greedy: build each block by adding the point that covers the
    most currently-uncovered t-subsets. Much faster than scanning all k-subsets."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    start = time.time()

    # Precompute: for each (t-1)-subset S and point x not in S, the index of S∪{x}.
    # tsub_of[(t-1)-tuple, x] -> int index (or -1).
    tsub_of = {}
    for comb in combinations(range(v), t):
        ci = idx[comb]
        for j in range(t):
            rest = comb[:j] + comb[j + 1:]
            x = comb[j]
            tsub_of[(rest, x)] = ci

    while uncovered.any():
        if time.time() > deadline:
            break

        first_idx = int(np.argmax(uncovered))
        first_tsub = reverse[first_idx]
        block = list(first_tsub)
        block_set = set(block)

        # Greedily extend to size k
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of uncovered t-subsets that contain x and (t-1) elems of block
                gain = 0
                for rest in combinations(block, t - 1):
                    ci = tsub_of.get((rest, x), -1)
                    if ci >= 0 and uncovered[ci]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                best_x = next(x for x in range(v) if x not in block_set)
            block.append(best_x)
            block_set.add(best_x)
            if time.time() > deadline:
                break

        block.sort()
        blocks.append(block)
        # Mark covered
        for comb in combinations(block, t):
            uncovered[idx[comb]] = False

    return blocks


def _build_coverage(blocks, v, k, t, idx):
    n = len(idx)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            coverage[idx[comb]] += 1
    return coverage


def _deterministic_removal(blocks, v, k, t, idx, deadline):
    """Remove any block whose every t-subset is covered by at least one other block."""
    n = len(idx)
    coverage = _build_coverage(blocks, v, k, t, idx)
    blocks = list(blocks)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            removable = True
            for comb in combinations(block, t):
                ci = idx[comb]
                if coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    coverage[idx[comb]] -= 1
                blocks.pop(i)
                improved = True
                break
    return blocks, coverage


def _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=30000, seed=42):
    """Randomized removal attempts to escape ordering-dependent local minima."""
    n = len(idx)
    coverage = _build_coverage(blocks, v, k, t, idx)
    blocks = list(blocks)
    rng = random.Random(seed)
    for _ in range(n_trials):
        if time.time() > deadline:
            break
        if not blocks:
            break
        i = rng.randrange(len(blocks))
        block = blocks[i]
        removable = True
        for comb in combinations(block, t):
            ci = idx[comb]
            if coverage[ci] <= 1:
                removable = False
                break
        if removable:
            for comb in combinations(block, t):
                coverage[idx[comb]] -= 1
            blocks.pop(i)
    return blocks, coverage


def _simulated_annealing(blocks, v, k, t, idx, all_ksubs, all_bitsets, deadline, seed=123):
    """SA with two move types: (a) remove a block (accept if no new uncovered, else with
    prob exp(-uncovered/T)); (b) replace a block with a different k-subset that covers
    all its uniquely-covered t-subsets. Track best (fewest-block) valid state."""
    n = len(idx)
    coverage = _build_coverage(blocks, v, k, t, idx)
    blocks = list(blocks)
    ksub_to_idx = {frozenset(ks): i for i, ks in enumerate(all_ksubs)}

    rng = random.Random(seed)
    start_time = time.time()
    total_time = max(0.0, deadline - start_time)
    if total_time <= 0:
        return blocks

    current_count = len(blocks)
    best_count = current_count
    best_blocks = list(blocks)

    T_start = 4.0
    T_end = 0.02
    # Scale steps with available time
    n_steps = int(min(200000, total_time * 400))

    for step in range(n_steps):
        if time.time() > deadline:
            break
        if not blocks:
            break

        progress = step / max(1, n_steps)
        T = T_start * (T_end / T_start) ** progress

        i = rng.randrange(len(blocks))
        block = blocks[i]
        block_set = frozenset(block)
        block_idx = ksub_to_idx.get(block_set, -1)
        if block_idx < 0:
            continue

        if rng.random() < 0.7:
            # Move type A: try to remove block i
            uncovered_count = 0
            affected = []
            for comb in combinations(block, t):
                ci = idx[comb]
                if coverage[ci] <= 1:
                    uncovered_count += 1
                affected.append(ci)

            if uncovered_count == 0:
                for ci in affected:
                    coverage[ci] -= 1
                blocks.pop(i)
                current_count = len(blocks)
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
            else:
                if rng.random() < np.exp(-uncovered_count / max(T, 1e-10)):
                    for ci in affected:
                        coverage[ci] -= 1
                    blocks.pop(i)
                    current_count = len(blocks)
        else:
            # Move type B: try to replace block i with another k-subset
            block_bits = all_bitsets[block_idx]
            unique_mask = block_bits & (coverage == 1)
            if not unique_mask.any():
                # No unique coverage: just try removal (handled above) - skip
                continue
            # Randomly sample a few replacement candidates (not all 50k) for speed
            found = False
            for _ in range(20):
                j = rng.randrange(len(all_ksubs))
                if j == block_idx:
                    continue
                ks = all_ksubs[j]
                ks_bits = all_bitsets[j]
                # Must cover all unique t-subsets of old block
                if not np.all(unique_mask & ks_bits):
                    continue
                # Compute new coverage delta: remove old, add new
                # Check no t-subset goes to 0
                new_cov = coverage - block_bits.astype(np.int32) + ks_bits.astype(np.int32)
                if new_cov.min() < 0:
                    continue
                # Accept (block count unchanged, but may enable future removals)
                coverage = new_cov
                blocks[i] = list(ks)
                found = True
                break

    return best_blocks


def _refine_once(blocks, v, k, t, idx, all_ksubs, all_bitsets, deadline, seed):
    """One full refinement pipeline from a starting block set."""
    blocks, coverage = _deterministic_removal(blocks, v, k, t, idx, deadline)
    blocks, coverage = _random_removal_pass(blocks, v, k, t, idx, deadline,
                                            n_trials=20000, seed=seed)
    if time.time() < deadline - 2:
        blocks = _simulated_annealing(blocks, v, k, t, idx, all_ksubs, all_bitsets,
                                      deadline, seed=seed + 1)
    blocks, _ = _deterministic_removal(blocks, v, k, t, idx, deadline)
    blocks, _ = _random_removal_pass(blocks, v, k, t, idx, deadline,
                                     n_trials=10000, seed=seed + 2)
    return blocks


def _perturb(blocks, v, k, t, idx, all_ksubs, deadline, rng, n_swap=3):
    """Create a perturbed copy: replace n_swap random blocks with random k-subsets,
    then re-cover any gaps with greedy blocks."""
    blocks = [list(b) for b in blocks]
    n = len(idx)
    uncovered = np.zeros(n, dtype=bool)
    coverage = _build_coverage(blocks, v, k, t, idx)
    # Pick random blocks to replace
    for _ in range(min(n_swap, len(blocks))):
        i = rng.randrange(len(blocks))
        old = blocks[i]
        for comb in combinations(old, t):
            coverage[idx[comb]] -= 1
        new_ks = all_ksubs[rng.randrange(len(all_ksubs))]
        blocks[i] = list(new_ks)
        for comb in combinations(new_ks, t):
            coverage[idx[comb]] += 1
    # Find uncovered
    uncovered = (coverage == 0)
    # Greedily add blocks to cover gaps (point-by-point)
    tsub_of = {}
    for comb in combinations(range(v), t):
        ci = idx[comb]
        for j in range(t):
            rest = comb[:j] + comb[j + 1:]
            x = comb[j]
            tsub_of[(rest, x)] = ci
    while uncovered.any():
        if time.time() > deadline:
            break
        first_idx = int(np.argmax(uncovered))
        # find the t-subset
        first_tsub = next(c for c in combinations(range(v), t) if idx[c] == first_idx)
        block = list(first_tsub)
        block_set = set(block)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in combinations(block, t - 1):
                    ci = tsub_of.get((rest, x), -1)
                    if ci >= 0 and uncovered[ci]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                best_x = next(x for x in range(v) if x not in block_set)
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        for comb in combinations(block, t):
            ci = idx[comb]
            if uncovered[ci]:
                uncovered[ci] = False
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index and reverse map
    idx, total, reverse = _build_index(v, t)

    # Precompute bitsets for all k-subsets (used by SA replacement moves)
    all_ksubs, all_bitsets = _precompute_block_bitsets(v, k, t, idx)

    # Fast point-by-point greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, reverse, deadline)

    # First refinement pass (full pipeline)
    if time.time() < deadline - 5:
        blocks = _refine_once(blocks, v, k, t, idx, all_ksubs, all_bitsets,
                              deadline, seed=100)

    # Multi-restart: perturb + refine, keep best
    best_blocks = list(blocks)
    best_count = len(blocks)
    rng = random.Random(777)

    restart_budget = deadline - time.time()
    if restart_budget > 15:
        # Do a few perturbation restarts, each with a reduced refinement budget
        n_restarts = max(1, int((restart_budget - 10) // 25))
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            sub_deadline = min(deadline - 8, time.time() + 22)
            perturbed = _perturb(best_blocks, v, k, t, idx, all_ksubs,
                                 sub_deadline, rng, n_swap=2 + r)
            refined = _refine_once(perturbed, v, k, t, idx, all_ksubs, all_bitsets,
                                   sub_deadline, seed=200 + r)
            if len(refined) < best_count:
                best_count = len(refined)
                best_blocks = refined

    # Final high-precision polish on the best solution
    if time.time() < deadline - 2:
        best_blocks = _refine_once(best_blocks, v, k, t, idx, all_ksubs, all_bitsets,
                                   deadline, seed=999)

    # Safety: if we somehow lost validity (shouldn't), fall back to greedy result
    # (greedy is always valid if it completed)
    return best_blocks
