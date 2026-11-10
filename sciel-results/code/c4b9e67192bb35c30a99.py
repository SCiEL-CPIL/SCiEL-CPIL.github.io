"""Covering design C(19,7,4) - bitset greedy + removal + simulated annealing + swap."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0


def _build_index(v, t):
    """Map each t-subset to an integer index. Return (idx_dict, total)."""
    idx = {}
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        i += 1
    return idx, i


def _precompute_bitsets(v, k, t, idx, total):
    """Precompute bitsets for all k-subsets and reverse index."""
    all_ksubs = list(combinations(range(v), k))
    n = len(all_ksubs)
    all_bitsets = np.zeros((n, total), dtype=bool)
    
    tsub_to_ksubs = [[] for _ in range(total)]
    
    for i, ks in enumerate(all_ksubs):
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                all_bitsets[i, ci] = True
                tsub_to_ksubs[ci].append(i)
    
    return all_ksubs, all_bitsets, tsub_to_ksubs


def _greedy_construct(v, k, t, idx, total, all_ksubs, all_bitsets, tsub_to_ksubs, deadline):
    """Fast greedy using precomputed bitsets."""
    uncovered = np.ones(total, dtype=bool)
    blocks = []
    used = np.zeros(len(all_ksubs), dtype=bool)
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        first_idx = int(np.argmax(uncovered))
        candidates = tsub_to_ksubs[first_idx]
        
        best_i = -1
        best_gain = -1
        for ci in candidates:
            if used[ci]:
                continue
            gain = int(np.sum(all_bitsets[ci] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_i = ci
        
        if best_i < 0:
            break
        
        blocks.append(list(all_ksubs[best_i]))
        used[best_i] = True
        uncovered &= ~all_bitsets[best_i]
    
    return blocks


def _build_coverage(blocks, v, k, t, idx, total):
    """Build coverage count array."""
    coverage = np.zeros(total, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                coverage[ci] += 1
    return coverage


def _improve_blocks(blocks, v, k, t, idx, total, deadline):
    """Deterministic removal pass."""
    coverage = _build_coverage(blocks, v, k, t, idx, total)
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
                ci = idx.get(comb, -1)
                if ci >= 0 and coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        coverage[ci] -= 1
                blocks.pop(i)
                improved = True
                break
    return blocks


def _random_removal_pass(blocks, v, k, t, idx, total, deadline, n_trials=20000):
    """Randomized removal attempts."""
    coverage = _build_coverage(blocks, v, k, t, idx, total)
    blocks = list(blocks)
    rng = random.Random(42)
    for _ in range(n_trials):
        if time.time() > deadline:
            break
        if not blocks:
            break
        i = rng.randrange(len(blocks))
        block = blocks[i]
        removable = True
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0 and coverage[ci] <= 1:
                removable = False
                break
        if removable:
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
                    coverage[ci] -= 1
            blocks.pop(i)
    return blocks


def _block_swap(blocks, v, k, t, idx, total, all_ksubs, all_bitsets, deadline):
    """Try to replace blocks with better alternatives."""
    coverage = _build_coverage(blocks, v, k, t, idx, total)
    blocks = list(blocks)
    
    ksub_to_idx = {frozenset(ks): i for i, ks in enumerate(all_ksubs)}
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            block_set = frozenset(block)
            block_idx = ksub_to_idx.get(block_set, -1)
            if block_idx < 0:
                continue
            block_bits = all_bitsets[block_idx]
            
            # Find t-subsets uniquely covered by this block
            unique_mask = block_bits & (coverage == 1)
            if not unique_mask.any():
                continue
            
            # Try to find another block that covers all unique t-subsets
            best_j = -1
            best_overlap = -1
            for j in range(len(all_ksubs)):
                if j == block_idx:
                    continue
                ks_bits = all_bitsets[j]
                if not np.all(unique_mask & ks_bits):
                    continue
                overlap = int(np.sum(block_bits & ks_bits))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_j = j
            
            if best_j >= 0:
                old_bits = all_bitsets[block_idx]
                new_bits = all_bitsets[best_j]
                coverage -= old_bits.astype(np.int32)
                coverage += new_bits.astype(np.int32)
                blocks[i] = list(all_ksubs[best_j])
                improved = True
    
    return blocks


def _simulated_annealing(blocks, v, k, t, idx, total, all_ksubs, all_bitsets, deadline):
    """Simulated annealing with removals and additions."""
    coverage = _build_coverage(blocks, v, k, t, idx, total)
    blocks = list(blocks)
    
    rng = random.Random(123)
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return blocks
    
    current_count = len(blocks)
    best_count = current_count
    best_blocks = list(blocks)
    
    T_start = 5.0
    T_end = 0.01
    n_steps = int(min(120000, total_time * 400))
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(1, n_steps)
        T = T_start * (T_end / T_start) ** progress
        
        if blocks and rng.random() < 0.7:
            # Removal move
            i = rng.randrange(len(blocks))
            block = blocks[i]
            uncovered_count = 0
            affected = []
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
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
            # Addition move
            uncovered_mask = (coverage == 0)
            if not uncovered_mask.any():
                continue
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                continue
            pick = int(unc_idx[rng.randrange(len(unc_idx))])
            
            best_i = -1
            best_gain = 0
            for j in range(len(all_ksubs)):
                bits = all_bitsets[j]
                if not bits[pick]:
                    continue
                gain = int(np.sum(bits & uncovered_mask))
                if gain > best_gain:
                    best_gain = gain
                    best_i = j
            
            if best_i >= 0:
                blocks.append(list(all_ksubs[best_i]))
                coverage += all_bitsets[best_i].astype(np.int32)
                current_count = len(blocks)
    
    return best_blocks


def _pairwise_swap(blocks, v, k, t, idx, total, all_ksubs, all_bitsets, deadline):
    """Try to replace two blocks with one that covers all their unique t-subsets."""
    coverage = _build_coverage(blocks, v, k, t, idx, total)
    blocks = list(blocks)
    
    ksub_to_idx = {frozenset(ks): i for i, ks in enumerate(all_ksubs)}
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(blocks)):
                if time.time() > deadline:
                    break
                block_i = frozenset(blocks[i])
                block_j = frozenset(blocks[j])
                idx_i = ksub_to_idx.get(block_i, -1)
                idx_j = ksub_to_idx.get(block_j, -1)
                if idx_i < 0 or idx_j < 0:
                    continue
                
                bits_i = all_bitsets[idx_i]
                bits_j = all_bitsets[idx_j]
                
                # Unique t-subsets of block i (not covered by block j and coverage==1)
                unique_i = bits_i & (coverage == 1) & ~bits_j
                # Unique t-subsets of block j (not covered by block i and coverage==1)
                unique_j = bits_j & (coverage == 1) & ~bits_i
                
                if not unique_i.any() and not unique_j.any():
                    # Both blocks have no unique t-subsets relative to each other
                    # Try to remove one
                    for ci in np.where(bits_i)[0]:
                        coverage[ci] -= 1
                    blocks.pop(i)
                    if j > i:
                        j -= 1
                    improved = True
                    break
                
                # Try to find a single block that covers all unique t-subsets from both
                required = unique_i | unique_j
                if required.any():
                    best_k = -1
                    for k in range(len(all_ksubs)):
                        if k == idx_i or k == idx_j:
                            continue
                        bits_k = all_bitsets[k]
                        if np.all(required & bits_k):
                            best_k = k
                            break
                    
                    if best_k >= 0:
                        # Replace both with one
                        coverage -= bits_i.astype(np.int32)
                        coverage -= bits_j.astype(np.int32)
                        coverage += all_bitsets[best_k].astype(np.int32)
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(list(all_ksubs[best_k]))
                        improved = True
                        break
            if improved:
                break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    idx, total = _build_index(v, t)
    all_ksubs, all_bitsets, tsub_to_ksubs = _precompute_bitsets(v, k, t, idx, total)
    
    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, all_ksubs, all_bitsets, tsub_to_ksubs, deadline)
    
    # Deterministic improvement
    if time.time() < deadline - 15:
        blocks = _improve_blocks(blocks, v, k, t, idx, total, deadline)
    
    # Randomized removal
    if time.time() < deadline - 12:
        blocks = _random_removal_pass(blocks, v, k, t, idx, total, deadline, n_trials=30000)
    
    # Block swap
    if time.time() < deadline - 10:
        blocks = _block_swap(blocks, v, k, t, idx, total, all_ksubs, all_bitsets, deadline)
    
    # Pairwise swap
    if time.time() < deadline - 8:
        blocks = _pairwise_swap(blocks, v, k, t, idx, total, all_ksubs, all_bitsets, deadline)
    
    # Simulated annealing
    if time.time() < deadline - 5:
        blocks = _simulated_annealing(blocks, v, k, t, idx, total, all_ksubs, all_bitsets, deadline)
    
    # Final cleanup
    if time.time() < deadline - 2:
        blocks = _improve_blocks(blocks, v, k, t, idx, total, deadline)
        blocks = _random_removal_pass(blocks, v, k, t, idx, total, deadline, n_trials=10000)
    
    # High-precision polish
    if time.time() < deadline - 0.5:
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            blocks = _improve_blocks(blocks, v, k, t, idx, total, deadline)
            blocks = _random_removal_pass(blocks, v, k, t, idx, total, deadline, n_trials=15000)
    
    return blocks
