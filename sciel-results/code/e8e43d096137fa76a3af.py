"""Combined approach for covering design C(v,k,t).
Strategy:
- Fast greedy seed with vectorized gain computation.
- Index-based bookkeeping for t-subsets (like inspiration 1) for speed.
- 1-exchange local search with smarter candidate selection.
- Portfolio of randomized greedy + local search within a strict deadline.

Key ideas:
1. Precompute all t-subsets and their indices for O(1) lookup.
2. Greedy: pick the uncovered t-subset with the fewest candidate blocks (most constrained).
3. Local search: 1-exchange with metric-based candidate scoring.
4. Time management: explicit deadline guard with safety margin.
"""

import time
import itertools
import random
import sys
from collections import defaultdict

_DEADLINE = None
_TIME_LIMIT = 120
_MARGIN = 5


def _check_time():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    return time.time() >= _DEADLINE


def _precompute(v, k, t):
    """Precompute t-subset indices and block->t-subset mappings."""
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and which t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_t_list = []  # block_t_list[i] = list of t-subset indices covered by block i
    t_to_blocks = [[] for _ in range(num_t)]
    
    for i, b in enumerate(all_blocks):
        t_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_to_idx[ts]
            t_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_t_list.append(t_indices)
    
    return all_t_subsets, t_to_idx, all_blocks, block_t_list, t_to_blocks


def _greedy_cover(v, k, t, rng, time_limit_fn, precomp=None):
    """Fast greedy covering with index-based bookkeeping."""
    if precomp is None:
        precomp = _precompute(v, k, t)
    all_t_subsets, t_to_idx, all_blocks, block_t_list, t_to_blocks = precomp
    num_t = len(all_t_subsets)
    
    # uncovered_count[i] = number of blocks currently covering t-subset i (0 means uncovered)
    covered_count = [0] * num_t
    selected = []  # list of block indices
    selected_set = set()
    
    while True:
        if time_limit_fn():
            break
        
        # Find uncovered t-subsets
        uncovered_indices = [i for i in range(num_t) if covered_count[i] == 0]
        if not uncovered_indices:
            break
        
        # Pick the most constrained uncovered t-subset (fewest candidate blocks)
        # For diversity, sometimes pick randomly
        if rng.random() < 0.7:
            # Most constrained: fewest blocks can cover it
            best_idx = min(uncovered_indices, key=lambda i: len(t_to_blocks[i]))
        else:
            best_idx = rng.choice(uncovered_indices)
        
        # Find a block that covers this t-subset and maximizes new coverage
        candidate_blocks = t_to_blocks[best_idx]
        if not candidate_blocks:
            # Should not happen, but skip
            continue
        
        best_block = -1
        best_gain = -1
        best_tiebreak = -1
        
        for b_idx in candidate_blocks:
            if b_idx in selected_set:
                continue
            gain = 0
            for t_idx in block_t_list[b_idx]:
                if covered_count[t_idx] == 0:
                    gain += 1
            # Tie-break: prefer blocks with less overlap with already selected
            overlap = sum(1 for t_idx in block_t_list[b_idx] if covered_count[t_idx] > 0)
            if gain > best_gain or (gain == best_gain and overlap < best_tiebreak):
                best_gain = gain
                best_tiebreak = overlap
                best_block = b_idx
        
        if best_block == -1:
            # No improvement possible; pick a random candidate
            best_block = rng.choice(candidate_blocks)
        
        # Add the block
        selected.append(best_block)
        selected_set.add(best_block)
        for t_idx in block_t_list[best_block]:
            covered_count[t_idx] += 1
    
    # Convert block indices to actual blocks
    return [list(all_blocks[i]) for i in selected]


def _remove_redundant(v, k, t, blocks, time_limit_fn, precomp=None):
    """Remove redundant blocks."""
    if precomp is None:
        precomp = _precompute(v, k, t)
    all_t_subsets, t_to_idx, all_blocks, block_t_list, t_to_blocks = precomp
    num_t = len(all_t_subsets)
    
    # Convert blocks to indices
    block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    block_indices = []
    for b in blocks:
        key = tuple(sorted(b))
        if key in block_to_idx:
            block_indices.append(block_to_idx[key])
    
    if not block_indices:
        return []
    
    covered_count = [0] * num_t
    for b_idx in block_indices:
        for t_idx in block_t_list[b_idx]:
            covered_count[t_idx] += 1
    
    remaining = list(block_indices)
    remaining_set = set(remaining)
    
    changed = True
    while changed and not time_limit_fn():
        changed = False
        for i in range(len(remaining)):
            if time_limit_fn():
                break
            b_idx = remaining[i]
            removable = True
            for t_idx in block_t_list[b_idx]:
                if covered_count[t_idx] <= 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_t_list[b_idx]:
                    covered_count[t_idx] -= 1
                remaining.pop(i)
                remaining_set.discard(b_idx)
                changed = True
                break
    
    return [list(all_blocks[i]) for i in remaining]


def _one_exchange(v, k, t, blocks, rng, time_limit_fn, precomp=None):
    """1-exchange local search: try to swap one block for another to reduce count."""
    if precomp is None:
        precomp = _precompute(v, k, t)
    all_t_subsets, t_to_idx, all_blocks, block_t_list, t_to_blocks = precomp
    num_t = len(all_t_subsets)
    
    block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    block_indices = []
    for b in blocks:
        key = tuple(sorted(b))
        if key in block_to_idx:
            block_indices.append(block_to_idx[key])
    
    if not block_indices:
        return []
    
    current = list(block_indices)
    current_set = set(current)
    
    covered_count = [0] * num_t
    for b_idx in current:
        for t_idx in block_t_list[b_idx]:
            covered_count[t_idx] += 1
    
    improved = True
    iterations = 0
    max_iters = 500
    
    while improved and iterations < max_iters and not time_limit_fn():
        improved = False
        iterations += 1
        
        # First: remove any redundant blocks
        i = 0
        while i < len(current):
            if time_limit_fn():
                break
            b_idx = current[i]
            removable = True
            for t_idx in block_t_list[b_idx]:
                if covered_count[t_idx] <= 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_t_list[b_idx]:
                    covered_count[t_idx] -= 1
                current.pop(i)
                current_set.discard(b_idx)
                improved = True
            else:
                i += 1
        
        if improved:
            continue
        
        # Try 1-exchanges
        for i in range(len(current)):
            if time_limit_fn():
                break
            b_idx = current[i]
            
            # Find essential t-subsets (only covered by this block)
            essential = [t_idx for t_idx in block_t_list[b_idx] if covered_count[t_idx] == 1]
            
            if not essential:
                # No essential t-subsets: block is redundant
                for t_idx in block_t_list[b_idx]:
                    covered_count[t_idx] -= 1
                current.pop(i)
                current_set.discard(b_idx)
                improved = True
                break
            
            # Find candidate blocks that cover all essential t-subsets
            candidate_set = set(t_to_blocks[essential[0]])
            candidate_set -= current_set
            
            filtered = []
            for cand in candidate_set:
                all_covered = True
                for t_idx in essential:
                    if t_idx not in block_t_list[cand]:
                        all_covered = False
                        break
                if all_covered:
                    filtered.append(cand)
            
            if not filtered:
                continue
            
            if len(filtered) > 2000:
                rng.shuffle(filtered)
                filtered = filtered[:2000]
            
            best_swap = -1
            best_metric = float('-inf')
            
            # Precompute: t-subsets uniquely covered by old block
            old_unique = set()
            for t_idx in block_t_list[b_idx]:
                if covered_count[t_idx] == 1:
                    old_unique.add(t_idx)
            
            for cand in filtered:
                # Net new coverage
                new_cover = 0
                for t_idx in block_t_list[cand]:
                    if covered_count[t_idx] == 0:
                        new_cover += 1
                
                # t-subsets that become uncovered (uniquely covered by old, not by new)
                cand_t_set = set(block_t_list[cand])
                unique_lost = len(old_unique - cand_t_set)
                
                # Overlap penalty
                overlap = 0
                for t_idx in block_t_list[cand]:
                    if covered_count[t_idx] > 0:
                        overlap += 1
                
                metric = new_cover * 2.0 - unique_lost * 3.0 - overlap * 0.5
                
                if metric > best_metric:
                    best_metric = metric
                    best_swap = cand
            
            if best_swap != -1 and best_metric > 0:
                # Perform the swap
                for t_idx in block_t_list[b_idx]:
                    covered_count[t_idx] -= 1
                for t_idx in block_t_list[best_swap]:
                    covered_count[t_idx] += 1
                current[i] = best_swap
                current_set.discard(b_idx)
                current_set.add(best_swap)
                improved = True
                break
    
    return [list(all_blocks[i]) for i in current]


def _is_valid(v, k, t, blocks):
    """Check validity."""
    if not blocks:
        return v < t
    covered = set()
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False
        for x in b:
            if x < 0 or x >= v:
                return False
        covered.update(itertools.combinations(b, t))
    return covered == set(itertools.combinations(range(v), t))


def _dedupe(blocks):
    seen = set()
    result = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            result.append(list(key))
    return result


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    
    def time_limit_fn():
        return time.time() >= _DEADLINE
    
    if v < t:
        return []
    if k > v:
        k = v
    if t > k:
        return []
    
    # Precompute
    precomp = _precompute(v, k, t)
    
    # Initial greedy seed
    rng0 = random.Random(42)
    best_blocks = _greedy_cover(v, k, t, rng0, time_limit_fn, precomp)
    best_blocks = _dedupe(best_blocks)
    best_blocks = _remove_redundant(v, k, t, best_blocks, time_limit_fn, precomp)
    best_blocks = _one_exchange(v, k, t, best_blocks, rng0, time_limit_fn, precomp)
    
    if _is_valid(v, k, t, best_blocks):
        best_len = len(best_blocks)
    else:
        best_len = float('inf')
    
    # Portfolio of randomized trials
    trial = 0
    while not time_limit_fn() and trial < 300:
        trial += 1
        rng = random.Random(1000 + trial)
        
        candidate = _greedy_cover(v, k, t, rng, time_limit_fn, precomp)
        candidate = _dedupe(candidate)
        candidate = _remove_redundant(v, k, t, candidate, time_limit_fn, precomp)
        candidate = _one_exchange(v, k, t, candidate, rng, time_limit_fn, precomp)
        
        if _is_valid(v, k, t, candidate):
            if len(candidate) < best_len:
                best_len = len(candidate)
                best_blocks = candidate
    
    return best_blocks
