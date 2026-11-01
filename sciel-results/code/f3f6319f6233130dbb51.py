"""Max-SAT-style local search for covering design C(24,7,3)."""
import itertools
import random
import time
import sys

import numpy as np

# Precompute all 3-subsets of {0..23}
ALL_TSETS = list(itertools.combinations(range(24), 3))
NUM_TSETS = len(ALL_TSETS)  # 2024

# Precompute for each 7-subset, which 3-subsets it covers (as indices into ALL_TSETS)
_TSET_INDEX = {ts: i for i, ts in enumerate(ALL_TSETS)}

def _generate_candidate_pool(v, k, t, num_candidates, rng):
    """Generate a diverse pool of candidate k-subsets."""
    pool = []
    pool_set = set()
    
    # Add some structured candidates
    # Random samples
    for _ in range(num_candidates):
        b = tuple(sorted(rng.sample(range(v), k)))
        if b not in pool_set:
            pool_set.add(b)
            pool.append(b)
    
    # Add some biased candidates: clusters around low indices
    for _ in range(num_candidates // 2):
        center = rng.randint(0, v - 1)
        # Sample k points with bias toward center
        weights = [1.0 / (1 + abs(i - center)) for i in range(v)]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        # Sample without replacement using weighted sampling
        chosen = set()
        while len(chosen) < k:
            r = rng.random()
            cum = 0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum and i not in chosen:
                    chosen.add(i)
                    break
            if len(chosen) < k:
                # Fallback: add random
                remaining = [i for i in range(v) if i not in chosen]
                if remaining:
                    chosen.add(rng.choice(remaining))
        b = tuple(sorted(chosen))
        if b not in pool_set:
            pool_set.add(b)
            pool.append(b)
    
    return pool


def _precompute_coverage_matrix(pool, v, t):
    """
    For each block in pool, precompute the set of t-set indices it covers.
    Returns a list of sets (or frozensets) of t-set indices.
    """
    coverage = []
    for b in pool:
        bset = set(b)
        covered = set()
        for ts in itertools.combinations(b, t):
            idx = _TSET_INDEX[ts]
            covered.add(idx)
        coverage.append(covered)
    return coverage


def _walksat_covering(v, k, t, pool, coverage, rng, time_limit, target_blocks=72):
    """
    WalkSAT-style local search.
    State: boolean vector of length len(pool), True = block selected.
    Objective: minimize number of selected blocks subject to all t-sets covered.
    Penalty: for each uncovered t-set, add penalty.
    """
    deadline = time.time() + time_limit
    n_pool = len(pool)
    
    # Initialize: select blocks greedily to get a valid starting point
    # Start with a random valid covering from greedy
    uncovered = set(range(NUM_TSETS))
    selected = [False] * n_pool
    selected_count = 0
    
    # Greedy initialization
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_gain = -1
        for i in range(n_pool):
            if selected[i]:
                continue
            gain = len(coverage[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0 or best_gain <= 0:
            break
        selected[best_idx] = True
        selected_count += 1
        uncovered -= coverage[best_idx]
    
    if uncovered:
        # Fallback: add random blocks until covered
        remaining_pool = [i for i in range(n_pool) if not selected[i]]
        rng.shuffle(remaining_pool)
        for i in remaining_pool:
            if not uncovered:
                break
            selected[i] = True
            selected_count += 1
            uncovered -= coverage[i]
    
    if uncovered:
        # Still not covered - this shouldn't happen with enough pool
        return None
    
    best_selected = list(selected)
    best_count = selected_count
    
    # WalkSAT-style search
    # Penalty function: count of uncovered t-sets
    # We want to minimize selected_count + lambda * uncovered_count
    
    # Precompute for each t-set, which pool blocks cover it
    tset_to_blocks = [[] for _ in range(NUM_TSETS)]
    for i in range(n_pool):
        for ts_idx in coverage[i]:
            tset_to_blocks[ts_idx].append(i)
    
    def get_uncovered(selected_vec):
        """Return set of uncovered t-set indices."""
        covered = set()
        for i in range(n_pool):
            if selected_vec[i]:
                covered |= coverage[i]
        return set(range(NUM_TSETS)) - covered
    
    def count_selected(selected_vec):
        return sum(1 for s in selected_vec if s)
    
    # Local search loop
    lam = 2.0  # Penalty weight for uncovered t-sets
    
    for iteration in range(200000):
        if time.time() >= deadline:
            break
        
        # Try to remove a block
        selected_indices = [i for i in range(n_pool) if selected[i]]
        if not selected_indices:
            break
        
        # Find removable blocks (those whose removal doesn't uncover any t-set)
        removable = []
        for i in selected_indices:
            # Check if any t-set is covered ONLY by block i
            can_remove = True
            for ts_idx in coverage[i]:
                # Check if any other selected block covers ts_idx
                other_cover = False
                for j in tset_to_blocks[ts_idx]:
                    if j != i and selected[j]:
                        other_cover = True
                        break
                if not other_cover:
                    can_remove = False
                    break
            if can_remove:
                removable.append(i)
        
        if removable:
            # Remove one at random (or the one that frees the most)
            i = rng.choice(removable)
            selected[i] = False
            selected_count -= 1
            
            # Verify still covered
            # Quick check: since we verified removable, it should be fine
            if selected_count < best_count:
                best_count = selected_count
                best_selected = list(selected)
            continue
        
        # No removable blocks - try a swap or perturbation
        # Perturbation: flip a random selected block off, then re-cover
        if rng.random() < 0.3:
            # Remove a random block and try to fix
            i = rng.choice(selected_indices)
            selected[i] = False
            selected_count -= 1
            
            # Now find uncovered t-sets
            covered = set()
            for j in range(n_pool):
                if selected[j]:
                    covered |= coverage[j]
            uncovered = set(range(NUM_TSETS)) - covered
            
            # Greedily add blocks to cover uncovered
            for _ in range(len(uncovered) + 5):
                if not uncovered:
                    break
                best_j = -1
                best_gain = -1
                for j in range(n_pool):
                    if selected[j]:
                        continue
                    gain = len(coverage[j] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j < 0 or best_gain <= 0:
                    break
                selected[best_j] = True
                selected_count += 1
                uncovered -= coverage[best_j]
            
            # If we ended up with fewer blocks, great
            if selected_count < best_count:
                best_count = selected_count
                best_selected = list(selected)
        
        # Try to swap: remove one, add another that covers more
        elif selected_indices:
            i = rng.choice(selected_indices)
            selected[i] = False
            selected_count -= 1
            
            covered = set()
            for j in range(n_pool):
                if selected[j]:
                    covered |= coverage[j]
            uncovered = set(range(NUM_TSETS)) - covered
            
            # Find best block to add
            best_j = -1
            best_gain = -1
            for j in range(n_pool):
                if selected[j]:
                    continue
                gain = len(coverage[j] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            
            if best_j >= 0 and best_gain > 0:
                selected[best_j] = True
                selected_count += 1
            
            if selected_count < best_count:
                best_count = selected_count
                best_selected = list(selected)
    
    # Convert back to blocks
    if best_selected is None:
        return None
    
    result = []
    for i in range(n_pool):
        if best_selected[i]:
            result.append(list(pool[i]))
    
    return result


def _greedy_covering_fallback(v, k, t, rng, time_limit):
    """Simple greedy covering as fallback."""
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x not in S:
                        tset = tuple(sorted(S + (x,)))
                        if tset in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        block.sort()
        blocks.append(tuple(block))
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]


def _verify_covering(blocks, v, t):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    all_tsets = set(itertools.combinations(range(v), t))
    return covered == all_tsets


def _local_search_refine(blocks, v, k, t, rng, time_limit):
    """Refine a covering by trying to remove or swap blocks."""
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    if not blocks:
        return blocks
    
    # Build coverage index
    def build_index(blks):
        tset_to_blocks = {}
        for i, b in enumerate(blks):
            for ts in itertools.combinations(b, t):
                if ts not in tset_to_blocks:
                    tset_to_blocks[ts] = []
                tset_to_blocks[ts].append(i)
        return tset_to_blocks
    
    tset_to_blocks = build_index(blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        
        # Try to remove redundant blocks
        for i in range(n):
            if time.time() >= deadline:
                break
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                covering = tset_to_blocks.get(ts, [])
                if covering == [i]:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(blocks[i], t):
                    if ts in tset_to_blocks and i in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(i)
                blocks.pop(i)
                tset_to_blocks = build_index(blocks)
                improved = True
                break
        
        # Try swaps
        if not improved:
            for idx in range(len(blocks)):
                if time.time() >= deadline:
                    break
                
                old_block = blocks[idx]
                critical = []
                for ts in itertools.combinations(old_block, t):
                    covering = tset_to_blocks.get(ts, [])
                    if covering == [idx]:
                        critical.append(ts)
                
                if not critical:
                    # Redundant
                    for ts in itertools.combinations(old_block, t):
                        if ts in tset_to_blocks and idx in tset_to_blocks[ts]:
                            tset_to_blocks[ts].remove(idx)
                    blocks.pop(idx)
                    tset_to_blocks = build_index(blocks)
                    improved = True
                    break
                
                # Try to find replacement
                best_cand = None
                best_extra = -1
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    cand = sorted(rng.sample(range(v), k))
                    cand_set = set(cand)
                    covers_all = all(set(ts).issubset(cand_set) for ts in critical)
                    if not covers_all:
                        continue
                    extra = 0
                    for ts in itertools.combinations(cand, t):
                        covering = tset_to_blocks.get(ts, [])
                        other = [c for c in covering if c != idx]
                        if not other:
                            extra += 1
                    if extra > best_extra:
                        best_extra = extra
                        best_cand = cand
                
                if best_cand is not None and best_extra > 0:
                    for ts in itertools.combinations(old_block, t):
                        if ts in tset_to_blocks and idx in tset_to_blocks[ts]:
                            tset_to_blocks[ts].remove(idx)
                    blocks[idx] = best_cand
                    for ts in itertools.combinations(best_cand, t):
                        if ts not in tset_to_blocks:
                            tset_to_blocks[ts] = []
                        if idx not in tset_to_blocks[ts]:
                            tset_to_blocks[ts].append(idx)
                    improved = True
                    break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Generate candidate pool
    pool_size = 800
    rng_pool = random.Random(42)
    pool = _generate_candidate_pool(v, k, t, pool_size, rng_pool)
    
    # Precompute coverage
    coverage = _precompute_coverage_matrix(pool, v, t)
    
    # Phase 2: WalkSAT-style search with multiple seeds
    walksat_budget = total_budget * 0.55
    per_walksat = walksat_budget / 4
    
    for seed in [1, 42, 137, 274]:
        elapsed = time.time() - start_time
        if elapsed > total_budget - 30.0:
            break
        
        rng = random.Random(seed)
        result = _walksat_covering(v, k, t, pool, coverage, rng, per_walksat, target_blocks=72)
        
        if result is not None and _verify_covering(result, v, t):
            if len(result) < best_len:
                best_len = len(result)
                best_blocks = result
    
    # Phase 3: Fallback greedy if WalkSAT didn't find anything good
    elapsed = time.time() - start_time
    if best_blocks is None or best_len > 75:
        remaining = total_budget - 30.0 - elapsed
        if remaining > 5.0:
            rng = random.Random(999)
            greedy_blocks = _greedy_covering_fallback(v, k, t, rng, remaining)
            if _verify_covering(greedy_blocks, v, t):
                if len(greedy_blocks) < best_len:
                    best_len = len(greedy_blocks)
                    best_blocks = greedy_blocks
    
    # Phase 4: Local search refinement
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 5.0 and best_blocks is not None:
        rng = random.Random(12345)
        refined = _local_search_refine(best_blocks, v, k, t, rng, remaining)
        if _verify_covering(refined, v, t):
            if len(refined) < best_len:
                best_blocks = refined
                best_len = len(refined)
    
    # Final safety
    if time.time() - start_time > total_budget - 3.0:
        if best_blocks is None:
            # Emergency fallback
            rng = random.Random(1)
            best_blocks = _greedy_covering_fallback(v, k, t, rng, 3.0)
        return best_blocks
    
    if best_blocks is None:
        rng = random.Random(1)
        best_blocks = _greedy_covering_fallback(v, k, t, rng, 5.0)
    
    return best_blocks
