import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # 4. Build reverse mapping: for each t-subset, which blocks cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            sub_to_blocks[block_t_indices[i, j]].append(i)

    # 5. Calculate Weights for Weighted Greedy
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)

    # 6. Helper functions
    def remove_redundant(remaining, block_t_indices, n_t, deadline):
        if not remaining:
            return remaining
        
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in remaining:
                coverage_count[block_t_indices[bi]] += 1
            
            to_remove = []
            for bi in remaining:
                ts = block_t_indices[bi]
                if np.all(coverage_count[ts] > 1):
                    to_remove.append(bi)
            
            if to_remove:
                remove_set = set(to_remove)
                remaining = [bi for bi in remaining if bi not in remove_set]
                changed = True
                
        return remaining

    def try_swap_improve(remaining, block_t_indices, n_t, sub_to_blocks, deadline, rng):
        """Try to swap blocks to reduce total count."""
        if not remaining or time.time() > deadline - 5.0:
            return remaining, False
        
        remaining = list(remaining)
        improved = False
        
        # Compute coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in remaining:
            coverage_count[block_t_indices[bi]] += 1
        
        # Try removing each block and finding a replacement
        for idx in range(len(remaining)):
            if time.time() > deadline - 5.0:
                break
            
            bi = remaining[idx]
            ts = block_t_indices[bi]
            
            # Find critical subsets (covered only by this block)
            critical = ts[coverage_count[ts] == 1]
            
            if len(critical) == 0:
                # Block is redundant, just remove it
                remaining.pop(idx)
                coverage_count[ts] -= 1
                improved = True
                idx -= 1  # Re-check same position
                continue
            
            if len(critical) > 3:
                # Too many critical subsets, hard to find replacement
                continue
            
            # Find candidate blocks that cover all critical subsets
            # Start with candidates covering the first critical subset
            candidate_set = set(sub_to_blocks[critical[0]])
            for c in critical[1:]:
                candidate_set &= set(sub_to_blocks[c])
                if not candidate_set:
                    break
            
            if not candidate_set:
                continue
            
            # Remove current block from consideration
            candidate_set.discard(bi)
            
            if not candidate_set:
                continue
            
            # Score candidates: prefer those that cover many other subsets
            # (making other blocks redundant)
            best_cand = None
            best_score = -1
            
            for cand in list(candidate_set)[:20]:  # Limit candidates checked
                if time.time() > deadline - 5.0:
                    break
                
                # Simulate: remove bi, add cand
                # New coverage = coverage_count - ts + cand_ts
                cand_ts = block_t_indices[cand]
                
                # Check if adding cand makes any other block redundant
                # A block is redundant if all its subsets have coverage >= 2
                # after the swap
                new_coverage = coverage_count.copy()
                new_coverage[ts] -= 1
                new_coverage[cand_ts] += 1
                
                redundant_count = 0
                for other_idx in range(len(remaining)):
                    if other_idx == idx:
                        continue
                    other_bi = remaining[other_idx]
                    other_ts = block_t_indices[other_bi]
                    if np.all(new_coverage[other_ts] > 1):
                        redundant_count += 1
                
                score = redundant_count
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand is not None and best_score >= 1:
                # Perform the swap
                remaining[idx] = best_cand
                coverage_count[ts] -= 1
                coverage_count[block_t_indices[best_cand]] += 1
                improved = True
        
        return remaining, improved

    def greedy_fill(uncovered, block_t_indices, n_t, t_weights, deadline, rng):
        """Greedy fill to cover all uncovered subsets."""
        selected = []
        while True:
            if time.time() > deadline - 3.0:
                break
            if not uncovered.any():
                break
            
            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False
        
        return selected

    # 7. Main search loop
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)

    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1

        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []

        # Greedy loop
        while True:
            if time.time() > deadline - 10.0:
                break

            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False

        if time.time() > deadline - 10.0:
            break

        # Post-optimization: Remove redundant blocks
        if selected_blocks:
            remaining = remove_redundant(list(selected_blocks), block_t_indices, n_t, deadline)

            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]

            # Local Search: Swap-based improvement
            if time.time() < deadline - 15.0 and len(remaining) > 0:
                # Multiple rounds of swap improvement
                for round_num in range(3):
                    if time.time() > deadline - 10.0:
                        break
                    remaining, improved = try_swap_improve(
                        remaining, block_t_indices, n_t, sub_to_blocks, deadline, rng
                    )
                    if not improved:
                        break
                    
                    # Remove any new redundancies
                    remaining = remove_redundant(remaining, block_t_indices, n_t, deadline)
                    
                    if len(remaining) < best_size:
                        best_size = len(remaining)
                        best_solution = [k_subsets[i] for i in remaining]

                # Shake and Repair
                if time.time() < deadline - 10.0 and len(remaining) > 0:
                    shake_size = max(1, len(remaining) // 8)
                    shake_indices = rng.sample(range(len(remaining)), min(shake_size, len(remaining)))
                    to_remove = set(remaining[i] for i in shake_indices)
                    new_remaining = [b for b in remaining if b not in to_remove]

                    # Re-calculate uncovered based on new_remaining
                    new_uncovered = np.ones(n_t, dtype=bool)
                    for bi in new_remaining:
                        new_uncovered[block_t_indices[bi]] = False

                    # Re-greedy to fill gaps
                    fill_blocks = greedy_fill(new_uncovered, block_t_indices, n_t, t_weights, deadline, rng)
                    new_remaining.extend(fill_blocks)

                    # Re-remove redundants and try swaps
                    new_remaining = remove_redundant(new_remaining, block_t_indices, n_t, deadline)
                    
                    # One more round of swap improvement
                    new_remaining, _ = try_swap_improve(
                        new_remaining, block_t_indices, n_t, sub_to_blocks, deadline, rng
                    )
                    new_remaining = remove_redundant(new_remaining, block_t_indices, n_t, deadline)

                    if len(new_remaining) < best_size:
                        best_size = len(new_remaining)
                        best_solution = [k_subsets[i] for i in new_remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]
