import itertools
import numpy as np
import time
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a dual ascent approach 
    followed by weighted greedy selection.
    """
    # 1. Precompute all t-subsets and k-subsets
    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    k_subsets = list(itertools.combinations(points, k))
    
    n_t = len(t_subsets)
    n_k = len(k_subsets)
    
    # Map t-subset to index
    t_idx_map = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute for each k-subset, which t-subsets it contains
    # blocks_to_subs[j] = list of indices of t-subsets contained in k_subsets[j]
    blocks_to_subs = []
    for B in k_subsets:
        subs = []
        for S in itertools.combinations(B, t):
            subs.append(t_idx_map[S])
        blocks_to_subs.append(subs)
        
    # Precompute for each t-subset, which k-subsets contain it
    # subs_to_blocks[i] = list of indices of k-subsets containing t_subsets[i]
    subs_to_blocks = [[] for _ in range(n_t)]
    for j, subs in enumerate(blocks_to_subs):
        for i in subs:
            subs_to_blocks[i].append(j)
            
    # 2. Dual Ascent to find good dual weights
    # We want to find dual variables mu[i] >= 0 such that for all j: sum_{i in blocks_to_subs[j]} mu[i] <= 1
    # The objective is to maximize sum(mu[i]) / n_t (normalized)
    # But for greedy, we just want mu to be large for "hard" sets.
    # We use a simple subgradient update.
    
    mu = np.zeros(n_t)
    max_iter = 2000
    step_size = 0.05
    min_step = 0.001
    
    # To speed up "find best block", we maintain the current "score" of each block
    # score[j] = sum(mu[i] for i in blocks_to_subs[j])
    scores = np.zeros(n_k)
    
    # Initialize scores (all 0)
    
    best_score_val = -1
    best_score_idx = -1
    
    for it in range(max_iter):
        # Find block with max score
        # Instead of scanning all n_k every time, we can just scan. 
        # n_k ~ 50k, max_iter 2000 -> 100M ops, might be slow in pure Python.
        # Let's use numpy for the update and argmax.
        
        # Current best block
        max_idx = np.argmax(scores)
        max_val = scores[max_idx]
        
        if max_val <= 1.0 + 1e-9:
            # No violated constraint, dual is feasible. We can stop or continue with smaller steps to refine?
            # For greedy, we just need decent duals. Feasible duals are good.
            # Actually, if max_val <= 1, the dual is feasible.
            break
            
        # Update mu for subsets in the best block
        subs_in_best = blocks_to_subs[max_idx]
        if not subs_in_best:
            break
            
        # Update mu
        mu[subs_in_best] += step_size
        
        # Update scores for all blocks that share these subsets
        # This is the expensive part.
        # We can batch this.
        # For each subset i in subs_in_best:
        #   For each block j in subs_to_blocks[i]:
        #     scores[j] += step_size
        
        # To optimize, we can just use a sparse matrix multiplication or similar,
        # but with n_t=3876 and n_k=50388, dense operations might be okay if we vectorize?
        # No, 3876 * 50388 is ~ 190M floats, too big for dense.
        
        # Let's do it with loops but efficiently.
        # Since step_size is constant, we just increment scores.
        
        # Optimization: If step_size is fixed, we can just add step_size to scores[j]
        # for all j connected to any i in subs_in_best.
        
        # To avoid O(n_t * avg_degree) overhead, we can just recalculate scores?
        # No, that's O(n_k * t).
        
        # Let's stick to the incremental update.
        # The average degree of a t-subset in k-subsets is C(v-t, k-t) = C(15, 3) = 455.
        # So for each block update, we touch ~ C(7,4)*455 ~ 35 * 455 ~ 15k blocks.
        # 2000 iterations * 15k updates = 30M updates. This is feasible in Python if optimized.
        
        for i in subs_in_best:
            for j in subs_to_blocks[i]:
                scores[j] += step_size
                
        # Decay step size
        step_size *= 0.99
        
        if step_size < min_step:
            break
            
    # 3. Weighted Greedy Cover
    # Select blocks greedily based on the sum of mu of *uncovered* subsets they contain.
    
    covered = np.zeros(n_t, dtype=bool)
    blocks_selected = []
    selected_block_indices = set()
    
    # We need to efficiently find the best block at each step.
    # Current weight of block j = sum(mu[i] for i in blocks_to_subs[j] if not covered[i])
    
    # We can maintain an array `current_weights` for blocks.
    current_weights = scores.copy() # Initial weight is sum of all mu
    
    # However, as we cover subsets, weights decrease.
    # We use a heap for lazy deletion.
    import heapq
    
    # Heap elements: (-weight, block_index, version)
    # We need a version counter or just re-check validity.
    # Simpler: Just re-evaluate the top few candidates or use a priority queue with stale checks.
    
    heap = [(-current_weights[j], j) for j in range(n_k)]
    heapq.heapify(heap)
    
    while len(blocks_selected) < 500: # Safety cap, usually < 200
        if not heap:
            break
            
        neg_w, j = heapq.heappop(heap)
        w = -neg_w
        
        # Check if this entry is stale
        # Calculate actual current weight
        actual_w = 0.0
        subs = blocks_to_subs[j]
        for i in subs:
            if not covered[i]:
                actual_w += mu[i]
                
        if abs(actual_w - w) > 1e-9:
            # Stale entry, push updated value
            heapq.heappush(heap, (-actual_w, j))
            continue
            
        if actual_w <= 1e-9:
            # No more uncovered subsets with positive weight? 
            # Or all covered?
            # If all covered, we are done.
            if np.all(covered):
                break
            # If not all covered but weight is 0, it means mu is 0 for remaining uncovered.
            # This can happen if dual ascent didn't assign weight to some sets.
            # In that case, we should fall back to a simple greedy for the rest.
            # But let's assume mu is good enough. If w is 0, this block adds nothing "valuable".
            # We might need to handle "unweighted" greedy for leftovers.
            # For now, if best weight is 0, break and do fallback.
            break
            
        # Select block j
        blocks_selected.append(list(k_subsets[j]))
        selected_block_indices.add(j)
        
        # Update coverage and weights
        for i in subs:
            if not covered[i]:
                covered[i] = True
                # Subtract mu[i] from all blocks containing i
                for j_prime in subs_to_blocks[i]:
                    if j_prime not in selected_block_indices:
                        # We can't easily update current_weights array for heap without pushing
                        # We rely on the lazy deletion in the heap loop above.
                        # We just need to ensure the heap eventually gets the right value.
                        # The loop above recalculates actual_w on pop.
                        # So we don't need to push updates here?
                        # Actually, if we don't push, the old high weight remains in the heap.
                        # When popped, we recalculate actual_w, see it's lower, and push the new one.
                        # This works but might pop many stale entries.
                        # To mitigate, we could push (-actual_w, j_prime) here.
                        # Let's do that to speed up convergence.
                        new_w = 0.0
                        for i2 in blocks_to_subs[j_prime]:
                            if not covered[i2]:
                                new_w += mu[i2]
                        heapq.heappush(heap, (-new_w, j_prime))
                        
    # Check if all covered
    if not np.all(covered):
        # Fallback: Simple greedy for remaining uncovered sets
        # This handles cases where duals were zero for some sets
        remaining_uncovered_indices = [i for i in range(n_t) if not covered[i]]
        if remaining_uncovered_indices:
            rem_blocks = _fallback_greedy_specific(v, k, t, remaining_uncovered_indices, t_idx_map, k_subsets, blocks_to_subs, subs_to_blocks)
            blocks_selected.extend(rem_blocks)
            # Update covered (not strictly necessary for return, but for consistency)
            for B in rem_blocks:
                for S in itertools.combinations(B, t):
                    covered[t_idx_map[S]] = True
                    
    return blocks_selected

def _fallback_greedy_specific(v, k, t, remaining_indices, t_idx_map, k_subsets, blocks_to_subs, subs_to_blocks):
    """
    Greedy cover for specific remaining t-subsets.
    """
    if not remaining_indices:
        return []
        
    # Create a set of remaining t-subset indices for fast lookup
    rem_set = set(remaining_indices)
    
    blocks = []
    # We need to pick blocks that cover elements in rem_set
    # Weight is just count of covered remaining elements
    
    # Precompute count for each block?
    # Counts change as we pick blocks.
    
    # To be fast, we can just iterate.
    # n_k ~ 50k, iterations ~ 20. 1M ops. Fine.
    
    while rem_set:
        best_block = None
        best_count = 0
        
        for j, B in enumerate(k_subsets):
            count = 0
            for i in blocks_to_subs[j]:
                if i in rem_set:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
                
        if best_block is None or best_count == 0:
            break
            
        blocks.append(list(best_block))
        
        # Remove covered
        for i in blocks_to_subs[k_subsets.index(best_block)]:
            rem_set.discard(i)
            
    return blocks

# Dummy fallback in case the main logic fails completely
def _fallback_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    k_subsets = list(itertools.combinations(range(v), k))
    
    while uncovered:
        best_block = None
        best_count = -1
        for B in k_subsets:
            count = sum(1 for S in itertools.combinations(B, t) if S in uncovered)
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        for S in itertools.combinations(best_block, t):
            uncovered.discard(S)
    return blocks
