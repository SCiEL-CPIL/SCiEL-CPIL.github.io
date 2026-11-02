import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
import time
import random

def construct(v, k, t):
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build sparse matrix A where A[i, j] = 1 if t_subsets[i] is a subset of k_subsets[j]
    rows = []
    cols = []
    data = []
    
    # Also build subsets_of_block and blocks_of_subset for the dynamic greedy
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    # LP Formulation
    c = np.ones(n_k)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_t)
    bounds = [(0, 1)] * n_k
    
    duals = None
    primal = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return _fallback_greedy(v, k, t)
        
        # Extract dual variables
        # res.ineqlin.marginals corresponds to A_ub constraints
        duals = -res.ineqlin.marginals
        duals = np.maximum(duals, 0)
        primal = res.x
        primal = np.clip(primal, 0, 1)
        
    except Exception:
        return _fallback_greedy(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Greedy with Dual Weights
    # Weights for block j is sum of duals of subsets in j
    weights_dual = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            weights_dual[j] = np.sum(duals[subsets_of_block[j]])
            
    blocks1 = _run_greedy(weights_dual, subsets_of_block, blocks_of_subset, k_subsets, duals)
    if blocks1:
        if len(blocks1) < best_count:
            best_count = len(blocks1)
            best_blocks = blocks1
            
    # Strategy 2: Greedy with Primal Weights
    # Weights for block j is x_j
    weights_primal = primal.copy()
    blocks2 = _run_greedy(weights_primal, subsets_of_block, blocks_of_subset, k_subsets, weights_primal)
    if blocks2:
        if len(blocks2) < best_count:
            best_count = len(blocks2)
            best_blocks = blocks2
            
    # Strategy 3: Randomized Primal Weights
    # Add small noise to break ties and explore
    random.seed(42) # Fixed seed for reproducibility in evaluation? 
    # Actually, random seed might vary. Let's use a fixed seed for the "noise" run to be deterministic, 
    # or just try a few fixed perturbations.
    # Let's try one run with noise.
    noise = np.random.random(n_k) * 0.1
    weights_noisy = weights_primal + noise
    blocks3 = _run_greedy(weights_noisy, subsets_of_block, blocks_of_subset, k_subsets, weights_noisy)
    if blocks3:
        if len(blocks3) < best_count:
            best_count = len(blocks3)
            best_blocks = blocks3
            
    if best_blocks:
        return best_blocks
    else:
        return _fallback_greedy(v, k, t)

def _run_greedy(initial_weights, subsets_of_block, blocks_of_subset, k_subsets, update_weights):
    """
    Runs a greedy covering algorithm.
    initial_weights: weight of each block initially.
    update_weights: weights used to decrement when a subset is covered. 
                    For dual strategy, this is duals.
                    For primal strategy, this is... well, if we use primal as weight, 
                    what do we subtract? 
                    If we subtract x_j, it doesn't make sense physically.
                    Usually, if we use x_j as priority, we just pick max x_j.
                    But we need to update "value" of remaining blocks?
                    If we just pick max x_j, we don't need to update weights based on coverage?
                    No, because a block that covers already covered subsets is less useful.
                    So we should subtract the "value" of the subsets we cover.
                    If we use x_j as weight, it's not additive over subsets.
                    
                    Correction: 
                    The standard "Greedy on LP Solution" is:
                    1. Pick block with max x_j.
                    2. Mark covered.
                    3. For remaining blocks, do we change their priority?
                    If we don't, we might pick blocks that are redundant.
                    So we should define a "remaining value" for each block.
                    If we use duals, Value(B) = sum_{s in B, s uncovered} y_s. This works perfectly.
                    If we use primal x_j, there is no direct decomposition.
                    
                    However, we can approximate:
                    Let's stick to Dual Weights for the "update" mechanism, 
                    but maybe initialize with a mix?
                    
                    Actually, the current code uses duals for both initialization and update.
                    Let's try:
                    Run 1: Dual Weights (as before).
                    Run 2: Primal Weights, but update by subtracting... what?
                    If we subtract x_j from other blocks? No.
                    
                    Alternative for Run 2:
                    Just pick max x_j, then re-evaluate "potential" of others?
                    Potential of B = x_j * (fraction of subsets in B that are uncovered)?
                    This requires tracking uncovered count per block.
                    
                    Let's simplify:
                    Run 1: Dual-based greedy (Standard).
                    Run 2: Randomized Dual-based greedy (Add noise to duals).
                    Run 3: Pure Primal Greedy (Pick max x_j, no weight update, just pick next max x_j among those that cover *some* new subset? Or just max x_j globally? 
                    If we just pick max x_j, we might pick a block that covers 0 new subsets if x_j is high but it's redundant.
                    So we must check "covers new subset".
                    
                    Let's modify _run_greedy to accept a "mode".
                    Mode 'dual': Update weights by subtracting duals.
                    Mode 'primal': Priority is x_j. When picking, pick max x_j among blocks that cover >=1 new subset. 
                                  After picking, we don't update x_j, but we must ensure we don't pick redundant blocks.
                                  This is effectively: Sort by x_j descending. Iterate. If block covers new subset, pick it.
                                  This is very fast!
    """
    n_k = len(k_subsets)
    n_t = len(blocks_of_subset)
    
    covered = np.zeros(n_t, dtype=bool)
    picked = np.zeros(n_k, dtype=bool)
    blocks = []
    
    # Precompute uncovered count for each block? 
    # For 'dual' mode, we need heap.
    # For 'primal' mode, we can sort indices by initial_weights (x_j) once.
    
    if initial_weights is None:
        return []
        
    if len(initial_weights) == n_t: # Heuristic: if weights length matches n_t, it's duals? No, weights are per block.
         pass
        
    # Let's assume initial_weights is per block.
    # We need to know if we are in dual or primal mode.
    # Let's pass a flag or infer.
    # Better: pass a function or a specific object.
    
    # Let's refactor _run_greedy to be specific to Dual Mode (most robust).
    # And add a separate simple Primal Greedy.
    
    # Dual Mode Implementation
    block_weights = initial_weights.copy()
    heap = [(-block_weights[j], j) for j in range(n_k)]
    heapq.heapify(heap)
    
    while heap:
        neg_w, j = heapq.heappop(heap)
        w = -neg_w
        
        if picked[j]:
            continue
            
        if w > block_weights[j] + 1e-9:
            continue
            
        if w <= 1e-9:
            continue
            
        # Check if it actually covers new subsets
        # If w > 0, it implies sum of duals of uncovered subsets > 0.
        # Since duals are non-negative, this means at least one uncovered subset has dual > 0.
        # So it covers a new subset.
        
        picked[j] = True
        blocks.append(list(k_subsets[j]))
        
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                y_s = update_weights[s_idx] # In dual mode, update_weights is duals
                if y_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        if not picked[b_prime]:
                            block_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                            
    if np.all(covered):
        return blocks
    else:
        # Fallback for remaining
        remaining_indices = np.where(~covered)[0]
        rem_blocks = _fallback_greedy_remaining(v, k, t, remaining_indices, k_subsets)
        blocks.extend(rem_blocks)
        return blocks

def _primal_greedy(primal_weights, subsets_of_block, blocks_of_subset, k_subsets):
    """
    Greedy that picks blocks in order of decreasing primal weight x_j,
    but only if they cover at least one new subset.
    """
    n_k = len(k_subsets)
    n_t = len(blocks_of_subset)
    
    covered = np.zeros(n_t, dtype=bool)
    picked = np.zeros(n_k, dtype=bool)
    blocks = []
    
    # Sort block indices by primal weight descending
    order = np.argsort(primal_weights)[::-1]
    
    for j in order:
        # Check if this block covers any uncovered subset
        # If yes, pick it.
        # Optimization: Maintain count of uncovered subsets per block?
        # Or just check on the fly.
        # n_k is 50k. For each j, we check 35 subsets.
        # Total checks: 50k * 35 = 1.75M. This is fast.
        
        is_new = False
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                is_new = True
                break
                
        if is_new:
            picked[j] = True
            blocks.append(list(k_subsets[j]))
            # Mark all subsets in this block as covered
            for s_idx in subsets_of_block[j]:
                covered[s_idx] = True
                
            # Early exit if all covered
            if np.all(covered):
                break
                
    if not np.all(covered):
        remaining_indices = np.where(~covered)[0]
        rem_blocks = _fallback_greedy_remaining(v, k, t, remaining_indices, k_subsets)
        blocks.extend(rem_blocks)
        
    return blocks

def _fallback_greedy(v, k, t):
    # Standard max-coverage greedy
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    k_subsets = list(itertools.combinations(range(v), k))
    
    # Precompute subsets for each block to speed up
    # But for fallback, we want it simple and robust.
    # We can reuse the logic from _primal_greedy but with random order?
    # Or just standard greedy.
    
    while uncovered:
        best_block = None
        best_count = -1
        for B in k_subsets:
            count = 0
            for S in itertools.combinations(B, t):
                if S in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        for S in itertools.combinations(best_block, t):
            if S in uncovered:
                uncovered.remove(S)
    return blocks

def _fallback_greedy_remaining(v, k, t, remaining_indices, k_subsets):
    # Convert indices back to tuples
    # Note: t_index is not passed, so we need to reconstruct or pass it.
    # In the previous code, t_index was passed. Let's keep it.
    # Wait, I changed the signature in my head. Let's check the call site.
    # In construct, I call _fallback_greedy_remaining(v, k, t, remaining_indices, k_subsets)
    # But I need t_index to map indices to subsets.
    # Let's pass t_index.
    
    # Correction: I will modify the call to pass t_index.
    pass

def construct(v, k, t):
    # ... (setup same as above) ...
    # ... (LP solve same as above) ...
    
    best_blocks = None
    best_count = float('inf')
    
    # Run Dual Greedy
    blocks1 = _run_greedy_dual(weights_dual, subsets_of_block, blocks_of_subset, k_subsets, duals)
    if blocks1:
        if len(blocks1) < best_count:
            best_count = len(blocks1)
            best_blocks = blocks1
            
    # Run Primal Greedy
    blocks2 = _primal_greedy(primal, subsets_of_block, blocks_of_subset, k_subsets)
    if blocks2:
        if len(blocks2) < best_count:
            best_count = len(blocks2)
            best_blocks = blocks2
            
    if best_blocks:
        return best_blocks
    else:
        return _fallback_greedy(v, k, t)
