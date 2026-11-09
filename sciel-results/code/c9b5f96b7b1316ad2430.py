import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Integer Linear Programming (ILP).
    
    Strategy:
    1. Formulate the covering problem as an ILP where variables x_b indicate if block b is used.
    2. Minimize sum(x_b) subject to covering all t-subsets.
    3. Use scipy.optimize.milp with a time limit.
    4. If ILP fails or times out without an integer solution, fall back to a fast greedy heuristic.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate all candidate blocks (k-subsets of v)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets (requirements)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # 3. Build the constraint matrix A
    # A[i, j] = 1 if t-subset i is contained in block j, else 0
    # For v=19, k=8, t=5:
    # num_blocks = C(19,8) = 75582
    # num_t_subsets = C(19,5) = 11628
    # Matrix size ~ 878 million entries. This is too large for dense storage in standard memory limits if not sparse.
    # However, scipy.optimize.milp accepts sparse matrices. We should build a sparse matrix.
    
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    # Efficiently populate the sparse matrix
    # For each block j, find all t-subsets it covers
    for j, block in enumerate(all_blocks):
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            # We need the index of this t-subset in all_t_subsets
            # Using a dictionary for O(1) lookup is better than searching the list
            # But building the dict for 11628 items is fast.
            pass 
            
    # Let's precompute the mapping from t-subset to index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    for j, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            i = t_subset_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_t_subsets, num_blocks))
    
    # 4. Define ILP problem
    # Minimize c^T x
    c = np.ones(num_blocks)
    
    # Constraints: A x >= 1
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
    # We need A x >= 1, so lb = 1, ub = inf
    constraints = LinearConstraint(A, 1, np.inf)
    
    # Bounds: 0 <= x <= 1
    integrality = np.ones(num_blocks)  # All variables are integer (binary)
    bounds = Bounds(0, 1)
    
    # 5. Solve
    res = None
    try:
        # time_limit is in seconds
        # options can include 'max_nodes', 'presolve', etc.
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": deadline - time.time() - 5.0, 
                            "presolve": True,
                            "max_nodes": 100000})
    except Exception as e:
        print(f"ILP Solver Error: {e}")
        res = None

    # Check if we have a valid integer solution
    if res is not None and res.success and res.x is not None:
        # res.x is the solution vector
        # Filter out blocks with x > 0.5
        selected_blocks = [all_blocks[i] for i in range(num_blocks) if res.x[i] > 0.5]
        
        # Verify validity (just in case)
        if is_valid_covering(selected_blocks, v, k, t, t_subset_to_idx, all_t_subsets):
            return [list(b) for b in selected_blocks]

    # Fallback: Greedy Heuristic
    # If ILP fails, times out, or returns invalid result, use a fast greedy method.
    return greedy_construct_fallback(v, k, t, deadline)

def is_valid_covering(blocks, v, k, t, t_subset_to_idx, all_t_subsets):
    """Check if blocks cover all t-subsets."""
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx.get(ts)
            if idx is not None:
                covered.add(idx)
    return len(covered) == len(all_t_subsets)

def greedy_construct_fallback(v, k, t, deadline):
    """
    Fast greedy fallback if ILP fails.
    """
    start_time = time.time()
    remaining_time = deadline - start_time
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    cover_count = [0] * num_t_subsets
    blocks = []
    
    # Uncovered sets
    uncovered = set(range(num_t_subsets))
    
    # Precompute blocks that cover each t-subset? No, that's the ILP matrix.
    # Just greedy pick.
    
    while uncovered and time.time() < deadline - 2.0:
        # Pick a random uncovered t-subset to prioritize? 
        # Or pick the one with fewest covering blocks? (Too slow to compute dynamically)
        # Simple greedy: Pick a random uncovered t-subset, find a block covering it with max new coverage.
        
        target_idx = list(uncovered)[0] # Or random
        target_subset = all_t_subsets[target_idx]
        
        best_block = None
        best_gain = -1
        
        # Iterate through all possible blocks? C(19,8) is 75k. 
        # For each block, check if it covers target_subset.
        # This is expensive if done naively for every step.
        
        # Optimization: Precompute for each t-subset, the list of blocks that cover it.
        # This is the transpose of the ILP matrix.
        # Let's build it once.
        
        # Actually, building the full incidence matrix in Python lists is memory heavy but doable for 11k x 75k (sparse).
        # But we already built the ILP matrix. We can reuse the logic.
        
        # For simplicity in fallback, let's just use a random restart greedy with limited time.
        
        # Random block selection heuristic
        # Pick random block that covers target_subset
        # We can sample random blocks until we find one covering target_subset?
        # Probability a random k-subset covers a specific t-subset:
        # C(v-t, k-t) / C(v, k)
        # For v=19, k=8, t=5: C(14,3)/C(19,8) = 364 / 75582 ~ 0.0048.
        # So we need ~200 samples on average. That's fine.
        
        found_block = False
        for _ in range(500): # Max 500 random tries
            b = tuple(sorted(random.sample(range(v), k)))
            if set(target_subset).issubset(set(b)):
                # Calculate gain
                gain = 0
                for ts in itertools.combinations(b, t):
                    idx = t_subset_to_idx[ts]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                if gain == len(list(uncovered)): # Perfect block? Unlikely
                    break
        if best_block:
            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                idx = t_subset_to_idx[ts]
                if idx in uncovered:
                    uncovered.remove(idx)
        
        if time.time() > deadline - 5.0:
            break
            
    # Final check and patch
    if uncovered:
        # Emergency patch: add blocks for remaining uncovered
        for idx in list(uncovered):
            ts = all_t_subsets[idx]
            # Add a block containing ts
            # Randomly fill the rest
            rest = random.sample([x for x in range(v) if x not in ts], k - t)
            b = tuple(sorted(list(ts) + rest))
            blocks.append(b)
            
    # Remove duplicates
    unique_blocks = list(set(blocks))
    
    # Reduce redundancy
    cover_count = [0] * num_t_subsets
    for b in unique_blocks:
        for ts in itertools.combinations(b, t):
            cover_count[t_subset_to_idx[ts]] += 1
            
    reduced_blocks = []
    for b in unique_blocks:
        removable = True
        for ts in itertools.combinations(b, t):
            if cover_count[t_subset_to_idx[ts]] == 1:
                removable = False
                break
        if removable:
            for ts in itertools.combinations(b, t):
                cover_count[t_subset_to_idx[ts]] -= 1
        else:
            reduced_blocks.append(b)
            
    return [list(b) for b in reduced_blocks]

# Helper for random import in fallback
import random

if __name__ == "__main__":
    # Test
    blocks = construct(19, 8, 5)
    print(f"Number of blocks: {len(blocks)}")
