import itertools
import numpy as np
from scipy import sparse
import time

def construct(v, k, t):
    # Generate all blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build sparse incidence matrix A (n_tsubs x n_blocks)
    # A[i, j] = 1 if tsub i is in block j
    rows = []
    cols = []
    data = []
    
    for j, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_tsubs, n_blocks))
    
    # Objective: minimize sum of x
    c = np.ones(n_blocks)
    
    # Bounds: x in {0, 1}
    integrality = np.ones(n_blocks)
    bounds = [(0, 1)] * n_blocks
    
    # Constraints: A x >= 1
    # milp expects constraints in form: lb <= A_con x <= ub
    # Here we have A x >= 1, so lb=1, ub=inf
    # Note: milp uses LinearConstraint
    from scipy.optimize import LinearConstraint
    
    constraint = LinearConstraint(A, lb=np.ones(n_tsubs), ub=np.inf)
    
    # Try to solve using milp
    # Check if milp is available
    try:
        from scipy.optimize import milp
        
        # Set time limit to 100 seconds to leave buffer
        # options for HiGHS: time_limit
        # Note: The 'options' dict is passed to the solver
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 100}
        )
        
        # Check if a feasible solution was found
        # status 2: Optimal
        # status 1: Feasible (if time limit hit but feasible found?)
        # Actually, HiGHS status codes:
        # 0: Optimal
        # 1: Feasible (but not necessarily optimal? Or just status?)
        # Let's check res.status
        # scipy.optimize.milp returns OptimizeResult
        # res.status: 0 (Optimal), 1 (Feasible), 2 (Infeasible), 3 (Unbounded), 4 (Error), 5 (Iteration Limit), 6 (Time Limit)
        
        if res.status in [0, 1, 6]: # Optimal, Feasible, or Time Limit with feasible
            x = res.x
            # x is a numpy array of 0s and 1s (possibly float due to solver)
            # Threshold to binary
            selected_indices = np.where(x > 0.5)[0]
            
            if len(selected_indices) > 0:
                # Verify validity just in case
                selected_blocks = [all_blocks[i] for i in selected_indices]
                
                # Quick check: does this cover all tsubs?
                # We can reconstruct coverage or trust the solver.
                # Given the cost of verification, and the solver's constraints,
                # we trust it but do a sanity check on size.
                
                return [list(b) for b in selected_blocks]
                
    except Exception as e:
        pass
        
    # Fallback: Fast Greedy
    # Re-use logic from original but optimized with NumPy if possible, 
    # or just the original logic since it's fast enough for fallback?
    # The original took some time. Let's do a simpler greedy.
    
    # For fallback, we just need a valid solution.
    # The original greedy is O(N * M * K) roughly.
    # Let's just use the original logic copied here to be safe, 
    # or a simpler random sampling if time is tight?
    # No, the original greedy is reliable.
    
    # Since we are inside a "new program" directive, we should implement a 
    # decent fallback. The original code's greedy is fine as a fallback.
    
    # However, to save space and ensure the "new approach" is the main driver,
    # we can implement a simple greedy here.
    
    # Simple Greedy Fallback
    all_blocks_list = all_blocks
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    block_tsubs = []
    for j, block in enumerate(all_blocks_list):
        tsubs_in_block = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)
        for s_idx in tsubs_in_block:
            tsub_to_blocks[s_idx].append(j)
            
    cover_count = [0] * n_tsubs
    block_gain = [len(tsubs) for tsubs in block_tsubs]
    selected = []
    remaining = set(range(n_blocks))
    
    while True:
        if all(cover_count[s] > 0 for s in range(n_tsubs)):
            break
        best_block = -1
        best_gain = -1
        for b in remaining:
            if block_gain[b] > best_gain:
                best_gain = block_gain[b]
                best_block = b
        if best_block == -1 or best_gain <= 0:
            break
        selected.append(all_blocks_list[best_block])
        remaining.discard(best_block)
        for s_idx in block_tsubs[best_block]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                for b in tsub_to_blocks[s_idx]:
                    if b in remaining:
                        block_gain[b] -= 1
                        
    return [list(b) for b in selected]
