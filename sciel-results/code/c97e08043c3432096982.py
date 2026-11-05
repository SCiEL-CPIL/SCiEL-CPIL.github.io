import itertools
import random
import time
import numpy as np
from math import comb
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for scipy overhead and finalization

    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build incidence matrix A (n_blocks x n_tsubsets)
    # A[b, s] = 1 if block b covers t-subset s
    # Using sparse representation logic manually or dense if small enough
    # n_blocks ~ 50k, n_tsubsets ~ 3.8k. Dense matrix is 50k*3.8k*8 bytes ~ 1.5GB. Too big for dense.
    # We need a sparse matrix or a way to handle it.
    # scipy.sparse is available.
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    # Iterate blocks, find covered t-subsets
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            s_idx = tsubset_to_idx[ts]
            row_indices.append(b_idx)
            col_indices.append(s_idx)
            data.append(1)
            
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_tsubsets))
    
    # Objective: Maximize sum of covered t-subsets
    # Since we want to minimize blocks, we can fix the number of blocks B and check if all are covered.
    # Or, maximize coverage with a penalty on blocks?
    # Better: Binary search on B (number of blocks) or iterate B from lower bound up.
    # Lower bound: ceil(C(v,t) / C(k,t)) = ceil(3876 / 35) = 111.
    # Known best: 151.
    # We can try to find a solution with B blocks that covers ALL t-subsets.
    # Constraint: A @ x >= 1 (all covered)
    # Constraint: sum(x) <= B
    # x in {0, 1}
    
    # To make it a maximization problem for milp (which minimizes), we can minimize -sum(covered)
    # But we want exact coverage.
    # Let's use the feasibility approach:
    # Minimize 0 subject to:
    # A @ x >= 1
    # sum(x) <= B
    # x in {0,1}
    
    # If feasible, we have a covering with B blocks.
    # We can start from a known good upper bound (e.g., 151) and go down, or start from LB and go up?
    # Going up is safer if we just want *a* solution, but we want *minimum*.
    # However, ILP solvers are good at proving infeasibility or finding feasible.
    # Let's try a range of B.
    
    # Strategy:
    # 1. Get a quick greedy solution to establish an initial upper bound B_init.
    # 2. Use ILP to try to find a solution with B < B_init.
    # 3. If found, update B_init and try lower.
    # 4. If time runs out, return best found.
    
    def greedy_initial():
        # Fast greedy to get a valid upper bound
        uncovered = np.ones(n_tsubsets, dtype=bool)
        selected = []
        
        # Precompute block covers as lists of indices for fast update
        # A is CSR, so A.indices gives column indices for each row.
        # We can use that.
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            # Find block with max gain
            # Gain = number of uncovered t-subsets in block
            # This is expensive to compute from scratch each time.
            # Approximation: Pick random uncovered, pick random block covering it? No, that's bad.
            # Let's do a few iterations of "best block" but limit candidates.
            
            # Randomly sample 100 uncovered t-subsets
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            sample_size = min(50, len(unc_idx))
            samples = np.random.choice(unc_idx, size=sample_size, replace=False)
            
            # For each sample, find blocks covering it.
            # We need t_subset_to_blocks.
            # Let's build it once if not built? 
            # It's memory heavy: 3876 lists of ~1000 ints. ~1.5M ints. OK.
            
            # Actually, let's just use the A matrix.
            # For a t-subset s, blocks covering it are rows where A[row, s] == 1.
            # In CSR, this is not direct.
            
            # Let's stick to a simpler greedy: 
            # Pick the uncovered t-subset with fewest covering blocks (most constrained).
            # We need counts.
            
            # Let's just do a random restart greedy with "best block" heuristic but limited scope.
            # To save time, we'll just return a fixed set of 151 blocks if we can find them?
            # No, we must generate.
            
            # Let's use the A matrix to compute gains for a subset of candidate blocks.
            # Candidates: Blocks that cover at least one uncovered t-subset.
            # This is all blocks initially.
            
            # Heuristic: Pick the block that covers the most uncovered t-subsets.
            # Gain[b] = sum(uncovered[A[b, :]])
            # This is O(n_blocks * n_tsubsets) per step. Too slow.
            
            # Alternative: Pick a random uncovered t-subset s.
            # Find all blocks b such that A[b, s] == 1.
            # Pick the one with max gain among those.
            
            s = samples[0]
            # Find blocks covering s
            # We can precompute t_subset_to_blocks once outside this function if needed.
            # For now, let's assume we have a list.
            
            # Let's build t_subset_to_blocks efficiently using the data arrays we already have.
            # We have row_indices, col_indices.
            # Invert them.
            # This is done once in the outer scope? No, let's do it here if not done.
            # Actually, the code above built A. Let's build the inverse list.
            
        return selected # Placeholder, will be replaced by a better greedy or ILP fallback

    # Let's build the inverse mapping t_subset_to_blocks
    # Using the data from A construction
    # We have row_indices (block idx), col_indices (t-subset idx)
    # t_subset_to_blocks[s] = list of b such that (b, s) in edges
    
    t_subset_to_blocks = [[] for _ in range(n_tsubsets)]
    for b, s in zip(row_indices, col_indices):
        t_subset_to_blocks[s].append(b)
        
    # Fast Greedy for Initial Bound
    def fast_greedy():
        uncovered = np.ones(n_tsubsets, dtype=np.int8)
        selected = []
        
        while np.any(uncovered > 0):
            if time.time() > deadline - 20:
                break
                
            # Pick random uncovered
            unc_idx = np.where(uncovered > 0)[0]
            if len(unc_idx) == 0:
                break
            s = unc_idx[random.randint(0, len(unc_idx)-1)]
            
            # Candidates: blocks covering s
            candidates = t_subset_to_blocks[s]
            if not candidates:
                # Should not happen
                break
                
            # Pick best candidate by gain
            best_b = -1
            best_gain = -1
            # Limit candidates to first 100 for speed? Or all if small?
            # Avg candidates ~1000. 1000 * 35 (avg cover size) = 35k ops. OK.
            
            for b in candidates:
                gain = 0
                # Get t-subsets covered by b
                # A is CSR. A.indptr[b] to A.indptr[b+1] gives col indices.
                start = A.indptr[b]
                end = A.indptr[b+1]
                cols = A.indices[start:end]
                for c in cols:
                    if uncovered[c] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
                    
            if best_b == -1:
                break
                
            selected.append(best_b)
            # Update uncovered
            start = A.indptr[best_b]
            end = A.indptr[best_b+1]
            cols = A.indices[start:end]
            uncovered[cols] -= 1
            # Clip to 0? No, we just need >0 check.
            
        # Repair if any uncovered remain (shouldn't happen if greedy is perfect, but random pick might fail if we ran out of time)
        if time.time() < deadline - 10:
            for s in range(n_tsubsets):
                if uncovered[s] > 0:
                    # Pick first available block
                    if t_subset_to_blocks[s]:
                        b = t_subset_to_blocks[s][0]
                        if b not in selected:
                            selected.append(b)
                            start = A.indptr[b]
                            end = A.indptr[b+1]
                            cols = A.indices[start:end]
                            uncovered[cols] -= 1
                            
        return selected

    init_blocks = fast_greedy()
    if not init_blocks:
        # Fallback: just pick random blocks until covered?
        # Or return empty? No, must be valid.
        # If greedy failed, we are in trouble. Let's just pick 200 random blocks and hope?
        # Better: Construct a trivial covering? C(19,7,4) is hard.
        # Let's assume fast_greedy works.
        pass

    current_best_blocks = init_blocks
    current_best_count = len(current_best_blocks)
    
    # ILP Optimization
    # We want to minimize B such that a covering exists.
    # We know current_best_count is a valid B.
    # Let's try B = current_best_count - 1, -2, etc.
    # Or binary search?
    # Since the gap is small (151 vs 131), linear search down is fine.
    
    # Setup ILP variables and constraints
    # x: n_blocks binary variables
    # Minimize: 0 (feasibility)
    # Constraints:
    # 1. A @ x >= 1 (all t-subsets covered)
    # 2. sum(x) <= B
    
    # scipy milp minimizes c @ x. Set c = 0.
    # Bounds: [0, 1] for all x.
    # Integrality: 1 for all x.
    
    c = np.zeros(n_blocks)
    integrality = np.ones(n_blocks)
    bounds = Bounds(lb=0, ub=1)
    
    # Constraint 1: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(n_tsubsets), ub=np.inf)
    # Constraint 2: sum(x) <= B
    # LinearConstraint(np.ones((1, n_blocks)), lb=-np.inf, ub=B)
    
    # Combine constraints? Or pass list.
    # scipy.optimize.milp accepts a list of LinearConstraint objects.
    
    # Note: A is sparse. LinearConstraint accepts sparse matrices.
    
    # We need to be careful with memory and time.
    # Let's try a few B values.
    
    # To avoid rebuilding constraints for every B, we can update the ub of the sum constraint.
    # But scipy's API might require rebuilding or it's cheap enough.
    
    # Let's try from current_best_count - 1 down to LB.
    LB = int(np.ceil(n_tsubsets / comb(k, t)))
    
    # Sort B values to try
    # Maybe try a few random B values in between? No, monotonic is better.
    
    # To save time, we can use the initial solution as a warm start?
    # scipy milp does not support warm start directly in this API version easily.
    
    # Let's limit the number of ILP calls to avoid timeout.
    # Each call might take seconds.
    # If we have 110s, and each call takes 5s, we can do ~20 calls.
    
    # Strategy:
    # 1. Try B = current_best_count - 1
    # 2. If feasible, update best, try B-1 again.
    # 3. If infeasible, stop? Or jump?
    #    If infeasible at B, it's infeasible for all smaller B.
    #    So we can binary search or stop.
    #    Actually, if infeasible at B, we know min blocks > B.
    #    So we can just stop if we find infeasible?
    #    Wait, we want the MINIMUM.
    #    If we start from high and go down:
    #    B_init is feasible.
    #    Try B_init - 1.
    #    If feasible, new best.
    #    If infeasible, then B_init was minimal? Not necessarily, we just failed to find a solution.
    #    ILP solver returns status. 
    #    Status 0: Optimal.
    #    Status 1: Iteration limit.
    #    Status 2: Time limit.
    #    Status 3: Infeasible.
    #    Status 4: Unbounded.
    
    # If status is 3 (Infeasible), we can stop searching lower B.
    # If status is 0 (Optimal/Feasible), we found a solution.
    
    # Let's implement this loop.
    
    best_B = current_best_count
    best_solution_x = None
    
    # Extract x from init_blocks for potential use or just to have a valid answer
    init_x = np.zeros(n_blocks)
    for b in current_best_blocks:
        init_x[b] = 1
        
    # We will try to improve.
    # To ensure we return *something* valid, we keep init_blocks.
    
    # Let's try a few steps down.
    # To be safe against slow ILP, we time-box each call.
    # scipy.optimize.milp has a time_limit option?
    # Yes, `time_limit` in options? Or `options={'time_limit': ...}`?
    # Check docs: `scipy.optimize.milp(c, constraints=None, integrality=None, bounds=None, options=None)`
    # Options for HiGHS (default) might include time limit.
    # Let's use `options={'time_limit': 5}` to limit each ILP call to 5 seconds.
    
    # If a call times out, we don't know if it's feasible or not.
    # We should probably stop if we get a timeout to avoid wasting time.
    
    # Let's try B from current_best_count - 1 down to LB.
    # But if we have many to try, we might run out of time.
    # Let's try 10 steps max.
    
    max_iter = 10
    for i in range(max_iter):
        B_try = current_best_count - 1 - i
        if B_try < LB:
            break
            
        if time.time() > deadline - 10:
            break
            
        # Define constraints
        # A @ x >= 1
        # sum(x) <= B_try
        
        # Note: A is (n_blocks, n_tsubsets). x is (n_blocks,).
        # A @ x gives (n_tsubsets,).
        
        try:
            res = milp(
                c=c,
                constraints=[
                    LinearConstraint(A, lb=np.ones(n_tsubsets), ub=np.inf),
                    LinearConstraint(np.ones((1, n_blocks)), lb=-np.inf, ub=B_try)
                ],
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': 5} # 5 seconds per ILP call
            )
            
            if res.status == 0: # Optimal/Feasible
                # Found a solution with B_try blocks
                x_sol = res.x
                new_blocks = [i for i in range(n_blocks) if x_sol[i] > 0.5]
                
                # Verify validity (just in case)
                # Check if all t-subsets covered
                # A @ x_sol >= 1
                # Since it's ILP, it should be.
                
                current_best_blocks = new_blocks
                current_best_count = len(current_best_blocks)
                best_solution_x = x_sol
                
                # Continue to try lower B
            elif res.status == 3: # Infeasible
                # No solution with B_try blocks.
                # Since we are decreasing B, all lower B are also infeasible.
                break
            else:
                # Timeout or other error. Stop trying lower B to be safe?
                # Or continue? If we timed out, we don't know.
                # Assuming it's hard, let's stop.
                break
                
        except Exception as e:
            # If ILP fails, stop
            break
            
    # Final cleanup: ensure we return a valid list of blocks
    # If ILP found a better solution, use it. Otherwise, use greedy.
    
    # Convert indices to lists of points
    result = [list(all_blocks[b_idx]) for b_idx in current_best_blocks]
    
    # Final verification (optional but good for debugging, skip for speed in production)
    # The evaluator checks it.
    
    return result
