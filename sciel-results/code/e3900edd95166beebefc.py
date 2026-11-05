import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix, vstack
import itertools
import time
import random

def construct(v, k, t):
    # Parameters
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence matrix A (n_t_subsets x n_blocks)
    rows = []
    cols = []
    
    # Precompute combinations for each block to speed up?
    # itertools.combinations is C-speed, so it's fine.
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    A_T = A.T.tocsr()
    
    # Precompute block_covers for fast greedy updates
    # Using CSC format might be better for column access?
    # A is CSR. A.T is CSC (conceptually).
    # Let's just use the rows/cols lists to build a list of lists, or use A_T's structure.
    # A_T is n_blocks x n_t_subsets.
    # In CSR A_T, row j contains the non-zero indices (t_subset indices) for block j.
    # A_T.indptr[j] to A_T.indptr[j+1] gives the slice in A_T.indices.
    
    # Fast Greedy with Randomization
    best_solution = None
    best_size = float('inf')
    
    # Run multiple trials
    n_trials = 20
    start_time = time.time()
    time_limit_heuristic = 30.0 # Reserve time for ILP
    
    for _ in range(n_trials):
        if time.time() - start_time > time_limit_heuristic:
            break
            
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_blocks = []
        
        # To speed up, we can maintain a gain array
        # But recomputing gains via matvec is robust and fast enough for 20 trials
        while np.any(uncovered):
            gains = A_T @ uncovered.astype(np.float32)
            
            # Find indices with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            best_block = candidates[random.randint(0, len(candidates)-1)]
            
            selected_blocks.append(best_block)
            
            # Update uncovered
            # Get t-subsets covered by best_block
            # A_T is CSR. Row best_block.
            start_idx = A_T.indptr[best_block]
            end_idx = A_T.indptr[best_block + 1]
            covered_indices = A_T.indices[start_idx:end_idx]
            uncovered[covered_indices] = False
            
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks.copy()
            
    if best_solution is None:
        # Fallback: simple greedy without randomization if something went wrong
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_blocks = []
        while np.any(uncovered):
            gains = A_T @ uncovered.astype(np.float32)
            best_block = int(np.argmax(gains))
            if gains[best_block] == 0: break
            selected_blocks.append(best_block)
            start_idx = A_T.indptr[best_block]
            end_idx = A_T.indptr[best_block + 1]
            covered_indices = A_T.indices[start_idx:end_idx]
            uncovered[covered_indices] = False
        best_solution = selected_blocks
        best_size = len(selected_blocks)

    # Post-processing: Remove redundant blocks from best_solution
    # This is important to get a tight bound for ILP
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in best_solution:
        start_idx = A_T.indptr[b]
        end_idx = A_T.indptr[b + 1]
        coverage_count[A_T.indices[start_idx:end_idx]] += 1
        
    final_selected = []
    for b in best_solution:
        start_idx = A_T.indptr[b]
        end_idx = A_T.indptr[b + 1]
        indices = A_T.indices[start_idx:end_idx]
        
        # Temporarily remove
        coverage_count[indices] -= 1
        if not np.any(coverage_count < 1):
            # Keep removed
            pass
        else:
            # Restore
            coverage_count[indices] += 1
            final_selected.append(b)
            
    best_solution = final_selected
    best_size = len(best_solution)
    
    # ILP Phase: Iterative Bound Tightening
    # We try to find a solution with size < best_size
    # We iterate target = best_size - 1, best_size - 2, ...
    
    ilp_start_time = time.time()
    total_time_limit = 110.0 # Leave 10s for safety
    
    # Prepare ILP components
    c = np.ones(n_blocks)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_blocks)
    
    # A is n_t_subsets x n_blocks
    # We need to add a row for sum(x) <= target
    # sum_row = np.ones((1, n_blocks))
    
    # We will solve feasibility problems:
    # min 0 (or sum x, but constrained)
    # s.t. A x >= 1
    #      sum x <= target
    #      0 <= x <= 1, integer
    
    # Actually, minimizing sum x with an upper bound is equivalent to feasibility if we just want to know if a solution of size <= target exists.
    # But milp minimizes. If we set objective to 0, it finds any feasible solution.
    # If we set objective to sum x, it finds the minimum, which might be <= target.
    # Using objective sum x is better because if it finds a solution, it's likely the best possible for that subproblem, and we can use the actual objective value as the new best_size.
    
    current_best_size = best_size
    current_best_solution = best_solution
    
    # Try to improve
    # We don't know how many iterations we can afford.
    # Each ILP call might take 5-10s.
    # Let's try up to 5 improvements.
    
    for iteration in range(5):
        if time.time() - ilp_start_time > total_time_limit - 10:
            break
            
        target = current_best_size - 1
        if target < 1:
            break
            
        # Construct constraints
        # A constraints: A x >= 1
        # Sum constraint: 1^T x <= target
        
        # Stack A and ones row
        # A is sparse. ones row is dense? No, we can make it sparse.
        # sum_row = csr_matrix(np.ones((1, n_blocks)))
        
        # It's more efficient to just add the constraint separately if possible, 
        # but LinearConstraint expects a single matrix.
        # So we must stack.
        
        # Creating a new sparse matrix every iteration is expensive.
        # We can pre-create the stacked matrix? No, target changes.
        # But the structure (non-zeros) of the sum row is all ones.
        
        # Alternative: Use two LinearConstraints? 
        # scipy.optimize.milp takes a single LinearConstraint object.
        # We can combine them.
        
        # Let's create the stacked matrix once, but the RHS changes.
        # A_total = vstack([A, ones_row])
        # lb_total = [1, 0]
        # ub_total = [inf, target]
        
        # Precompute A_total structure?
        # A is 3876 x 77520.
        # ones_row is 1 x 77520.
        # vstack is fast enough?
        
        # Let's try to avoid vstack if possible.
        # Actually, we can just pass the original A and add the sum constraint in the objective?
        # No, objective is minimization.
        
        # Let's just do the vstack. It's 3877 rows.
        
        # Optimization: Only do ILP if we have time.
        
        try:
            # Create sum row
            # Using lil_matrix for easy creation? No, just np.ones.
            # scipy.sparse.csr_matrix(np.ones((1, n_blocks)))
            
            # To be efficient, we can pre-create the sum_row
            sum_row = csr_matrix(np.ones((1, n_blocks)))
            
            A_total = vstack([A, sum_row], format='csr')
            
            lb = np.concatenate([np.ones(n_t_subsets), [0]])
            ub = np.concatenate([np.full(n_t_subsets, np.inf), [target]])
            
            constraints = LinearConstraint(A_total, lb, ub)
            
            # Objective: minimize sum x
            # If a solution exists with sum <= target, the solver will find the min sum.
            # If min sum > target, it will be infeasible?
            # No, if min sum > target, the constraint sum <= target makes it infeasible.
            # So if status is infeasible, we know no solution of size <= target exists.
            # If status is optimal, we have a solution with size = res.fun.
            
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={"time_limit": 10, "mip_rel_gap": 0.0})
            
            if res.status == 0 or res.status == 1: # Optimal or Time Limit
                if res.x is not None:
                    x_int = np.round(res.x).astype(int)
                    # Verify solution
                    # Check coverage
                    if np.all(A @ x_int >= 1):
                        new_size = int(np.sum(x_int))
                        if new_size < current_best_size:
                            current_best_size = new_size
                            current_best_solution = [i for i in range(n_blocks) if x_int[i] == 1]
                            # Continue to try to improve further
                            continue
                        else:
                            # Found a solution of same size, or larger (shouldn't happen if min)
                            # If new_size == current_best_size, we didn't improve.
                            # If new_size > target, it's infeasible? No, if it found a solution, it's feasible.
                            # But we constrained sum <= target. So new_size <= target < current_best_size.
                            # So it must be an improvement.
                            pass
                    else:
                        # Invalid solution from solver? Fallback
                        pass
            elif res.status == 2: # Infeasible
                # No solution of size <= target.
                # So current_best_size is optimal (or close to it).
                break
            else:
                # Other errors
                break
                
        except Exception:
            break

    # Final verification and conversion
    if current_best_solution is None:
        return []
        
    blocks = [list(all_blocks[i]) for i in current_best_solution]
    
    return blocks
