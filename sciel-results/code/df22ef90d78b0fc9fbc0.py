import numpy as np
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # New Strategy: Greedy Set Cover with Backtracking
    # We view the problem as: Find a set of rank-1 tensors T_c such that sum T_c = T_mm.
    # Since T_mm has entries 1, we need the sum of selected tensors to be 1 at each (i,j,k).
    # We restrict entries of T_c to {-1, 0, 1}.
    # This is equivalent to finding a sparse integer solution to A x = b where A is built from candidates.
    # But here we generate candidates on the fly.
    
    # A rank-1 tensor T is defined by vectors u (n*m), v (m*p), w (p*n).
    # T[i*m+j, j*p+k, k*n+i] = u[i*m+j] * v[j*p+k] * w[k*n+i].
    
    # We will try to construct a solution of rank R by picking R such tensors.
    # We start with a "deficit" vector d of size n*m*p, initialized to 1 (since we need sum=1).
    # We pick a tensor T that "helps" the most (maximizes reduction in L1 norm of deficit).
    # We subtract T from d. If d becomes 0, we have a solution.
    # If we get stuck (no tensor can reduce deficit further or deficit becomes negative in a way that can't be fixed), we backtrack.
    # To make this tractable, we limit the search depth and use heuristics.
    
    # Pre-compute indices for faster access
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_indices = len(indices)
    
    # Map (i,j,k) to index in flattened array
    idx_map = {}
    for idx, (i, j, k) in enumerate(indices):
        idx_map[(i, j, k)] = idx
        
    # We will search for ranks from 60 down to 40.
    # We use a randomized greedy approach with limited backtracking.
    
    random.seed(42)
    
    # Heuristic: A candidate tensor is "good" if it has many non-zeros that align with the current deficit.
    # Deficit d[idx] indicates how much more we need at that position.
    # Initially d[idx] = 1.
    # If we pick T, new_d = d - T.
    # We want new_d to be "closer" to 0.
    # Score(T) = sum(|d[idx]| - |d[idx] - T[idx]|)
    
    # To speed up, we pre-generate a pool of candidate tensors.
    # But the number of possible tensors with entries in {-1,0,1} is huge.
    # We generate random candidates and keep the best ones.
    
    def generate_candidate():
        u = np.array([random.choice([-1, 0, 1]) for _ in range(n * m)])
        v = np.array([random.choice([-1, 0, 1]) for _ in range(m * p)])
        w = np.array([random.choice([-1, 0, 1]) for _ in range(p * n)])
        
        # Compute tensor values
        t_vals = np.zeros(num_indices)
        for idx, (i, j, k) in enumerate(indices):
            t_vals[idx] = u[i * m + j] * v[j * p + k] * w[k * n + i]
        
        return (u, v, w, t_vals)
    
    # We will try to find a solution for a given rank R.
    def solve_for_rank(R, time_limit=10.0):
        start_time = time.time()
        
        # We use a beam search / greedy with restarts.
        # State: (current_deficit, list_of_selected_tensors)
        
        best_solution = None
        best_len = R + 1
        
        num_restarts = 50
        beam_width = 5
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initial state
            deficit = np.ones(num_indices)
            selected = []
            
            # Beam of states
            beam = [(deficit.copy(), selected)]
            
            for step in range(R):
                if time.time() - start_time > time_limit:
                    break
                
                new_beam = []
                # For each state in beam, generate candidates and pick best
                for state_deficit, state_selected in beam:
                    # Generate a few random candidates
                    # To be efficient, we generate candidates and score them
                    candidates = []
                    for _ in range(20): # 20 candidates per state
                        u, v, w, t_vals = generate_candidate()
                        # Score: how much does this reduce the L1 norm of deficit?
                        # new_deficit = state_deficit - t_vals
                        # reduction = sum(|state_deficit|) - sum(|state_deficit - t_vals|)
                        # We only care about positions where t_vals is non-zero
                        mask = t_vals != 0
                        if not np.any(mask):
                            continue
                        new_deficit_candidate = state_deficit - t_vals
                        # Check if any entry becomes too large? 
                        # We want to end up at 0. Intermediate values can be negative.
                        # But if |deficit| grows too large, it's unlikely to recover.
                        # Heuristic: penalty if max |new_deficit| > 2
                        if np.max(np.abs(new_deficit_candidate)) > 2:
                            continue
                            
                        reduction = np.sum(np.abs(state_deficit)) - np.sum(np.abs(new_deficit_candidate))
                        candidates.append((reduction, u, v, w, t_vals, new_deficit_candidate))
                    
                    if not candidates:
                        continue
                        
                    # Sort by reduction
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    # Keep top beam_width candidates
                    for cand in candidates[:beam_width]:
                        reduction, u, v, w, t_vals, new_deficit = cand
                        new_selected = state_selected + [(u, v, w)]
                        new_beam.append((new_deficit, new_selected))
                        
                        # Check if solved
                        if np.allclose(new_deficit, 0):
                            if len(new_selected) < best_len:
                                best_len = len(new_selected)
                                best_solution = new_selected[:]
                    
                if not new_beam:
                    break
                    
                # Prune beam to keep only unique best states? 
                # Or just keep top beam_width overall?
                # Let's keep top beam_width based on current "closeness" to 0
                # Closeness metric: L1 norm of deficit
                new_beam.sort(key=lambda x: np.sum(np.abs(x[0])))
                beam = new_beam[:beam_width]
                
                # If we found a solution, we can stop expanding this branch?
                # Actually, we want the shortest solution.
                # If we found a solution of length L, we only need to explore steps < L.
                if best_solution is not None:
                    # If we already have a solution of length <= R, we might be able to prune
                    # But we are targeting R, so if we find one shorter, great.
                    pass
                    
            # If we reached step R and didn't solve, this restart failed.
            # But maybe a shorter solution was found inside?
            # The loop above checks for solution at each step.
            
            if best_solution is not None:
                # We found a solution. Can we do better?
                # Since we are in a restart loop, we can try to find a shorter one.
                # But for now, if we found one, we return it.
                # The outer loop will try to find even shorter ones if we lower R.
                return best_solution
        
        return None
    
    # Try ranks from 60 down to 40
    # We start with higher ranks because they are easier to find.
    # Once we find a solution for R, we try R-1.
    
    for target_rank in range(60, 39, -1):
        if time.time() > deadline:
            break
            
        # If we already have a solution better than target_rank, skip
        if best_rank <= target_rank:
            continue
            
        # Try to find a solution of rank target_rank
        # We allow the solver to find a solution of rank <= target_rank
        sol = solve_for_rank(target_rank, time_limit=10.0)
        
        if sol is not None:
            # Construct U, V, W
            rank = len(sol)
            U_sol = [[0] * rank for _ in range(n * m)]
            V_sol = [[0] * rank for _ in range(m * p)]
            W_sol = [[0] * rank for _ in range(p * n)]
            
            for c, (u, v, w) in enumerate(sol):
                for i in range(n * m):
                    U_sol[i][c] = u[i]
                for j in range(m * p):
                    V_sol[j][c] = v[j]
                for k in range(p * n):
                    W_sol[k][c] = w[k]
            
            # Verify
            if verify(U_sol, V_sol, W_sol, n, m, p):
                if rank < best_rank:
                    best = {"U": U_sol, "V": V_sol, "W": W_sol}
                    best_rank = rank
                # If we found a solution, we can break and try lower ranks in next iteration
                # But the loop structure handles this by decrementing target_rank
                break # Break inner loop, outer loop continues to lower rank
                
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # New Strategy: Greedy Set Cover with Backtracking
    # We view the problem as: Find a set of rank-1 tensors T_c such that sum T_c = T_mm.
    # Since T_mm has entries 1, we need the sum of selected tensors to be 1 at each (i,j,k).
    # We restrict entries of T_c to {-1, 0, 1}.
    
    # Pre-compute indices for faster access
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_indices = len(indices)
    
    # Map (i,j,k) to index in flattened array
    idx_map = {}
    for idx, (i, j, k) in enumerate(indices):
        idx_map[(i, j, k)] = idx
        
    random.seed(42)
    
    def generate_candidate():
        u = np.array([random.choice([-1, 0, 1]) for _ in range(n * m)])
        v = np.array([random.choice([-1, 0, 1]) for _ in range(m * p)])
        w = np.array([random.choice([-1, 0, 1]) for _ in range(p * n)])
        
        # Compute tensor values
        t_vals = np.zeros(num_indices)
        for idx, (i, j, k) in enumerate(indices):
            t_vals[idx] = u[i * m + j] * v[j * p + k] * w[k * n + i]
        
        return (u, v, w, t_vals)
    
    # We will try to find a solution for a given rank R.
    def solve_for_rank(R, time_limit=10.0):
        start_time = time.time()
        
        # We use a beam search / greedy with restarts.
        # State: (current_deficit, list_of_selected_tensors)
        
        best_solution = None
        best_len = R + 1
        
        num_restarts = 50
        beam_width = 5
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initial state
            deficit = np.ones(num_indices)
            selected = []
            
            # Beam of states
            beam = [(deficit.copy(), selected)]
            
            for step in range(R):
                if time.time() - start_time > time_limit:
                    break
                
                new_beam = []
                # For each state in beam, generate candidates and pick best
                for state_deficit, state_selected in beam:
                    # Generate a few random candidates
                    # To be efficient, we generate candidates and score them
                    candidates = []
                    for _ in range(20): # 20 candidates per state
                        u, v, w, t_vals = generate_candidate()
                        # Score: how much does this reduce the L1 norm of deficit?
                        # new_deficit = state_deficit - t_vals
                        # reduction = sum(|state_deficit|) - sum(|state_deficit - t_vals|)
                        # We only care about positions where t_vals is non-zero
                        mask = t_vals != 0
                        if not np.any(mask):
                            continue
                        new_deficit_candidate = state_deficit - t_vals
                        # Check if any entry becomes too large? 
                        # We want to end up at 0. Intermediate values can be negative.
                        # But if |deficit| grows too large, it's unlikely to recover.
                        # Heuristic: penalty if max |new_deficit| > 2
                        if np.max(np.abs(new_deficit_candidate)) > 2:
                            continue
                            
                        reduction = np.sum(np.abs(state_deficit)) - np.sum(np.abs(new_deficit_candidate))
                        candidates.append((reduction, u, v, w, t_vals, new_deficit_candidate))
                    
                    if not candidates:
                        continue
                        
                    # Sort by reduction
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    # Keep top beam_width candidates
                    for cand in candidates[:beam_width]:
                        reduction, u, v, w, t_vals, new_deficit = cand
                        new_selected = state_selected + [(u, v, w)]
                        new_beam.append((new_deficit, new_selected))
                        
                        # Check if solved
                        if np.allclose(new_deficit, 0):
                            if len(new_selected) < best_len:
                                best_len = len(new_selected)
                                best_solution = new_selected[:]
                    
                if not new_beam:
                    break
                    
                # Prune beam to keep only unique best states? 
                # Or just keep top beam_width overall?
                # Let's keep top beam_width based on current "closeness" to 0
                # Closeness metric: L1 norm of deficit
                new_beam.sort(key=lambda x: np.sum(np.abs(x[0])))
                beam = new_beam[:beam_width]
                
                # If we found a solution, we can stop expanding this branch?
                # Actually, we want the shortest solution.
                # If we found a solution of length L, we only need to explore steps < L.
                if best_solution is not None:
                    # If we already have a solution of length <= R, we might be able to prune
                    # But we are targeting R, so if we find one shorter, great.
                    pass
                    
            # If we reached step R and didn't solve, this restart failed.
            # But maybe a shorter solution was found inside?
            # The loop above checks for solution at each step.
            
            if best_solution is not None:
                # We found a solution. Can we do better?
                # Since we are in a restart loop, we can try to find a shorter one.
                # But for now, if we found one, we return it.
                # The outer loop will try to find even shorter ones if we lower R.
                return best_solution
        
        return None
    
    # Try ranks from 60 down to 40
    # We start with higher ranks because they are easier to find.
    # Once we find a solution for R, we try R-1.
    
    for target_rank in range(60, 39, -1):
        if time.time() > deadline:
            break
            
        # If we already have a solution better than target_rank, skip
        if best_rank <= target_rank:
            continue
            
        # Try to find a solution of rank target_rank
        # We allow the solver to find a solution of rank <= target_rank
        sol = solve_for_rank(target_rank, time_limit=10.0)
        
        if sol is not None:
            # Construct U, V, W
            rank = len(sol)
            U_sol = [[0] * rank for _ in range(n * m)]
            V_sol = [[0] * rank for _ in range(m * p)]
            W_sol = [[0] * rank for _ in range(p * n)]
            
            for c, (u, v, w) in enumerate(sol):
                for i in range(n * m):
                    U_sol[i][c] = u[i]
                for j in range(m * p):
                    V_sol[j][c] = v[j]
                for k in range(p * n):
                    W_sol[k][c] = w[k]
            
            # Verify
            if verify(U_sol, V_sol, W_sol, n, m, p):
                if rank < best_rank:
                    best = {"U": U_sol, "V": V_sol, "W": W_sol}
                    best_rank = rank
                # If we found a solution, we can break and try lower ranks in next iteration
                # But the loop structure handles this by decrementing target_rank
                break # Break inner loop, outer loop continues to lower rank
                
    return best
# EVOLVE-BLOCK-END
