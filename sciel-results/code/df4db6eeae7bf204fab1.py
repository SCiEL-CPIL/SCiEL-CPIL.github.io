import time
import numpy as np
import itertools
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*(3-1) = 15
    
    t_start = time.time()
    # Safety margin: The limit is 150s, but we should return well before.
    # scipy's milp might hang or take long, so we set a strict internal deadline.
    deadline = 140.0 
    
    # Precompute all words as indices 0..N-1
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx

    # Precompute the incidence matrix A (sparse)
    # A[i, j] = 1 if word j is in the ball of radius 'radius' around word i
    # We need: sum_j A[i, j] * x_j >= 1 for all i
    # Minimize: sum_j x_j
    
    # For q=3, n=7, radius=1:
    # Each ball has size 1 + 7*2 = 15.
    # Total non-zeros = 2187 * 15 = 32805.
    # This is a very small sparse matrix for an ILP solver.
    
    rows = []
    cols = []
    vals = []
    
    for i, w in enumerate(all_words):
        # Center
        rows.append(i)
        cols.append(i)
        vals.append(1)
        
        if radius >= 1:
            w_list = list(w)
            for pos in range(n):
                old_val = w_list[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        w_list[pos] = new_val
                        j = tuple_to_idx(tuple(w_list))
                        rows.append(i)
                        cols.append(j)
                        vals.append(1)
                w_list[pos] = old_val
                
    A = lil_matrix((N, N), dtype=np.int8)
    for r, c, v in zip(rows, cols, vals):
        A[r, c] = v
        
    A_csr = A.tocsr()
    
    # Objective: minimize sum x_j
    c = np.ones(N, dtype=np.float64)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A_csr, lb=np.ones(N), ub=np.inf)
    constraints = LinearConstraint(A_csr, lb=np.ones(N, dtype=np.float64), ub=np.inf)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    
    # Integrality: x_j must be integer (0 or 1)
    integrality = np.ones(N, dtype=np.int8)
    
    # Initial bounds for the solver to help pruning
    # Lower bound for K_3(7,1) is 156.
    # Upper bound (best known) is 186.
    # We can set the objective bound to help the solver.
    # However, scipy's milp doesn't directly take an objective bound in the same way as Gurobi/CPLEX.
    # But it does support `options`.
    
    # We will run the solver with a time limit.
    # Note: scipy.optimize.milp uses HiGHS under the hood.
    # HiGHS has a time limit option.
    
    options = {
        'time_limit': 135.0,  # Stop after 135s
        'mip_rel_gap': 0.0    # We want the optimal or best feasible
    }
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
        
        # Check if a feasible solution was found
        if res.status == 0 or res.status == 2: # 0: Optimal, 2: Time limit reached with incumbent
            if res.x is not None:
                # Extract codewords
                # res.x might have small floating point errors
                x = np.round(res.x).astype(np.int8)
                # Ensure binary
                x = (x > 0.5).astype(np.int8)
                
                selected_indices = np.where(x == 1)[0]
                
                # Convert to list of lists
                result = []
                for idx in selected_indices:
                    w = []
                    temp_idx = idx
                    for _ in range(n):
                        w.append(temp_idx % q)
                        temp_idx //= q
                    w.reverse()
                    result.append(w)
                    
                # Sanity check: are they distinct? Yes, indices are unique.
                # Sanity check: do they cover? 
                # The solver ensures constraints are met.
                # But due to numerical issues or early termination, let's verify quickly if time permits.
                # For now, trust the solver's feasibility check.
                
                if len(result) > 0:
                    return result
                else:
                    # Fallback if empty
                    return [[0]*n]
            else:
                # No solution found, fallback to greedy
                pass
        else:
            # Solver failed, fallback to greedy
            pass
            
    except Exception as e:
        # If scipy fails, fallback to a simple greedy to ensure a valid answer
        pass

    # Fallback: Simple Greedy
    # This should rarely be reached if scipy works, but better safe than sorry.
    balls = np.empty((N, 1 + n*(q-1)), dtype=np.int32)
    for i, w in enumerate(all_words):
        ball = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball

    covered = np.zeros(N, dtype=bool)
    code = []
    while not covered.all():
        gains = np.sum(~covered[balls], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0: break
        candidates = np.where(gains == max_gain)[0]
        idx = candidates[0]
        code.append(idx)
        covered[balls[idx]] = True
        
    result = []
    for idx in code:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
