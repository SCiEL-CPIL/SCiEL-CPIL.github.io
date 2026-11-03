import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Initialization: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Helper functions for cost and delta calculation
    # M[v, k] = b[v, p[k]]
    # Prod[i, j] = sum_k a[i, k] * b[j, p[k]]
    
    def update_M_Prod(p):
        M = b[:, p]
        Prod = a @ M.T
        return M, Prod

    M, Prod = update_M_Prod(p)
    
    # Initial Cost
    # Cost = sum_i,j a[i,j] b[p[i], p[j]]
    # = sum_i a[i, :] @ b[p[i], p]
    # = sum_i a[i, :] @ M[p[i], :]
    # = sum_i Prod[i, p[i]]? No.
    # Prod[i, j] = sum_k a[i, k] b[j, p[k]].
    # We want sum_i,k a[i, k] b[p[i], p[k]].
    # This is sum_i (a[i, :] @ M[p[i], :]).
    cost = np.sum([np.dot(a[i, :], M[p[i], :]) for i in range(n)])
    
    best_cost = cost
    best_p = p.copy()
    
    # Tabu parameters
    tabu_tenure = 50
    tabu_list = {}  # key: (i, j), value: iteration number
    iteration = 0
    
    # Precompute a[i,j] for quick access? a is numpy array, fast enough.
    
    while time.time() < deadline:
        iteration += 1
        
        # Find best move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # We iterate i, and for each i, compute deltas for all j
        # Delta_ij = (Base_i - Prod[i, j]) - a[i, j] * X_j - (Base_j - Self_j) + a[j, i] * X_i
        # X_j = b[u, v] - b[v, v] where u=p[i], v=p[j]
        # X_i = b[u, u] - b[v, u]
        
        # To speed up, we can vectorize the inner loop over j for a fixed i.
        
        for i in range(n):
            u = p[i]
            # Base_i = a[i, :] @ M[u, :]
            Base_i = np.dot(a[i, :], M[u, :])
            
            # We need Base_j for all j: a[j, :] @ M[u, :]
            # This is a @ M[u, :]
            Base_all = a @ M[u, :]
            
            # Self_j = Prod[j, j]
            Self = np.diag(Prod)
            
            # X_j vector: b[u, p[j]] - b[p[j], p[j]]
            # b[u, p[j]] is M[u, j]
            # b[p[j], p[j]] is M[p[j], j]
            # So X_j = M[u, :] - M[p, :]? No.
            # M[v, k] = b[v, p[k]].
            # M[u, j] = b[u, p[j]]. Correct.
            # M[p[j], j] = b[p[j], p[j]]. Correct.
            X_j_vec = M[u, :] - M[p, :] # M[p, :] is row p of M? No, M is indexed by location.
            # M is shape (n, n). M[row, col].
            # M[p[j], j] is element at row p[j], col j.
            # So X_j_vec[j] = M[u, j] - M[p[j], j].
            # We can compute this as M[u, :] - np.array([M[p[j], j] for j in range(n)])
            diag_M_p = np.array([M[p[j], j] for j in range(n)])
            X_j_vec = M[u, :] - diag_M_p
            
            # X_i vector: b[u, u] - b[p[j], u]
            # b[u, u] is scalar M[u, u]? No, M[u, u] = b[u, p[u]].
            # We need b[u, u]. Let's get it from b directly or M?
            # M[v, k] = b[v, p[k]].
            # If p[k] = u, then M[v, k] = b[v, u].
            # We need b[p[j], u].
            # Find k such that p[k] = u. Let k_u.
            # Then M[p[j], k_u] = b[p[j], u].
            # This is annoying.
            # Let's just use b directly for the X terms if needed, or precompute.
            # Actually, X_i and X_j are small terms.
            
            # Let's refine the Delta formula calculation.
            # Delta_ij = Base_i - Prod[i, j] - a[i, j] * X_j - (Base_j - Self_j) + a[j, i] * X_i
            
            # Prod[i, :] is available.
            # Base_all is available.
            # Self is available.
            
            # X_j_vec is available.
            # X_i_vec: b[u, u] - b[p[j], u]
            # b[u, u] is constant for this i.
            # b[p[j], u]: we need column of b corresponding to location u, indexed by p.
            # Let Col_u = b[:, u].
            # Then b[p[j], u] = Col_u[p[j]].
            # So X_i_vec = b[u, u] - Col_u[p].
            
            Col_u = b[:, u]
            X_i_vec = b[u, u] - Col_u[p]
            
            # Calculate Delta for all j
            # Term 1: Base_i - Prod[i, :]
            # Term 2: -a[i, :] * X_j_vec
            # Term 3: -(Base_all - Self)
            # Term 4: +a[:, i] * X_i_vec (Note: a[j, i])
            
            deltas = (Base_i - Prod[i, :]) - a[i, :] * X_j_vec - (Base_all - Self) + a[:, i] * X_i_vec
            
            # Exclude j == i
            deltas[i] = np.inf
            
            # Check tabu
            # We need to check if (i, j) is tabu.
            # This is hard to vectorize with a dictionary.
            # We can filter non-tabu moves.
            
            # For now, find min delta
            j_best = np.argmin(deltas)
            d_best = deltas[j_best]
            
            # Check tabu for this best candidate
            # If tabu, we might need to find next best.
            # To keep it simple and fast, we can just check the best.
            # If it's tabu, we skip this i? Or find next best?
            # Finding next best is O(n).
            
            # Let's check if (i, j_best) is tabu
            if tabu_list.get((i, j_best), 0) > iteration - tabu_tenure:
                 # It is tabu. Find next best?
                 # For simplicity, skip this i in this iteration?
                 # Or mark it and continue?
                 # Let's just skip if the best is tabu. This is a heuristic.
                 continue
                 
            if d_best < best_delta:
                best_delta = d_best
                best_i = i
                best_j = j_best
        
        if best_i == -1:
            # No improving move found? Or all tabu?
            # If no move found, we might need to restart or accept a worsening move.
            # For now, break or random restart.
            break
            
        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
        # Update M and Prod incrementally
        # Swap columns best_i and best_j in M
        M[:, best_i], M[:, best_j] = M[:, best_j].copy(), M[:, best_i].copy()
        
        # Update Prod rows best_i and best_j?
        # Prod[i, j] = a[i, :] @ M[j, :]
        # M changed in columns best_i, best_j.
        # So Prod[:, best_i] and Prod[:, best_j] change.
        # Prod[:, best_i] = a @ M[best_i, :]
        # Prod[:, best_j] = a @ M[best_j, :]
        Prod[:, best_i] = a @ M[best_i, :]
        Prod[:, best_j] = a @ M[best_j, :]
        
        # Update tabu list
        tabu_list[(best_i, best_j)] = iteration
        tabu_list[(best_j, best_i)] = iteration

    return best_p.tolist()
