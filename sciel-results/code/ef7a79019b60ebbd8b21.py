"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    # Work with float64 for continuous relaxation
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    # Precompute row sums for initial heuristic
    a_sums = a_f.sum(axis=1)
    b_sums = b_f.sum(axis=1)

    def compute_cost(p):
        """Compute exact integer cost for permutation p."""
        return int(np.sum(a_int * b_int[p]))

    def steepest_descent_2opt(p, cost, time_limit, start_time):
        """Steepest descent 2-opt local search. Returns (best_p, best_cost)."""
        p = p.copy()
        best_cost = cost
        best_p = p.copy()
        
        while True:
            if time.time() - start_time > time_limit:
                break
            
            # Find the best improving move (steepest descent)
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    diff_a = a_f[i] - a_f[j]
                    diff_b = b_f[v, p] - b_f[u, p]
                    delta = np.dot(diff_a, diff_b) + 2 * a_f[i, j] * b_f[u, v]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta >= 0:
                break  # Local optimum
            
            # Apply the best move
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += int(best_delta)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        return best_p, best_cost

    def admm_relaxation(a, b, max_iter=20, tol=1e-4):
        """
        ADMM for the continuous QAP relaxation:
        min sum_{i,j} a[i,j] * X[i,:] @ b @ X[j,:]^T
        s.t. X is a doubly stochastic matrix
        
        Uses the formulation: min c^T x + 0.5 * x^T Q x
        where x = vec(X), but we use a simpler ADMM with row/column constraints.
        """
        # Initialize X as doubly stochastic
        X = np.ones((n, n)) / n
        
        # ADMM parameters
        rho = 1.0
        U = np.zeros((n, n))  # Dual variable
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # X-update: minimize f(X) + rho/2 * ||X - X_prev + U||^2
            # This is a QP: min trace(X^T * (a^T X b + rho*I)) + rho*trace(X^T*(X_prev - U))
            # Simplified: use projected gradient or just row/column normalization
            
            # For simplicity, use a gradient step + projection
            # Gradient of the objective: 2 * (a^T X b + X a b^T) but simplified
            # Actually, let's use a simpler approach: 
            # The objective is sum_{i,j} a[i,j] * (X[i,:] @ b @ X[j,:])
            # = sum_{i,j,k,l} a[i,j] X[i,k] b[k,l] X[j,l]
            # = trace(X^T A X B) where A=a, B=b (need to be careful)
            
            # Let's use a simpler heuristic: iteratively adjust X to be doubly stochastic
            # while minimizing a surrogate
            X_prev = X.copy()
            
            # Gradient step (approximate)
            # The gradient w.r.t. X[i,k] is sum_j a[i,j] * sum_l b[k,l] X[j,l]
            # = sum_j a[i,j] * (X @ b^T)[j,k] = (a @ X @ b^T)[i,k]
            grad = a_f @ X @ b_f.T
            X = X - 0.1 * grad
            
            # Project to doubly stochastic (Sinkhorn-Knopp)
            for _ in range(20):
                X = X / X.sum(axis=1, keepdims=True)
                X = X / X.sum(axis=0, keepdims=True)
                X = np.maximum(X, 1e-10)
            
            # Check convergence
            if np.abs(X - X_prev).max() < tol:
                break
        
        # Recover permutation via Hungarian algorithm
        # Minimize sum_{i,j} X[i,j] * cost[i,j] where cost[i,j] is the "relaxed cost"
        # Actually, we want to assign facility i to location j to minimize the overall cost
        # Use the final X as weights: minimize sum_{i,j} w[i,j] where w is derived from X
        # Better: use the Hungarian algorithm on a cost matrix derived from the relaxation
        
        # The cost of assigning facility i to location j in the relaxed problem
        # is related to the gradient or the interaction terms
        # Simple approach: use -X as the cost (we want high X values to be selected)
        cost_matrix = -X
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        p = np.empty(n, dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            p[i] = j
        
        return p

    # Generate multiple initial solutions
    solutions = []
    
    # Solution 1: Rank matching
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy_idx] = central_idx
    solutions.append(p1)
    
    # Solution 2: Random
    p2 = np.random.permutation(n)
    solutions.append(p2)
    
    # Solution 3: ADMM relaxation
    try:
        p3 = admm_relaxation(a_f, b_f, max_iter=15)
        solutions.append(p3)
    except:
        pass
    
    # Solution 4: Another random
    p4 = np.random.permutation(n)
    solutions.append(p4)
    
    # Solution 5: Greedy construction
    p5 = np.zeros(n, dtype=np.int64)
    used_locations = set()
    remaining_facilities = list(range(n))
    
    for _ in range(n):
        if time.time() - start_time > time_limit:
            break
        # Pick the facility with highest flow to unassigned locations
        best_fac = -1
        best_score = -np.inf
        best_loc = -1
        
        for fac in remaining_facilities:
            for loc in range(n):
                if loc in used_locations:
                    continue
                # Score: sum of flows to already assigned facilities * distances to their locations
                score = 0
                for other_fac in range(n):
                    if other_fac not in remaining_facilities and other_fac != fac:
                        other_loc = p5[other_fac]
                        score += a_f[fac, other_fac] * b_f[loc, other_loc]
                
                if score > best_score:
                    best_score = score
                    best_fac = fac
                    best_loc = loc
        
        if best_fac >= 0:
            p5[best_fac] = best_loc
            used_locations.add(best_loc)
            remaining_facilities.remove(best_fac)
    
    if len(used_locations) == n:
        solutions.append(p5)
    
    # Local search on each solution
    best_p = solutions[0]
    best_cost = compute_cost(best_p)
    
    for p_init in solutions:
        if time.time() - start_time > time_limit:
            break
        
        cost_init = compute_cost(p_init)
        p_local, cost_local = steepest_descent_2opt(p_init, cost_init, time_limit, start_time)
        
        if cost_local < best_cost:
            best_cost = cost_local
            best_p = p_local
    
    # Final polish with more iterations if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        # Try a few random perturbations and local search
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            
            p_pert = best_p.copy()
            # Perturb: swap 2-3 random pairs
            for _ in range(random.randint(2, 4)):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            cost_pert = compute_cost(p_pert)
            p_final, cost_final = steepest_descent_2opt(p_pert, cost_pert, time_limit, start_time)
            
            if cost_final < best_cost:
                best_cost = cost_final
                best_p = p_final
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
