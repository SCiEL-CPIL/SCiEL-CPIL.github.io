import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin within 120s limit
    
    # Helper: compute cost of permutation
    def compute_cost(pos):
        return int(np.sum(a * b[np.ix_(pos, pos)]))
    
    # Helper: compute delta for swapping positions i and j
    def swap_delta(pos, i, j):
        x = pos[i]
        y = pos[j]
        diff1 = b[y, pos] - b[x, pos]
        diff2 = b[x, pos] - b[y, pos]
        diff3 = b[pos, y] - b[pos, x]
        diff4 = b[pos, x] - b[pos, y]
        return (a[i, :] @ diff1 + a[j, :] @ diff2 + a[:, i] @ diff3 + a[:, j] @ diff4)
    
    # Helper: Compute the gradient of the QAP objective at a doubly stochastic matrix X
    # Objective: f(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # Gradient G[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    # G = A @ X @ B + B.T @ X.T @ A.T
    def compute_gradient(X):
        # Note: In QAP, the gradient for the minimization is often 2*(A X B + B^T X^T A^T)^T 
        # but for linear minimization <G, Y>, we just need the direction.
        # Let's use the standard form: G = A @ X @ B + B.T @ X.T @ A.T
        # Wait, let's derive carefully.
        # f(X) = tr(A X B X^T) ? No.
        # f(X) = sum_{i,j} a_{ij} sum_{k,l} b_{kl} x_{ik} x_{jl}
        # Let M = A X B. Then M_{il} = sum_j a_{ij} X_{jk} b_{kl}? No.
        # M_{ik} = sum_j a_{ij} X_{jk} ? No.
        # Let's stick to the matrix multiplication identity:
        # f(X) = sum_{i,j,k,l} a_{ij} b_{kl} x_{ik} x_{jl}
        # = sum_{i,k} x_{ik} (sum_{j,l} a_{ij} b_{kl} x_{jl})
        # Let C_{ik} = sum_{j,l} a_{ij} b_{kl} x_{jl}
        # C = A X B ?
        # (A X)_{ij} = sum_k a_{ik} x_{kj}. No.
        # (A X B)_{il} = sum_{j,k} a_{ij} x_{jk} b_{kl}.
        # So sum_{j,l} a_{ij} b_{kl} x_{jl} is not quite A X B.
        # Actually, if we define the cost as sum_{i,j} a_{ij} d_{\pi(i), \pi(j)},
        # and X is permutation matrix, X_{i, \pi(i)} = 1.
        # Cost = sum_{i,j,k,l} a_{ij} b_{kl} x_{ik} x_{jl}.
        # Gradient G_{ik} = sum_{j,l} a_{ij} b_{kl} x_{jl} + sum_{j,l} a_{ji} b_{lk} x_{jl} (symmetry of a and b usually, but not guaranteed in general QAP, though tai256c is symmetric).
        # Assuming symmetric a and b:
        # G_{ik} = 2 * sum_{j,l} a_{ij} b_{kl} x_{jl}
        # Let Y = A X B.
        # Y_{il} = sum_{j,k} a_{ij} x_{jk} b_{kl}.
        # This matches the term if we swap indices in the sum.
        # So G = 2 * (A X B) ?
        # Let's check dimensions. A(n,n) X(n,n) B(n,n) -> n,n.
        # G_{ik} = 2 * sum_{j,l} a_{ij} b_{kl} x_{jl}.
        # (A X B)_{ik} = sum_{j,l} a_{ij} x_{jl} b_{lk}. Since b is symmetric, b_{lk}=b_{kl}.
        # So G = 2 * A @ X @ B.
        
        # However, if a and b are not symmetric, we need:
        # G = A @ X @ B + B.T @ X.T @ A.T
        # For symmetric matrices, this is 2 * A @ X @ B.
        
        return 2.0 * (a @ X @ b)
    
    # Frank-Wolfe on Birkhoff polytope
    def frank_wolfe_init():
        """
        Use Frank-Wolfe method on the Birkhoff polytope to find a good
        doubly stochastic matrix, then extract a permutation.
        """
        rng = np.random.default_rng(123)
        
        # Start with a random permutation as initial point
        init_perm = rng.permutation(n)
        X = np.zeros((n, n))
        X[init_perm, np.arange(n)] = 1.0
        
        # Frank-Wolfe iterations
        best_X = X.copy()
        best_cost = compute_cost(init_perm)
        
        for it in range(100):
            if time.time() > deadline - 60.0:
                break
            
            # Compute gradient
            G = compute_gradient(X)
            
            # Find linear minimizer: min_{Y in Birkhoff} <G, Y>
            # This is the linear assignment problem: min sum G[i,k] Y[i,k]
            row_idx, col_idx = linear_sum_assignment(G)
            Y = np.zeros((n, n))
            Y[row_idx, col_idx] = 1.0
            
            # Line search for step size gamma in [0, 1]
            # f(X) = <A, X @ B @ X.T> approx? No, use exact cost of projection if needed, 
            # but FW usually minimizes the smooth objective f(X).
            # We will evaluate the cost of the permutation extracted from X_new.
            # To be safe, we can do a few trial gammas.
            
            best_gamma = 0.0
            best_perm_cost = best_cost
            best_X_new = X.copy()
            
            # Try a few step sizes: 1.0 (full step), 0.5, 0.25, and the standard 2/(t+2)
            gammas = [1.0, 0.5, 0.25, 2.0 / (it + 2.0)]
            
            for gamma in gammas:
                X_new = (1 - gamma) * X + gamma * Y
                
                # Extract permutation from X_new
                # Maximize <X_new, P> is equivalent to minimizing -<X_new, P>
                r_idx, c_idx = linear_sum_assignment(-X_new)
                perm = c_idx
                
                cost = compute_cost(perm)
                
                if cost < best_perm_cost:
                    best_perm_cost = cost
                    best_gamma = gamma
                    best_X_new = X_new
            
            # Update X
            if best_gamma > 0:
                X = best_X_new
                if best_perm_cost < best_cost:
                    best_cost = best_perm_cost
                    best_X = X.copy()
            else:
                # If no improvement with these steps, break or use small step
                # Standard FW guarantees convergence, but for discrete extraction, we might stall.
                # If we stall, we can just break and rely on local search.
                break

        # Final extraction
        row_idx, col_idx = linear_sum_assignment(-X)
        return col_idx, best_cost
    
    # Local search with 2-opt and Hungarian moves
    def local_search(pos, max_iter=5000):
        """
        Hybrid local search:
        1. Full sweep 2-opt (first improvement)
        2. Periodic Hungarian move (global reassignment based on gradient)
        """
        pos = pos.copy()
        current_cost = compute_cost(pos)
        rng = np.random.default_rng(456)
        
        # Precompute indices for vectorized swap delta?
        # For n=256, O(n^2) delta calculation is 65k ops. 
        # We can vectorize the delta calculation for all j for a fixed i.
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Phase 1: 2-opt sweep
            # To make it faster, we can check random pairs first, or do a full sweep.
            # Full sweep: for i in range(n), for j in range(i+1, n).
            # 256*255/2 ~ 32k pairs.
            # Vectorizing delta for all j given i:
            # delta(i, j) depends on pos[i], pos[j].
            # Let x = pos[i].
            # We want to find j such that swapping i and j improves cost.
            
            # We will do a few passes of "best improvement" or "first improvement".
            # Given time constraints, a single full pass of first-improvement is good.
            
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                
                x = pos[i]
                
                # Calculate deltas for all j != i
                # delta(i, j) = a[i, :] @ (b[pos[j], pos] - b[x, pos]) + 
                #              a[j, :] @ (b[x, pos] - b[pos[j], pos]) +
                #              a[:, i] @ (b[pos, pos[j]] - b[pos, x]) +
                #              a[:, j] @ (b[pos, x] - b[pos, pos[j]])
                
                # This is hard to vectorize over j easily because a[j, :] and a[:, j] vary.
                # However, we can compute the "marginal cost" of moving facility i to location k.
                # Let's stick to the swap_delta function but call it in a loop.
                # 32k calls * ~10 ops = 320k ops. In Python, this is slow (~0.1s per pass?).
                # 100 passes = 10s. This might be too slow.
                
                # Optimization: Vectorize the swap delta calculation for all j.
                # Let P = pos.
                # b_x = b[x, P] (shape n)
                # b_P_x = b[P, x] (shape n)
                # For each j, we need b[P[j], P] and b[P, P[j]].
                # This requires gathering rows/cols of b.
                
                # Alternative: Use the "Marginal Cost" approach.
                # Cost = sum_{u,v} a[u,v] b[P[u], P[v]]
                # If we swap i and j, only terms involving i or j change.
                # Delta = 
                #   a[i,j] * (b[P[j], P[i]] - b[P[i], P[j]])  # Term (i,j) and (j,i)
                #   + sum_{k != i,j} a[i,k] * (b[P[j], P[k]] - b[P[i], P[k]])
                #   + sum_{k != i,j} a[j,k] * (b[P[i], P[k]] - b[P[j], P[k]])
                #   + sum_{k != i,j} a[k,i] * (b[P[k], P[j]] - b[P[k], P[i]])
                #   + sum_{k != i,j} a[k,j] * (b[P[k], P[i]] - b[P[k], P[j]])
                
                # Since a and b are symmetric:
                # Delta = 2 * a[i,j] * (b[P[j], P[i]] - b[P[i], P[j]]) # 0 if symmetric b
                #         + 2 * sum_{k != i,j} (a[i,k] + a[k,i]) * ... ? No.
                
                # Let's just use the vectorized swap_delta if possible, or optimize the loop.
                # For n=256, a simple python loop of 32k iterations is ~5-10ms.
                # 50 iterations = 250ms. This is acceptable.
                
                for j in range(i + 1, n):
                    delta = swap_delta(pos, i, j)
                    if delta < 0:
                        pos[i], pos[j] = pos[j], pos[i]
                        current_cost += delta
                        improved = True
                        # After swap, pos[i] changed, so we should restart i loop? 
                        # Or just continue. First improvement usually continues.
                        # To be safe, restart i loop if we want best improvement.
                        # For speed, we just continue.
            
            if not improved:
                # If 2-opt is stuck, try a Hungarian move
                # Linearize the QAP around current permutation P.
                # Gradient G = 2 * A @ X @ B where X is perm matrix.
                X = np.zeros((n, n))
                X[np.arange(n), pos] = 1.0 # Wait, X[i, pos[i]] = 1? 
                # Definition: Facility i at pos[i]. So X[i, pos[i]] = 1.
                # My previous FW used X[init_perm, np.arange(n)] = 1. 
                # Let's check consistency.
                # In FW: X[init_perm, np.arange(n)] = 1.0 -> X[perm[i], i] = 1.
                # This means Location i has Facility perm[i].
                # In solve return: p[i] is location for facility i.
                # So X[facility, location] = 1.
                # My local_search pos: pos[i] is location for facility i.
                # So X[i, pos[i]] = 1.
                
                # Recalculate X for gradient
                X = np.zeros((n, n))
                X[np.arange(n), pos] = 1.0
                
                G = compute_gradient(X)
                
                # Solve LASSO for G
                r_idx, c_idx = linear_sum_assignment(G)
                new_pos = c_idx.copy()
                
                new_cost = compute_cost(new_pos)
                
                if new_cost < current_cost:
                    pos = new_pos
                    current_cost = new_cost
                    improved = True
                else:
                    # No improvement, break
                    break
        
        return pos, current_cost
    
    # Phase 1: Frank-Wolfe initialization
    pos_fw, cost_fw = frank_wolfe_init()
    
    # Phase 2: Local search from FW solution
    pos_ls, cost_ls = local_search(pos_fw, max_iter=20)
    
    best_pos = pos_ls
    best_cost = cost_ls
    
    # Phase 3: Iterated local search with perturbations
    rng = np.random.default_rng(789)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb best solution
        pos = best_pos.copy()
        # Stronger perturbation: 3-opt or more swaps
        n_swaps = n // 4
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            pos[i], pos[j] = pos[j], pos[i]
        
        # Local search
        pos, cost = local_search(pos, max_iter=20)
        
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    
    # Final polish
    best_pos, best_cost = local_search(best_pos, max_iter=10)
    
    return best_pos.tolist()
