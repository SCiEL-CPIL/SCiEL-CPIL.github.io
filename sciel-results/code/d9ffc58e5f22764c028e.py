import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Safety margin: 120s limit, use 115s
    deadline = time.time() + 115.0

    # Helper: compute cost of permutation
    def compute_cost(pos):
        # pos[i] is location for facility i
        # Cost = sum_{i,j} a[i,j] * b[pos[i], pos[j]]
        return int(np.sum(a * b[np.ix_(pos, pos)]))

    # Helper: compute delta for swapping positions i and j in pos
    # pos[i] -> x, pos[j] -> y
    # New cost - Old cost
    def swap_delta(pos, i, j):
        x = pos[i]
        y = pos[j]
        # Terms involving i and j
        # Old: a[i,j]*b[x,y] + a[j,i]*b[y,x]
        #      + sum_{k!=i,j} a[i,k]*b[x,pos[k]] + a[k,i]*b[pos[k],x]
        #      + sum_{k!=i,j} a[j,k]*b[y,pos[k]] + a[k,j]*b[pos[k],y]
        # New: a[i,j]*b[y,x] + a[j,i]*b[x,y]
        #      + sum_{k!=i,j} a[i,k]*b[y,pos[k]] + a[k,i]*b[pos[k],y]
        #      + sum_{k!=i,j} a[j,k]*b[x,pos[k]] + a[k,j]*b[pos[k],x]
        
        # Vectorized calculation
        # diff_row_i = b[y, :] - b[x, :]  (change in row i's contribution to others)
        # diff_col_i = b[:, y] - b[:, x]  (change in col i's contribution from others)
        # But we need to be careful with the i,j interaction.
        
        # Let's use the formula:
        # Delta = a[i, :] @ (b[y, pos] - b[x, pos]) + a[:, i] @ (b[pos, y] - b[pos, x])
        #       + a[j, :] @ (b[x, pos] - b[y, pos]) + a[:, j] @ (b[pos, x] - b[pos, y])
        #       - 2 * (a[i, j] * b[x, y] + a[j, i] * b[y, x] - a[i, j] * b[y, x] - a[j, i] * b[x, y]) 
        #       Wait, the term a[i,j] is included in the sums above.
        # In a[i, :] @ (b[y, pos] - b[x, pos]), the j-th element is a[i,j]*(b[y, pos[j]] - b[x, pos[j]]) = a[i,j]*(b[y, y] - b[x, y]).
        # In a[j, :] @ (b[x, pos] - b[y, pos]), the i-th element is a[j,i]*(b[x, pos[i]] - b[y, pos[i]]) = a[j,i]*(b[x, x] - b[y, x]).
        # This double counts or misses the cross term correctly?
        
        # Let's stick to the simple explicit calculation for correctness, it's fast enough for n=256.
        # Actually, the vectorized version provided in the prompt is:
        # diff1 = b[y, pos] - b[x, pos]
        # diff2 = b[x, pos] - b[y, pos]
        # diff3 = b[pos, y] - b[pos, x]
        # diff4 = b[pos, x] - b[pos, y]
        # return (a[i, :] @ diff1 + a[j, :] @ diff2 + a[:, i] @ diff3 + a[:, j] @ diff4)
        
        # Let's verify this formula.
        # Term 1: a[i, :] @ diff1 = sum_k a[i,k] (b[y, pos[k]] - b[x, pos[k]])
        # Term 3: a[:, i] @ diff3 = sum_k a[k, i] (b[pos[k], y] - b[pos[k], x])
        # Term 2: a[j, :] @ diff2 = sum_k a[j,k] (b[x, pos[k]] - b[y, pos[k]])
        # Term 4: a[:, j] @ diff4 = sum_k a[k, j] (b[pos[k], x] - b[pos[k], y])
        
        # Sum = sum_k a[i,k]b[y,pos[k]] - a[i,k]b[x,pos[k]] + a[k,i]b[pos[k],y] - a[k,i]b[pos[k],x]
        #      + a[j,k]b[x,pos[k]] - a[j,k]b[y,pos[k]] + a[k,j]b[pos[k],x] - a[k,j]b[pos[k],y]
        
        # Let's group by new terms (with y where x was for i, and x where y was for j):
        # New i-row: sum_k a[i,k] b[y, pos[k]] (matches Term 1 first part)
        # New i-col: sum_k a[k,i] b[pos[k], y] (matches Term 3 first part)
        # New j-row: sum_k a[j,k] b[x, pos[k]] (matches Term 2 first part)
        # New j-col: sum_k a[k,j] b[pos[k], x] (matches Term 4 first part)
        
        # Old terms to subtract:
        # Old i-row: sum_k a[i,k] b[x, pos[k]] (matches Term 1 second part, with minus)
        # Old i-col: sum_k a[k,i] b[pos[k], x] (matches Term 3 second part, with minus)
        # Old j-row: sum_k a[j,k] b[y, pos[k]] (matches Term 2 second part, with minus)
        # Old j-col: sum_k a[k,j] b[pos[k], y] (matches Term 4 second part, with minus)
        
        # So the formula is correct.
        diff1 = b[y, pos] - b[x, pos]
        diff2 = b[x, pos] - b[y, pos]
        diff3 = b[pos, y] - b[pos, x]
        diff4 = b[pos, x] - b[pos, y]
        return (a[i, :] @ diff1 + a[j, :] @ diff2 + a[:, i] @ diff3 + a[:, j] @ diff4)

    # Helper: Compute gradient of QAP objective at doubly stochastic X
    # f(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # Gradient G[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    # G = A @ X @ B + B.T @ X.T @ A.T
    # For symmetric A and B, G = 2 * A @ X @ B
    def compute_gradient(X):
        return 2.0 * (a @ X @ b)

    # Phase 1: Frank-Wolfe initialization
    def frank_wolfe_init():
        rng = np.random.default_rng(123)
        
        # Start with a random permutation
        init_perm = rng.permutation(n)
        X = np.zeros((n, n))
        # X[i, pos[i]] = 1 means Facility i at location pos[i].
        # Wait, the prompt's FW used X[init_perm, np.arange(n)] = 1.0
        # This means X[perm[i], i] = 1. So Location i has Facility perm[i].
        # My local search uses pos[i] = location for facility i.
        # So X[facility, location] = 1.
        # If I want to be consistent with standard matrix notation where rows are facilities:
        # X[i, pos[i]] = 1.
        
        # Let's stick to X[i, pos[i]] = 1.
        X[np.arange(n), init_perm] = 1.0
        
        best_X = X.copy()
        best_cost = compute_cost(init_perm)
        
        for it in range(200):
            if time.time() > deadline - 60.0:
                break
            
            # Compute gradient
            G = compute_gradient(X)
            
            # Find linear minimizer: min <G, Y> s.t. Y in Birkhoff
            row_idx, col_idx = linear_sum_assignment(G)
            Y = np.zeros((n, n))
            Y[row_idx, col_idx] = 1.0
            
            # Line search
            best_gamma = 0.0
            best_perm_cost = best_cost
            best_X_new = X.copy()
            
            gammas = [1.0, 0.5, 0.25, 2.0 / (it + 2.0)]
            
            for gamma in gammas:
                X_new = (1 - gamma) * X + gamma * Y
                
                # Extract permutation from X_new
                # We want to find a permutation P such that <X_new, P> is maximized
                # which is equivalent to minimizing -<X_new, P>
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
                break

        # Final extraction
        row_idx, col_idx = linear_sum_assignment(-X)
        return col_idx, best_cost

    # Phase 2: Local Search
    def local_search(pos, max_iter=100):
        pos = pos.copy()
        current_cost = compute_cost(pos)
        rng = np.random.default_rng(456)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # 2-opt sweep (first improvement)
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                x = pos[i]
                for j in range(i + 1, n):
                    delta = swap_delta(pos, i, j)
                    if delta < 0:
                        pos[i], pos[j] = pos[j], pos[i]
                        current_cost += delta
                        improved = True
            
            if not improved:
                # Try Hungarian move
                X = np.zeros((n, n))
                X[np.arange(n), pos] = 1.0
                G = compute_gradient(X)
                r_idx, c_idx = linear_sum_assignment(G)
                new_pos = c_idx.copy()
                new_cost = compute_cost(new_pos)
                
                if new_cost < current_cost:
                    pos = new_pos
                    current_cost = new_cost
                    improved = True
                else:
                    break
        
        return pos, current_cost

    # Main execution
    # 1. FW Init
    pos_fw, cost_fw = frank_wolfe_init()
    
    # 2. Local Search
    pos_ls, cost_ls = local_search(pos_fw, max_iter=50)
    
    best_pos = pos_ls
    best_cost = cost_ls
    
    # 3. Iterated Local Search
    rng = np.random.default_rng(789)
    
    while time.time() < deadline - 10.0:
        # Perturb
        pos = best_pos.copy()
        n_swaps = n // 8  # Moderate perturbation
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            pos[i], pos[j] = pos[j], pos[i]
        
        # Local Search
        pos, cost = local_search(pos, max_iter=20)
        
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    
    # Final polish
    best_pos, best_cost = local_search(best_pos, max_iter=10)
    
    return best_pos.tolist()
