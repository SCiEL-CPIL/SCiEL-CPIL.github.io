import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    # Precompute some things? 
    # a and b are numpy arrays.
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate delta for swapping p[i] and p[j]
                # Let u = p[i], v = p[j]
                # The change is due to:
                # 1. i interacting with k (k != i, j): a[i,k] * (b[p[k], v] - b[p[k], u])
                # 2. j interacting with k (k != i, j): a[j,k] * (b[p[k], u] - b[p[k], v])
                # 3. i interacting with j: a[i,j] * (b[v, u] - b[u, v])
                # 4. j interacting with i: a[j,i] * (b[u, v] - b[v, u])
                # 5. i interacting with i: a[i,i] * (b[v, v] - b[u, u])
                # 6. j interacting with j: a[j,j] * (b[u, u] - b[v, v])
                
                u = p[i]
                v = p[j]
                
                # Vectorized sum for k != i, j
                # We need sum_{k != i, j} a[i,k] b[p[k], v] - sum_{k != i, j} a[i,k] b[p[k], u]
                # Let's create a mask or just sum all and subtract i, j terms.
                
                # Sum all k:
                # sum_i_v = np.dot(a[i, :], b[p, v])
                # sum_i_u = np.dot(a[i, :], b[p, u])
                # But we must exclude k=i and k=j.
                
                # b[p, v] is a vector of length n where element k is b[p[k], v]
                # b[p, u] is a vector of length n where element k is b[p[k], u]
                
                # Let's precompute rows of b indexed by p? No, p changes.
                # b is fixed. p is the permutation.
                # b[p, v] means for each k, b[p[k], v].
                
                # This requires gathering rows of b.
                # b_rows = b[p, :] -> shape (n, n). Row k is b[p[k], :].
                # Then b_rows[:, v] is the vector [b[p[0], v], ..., b[p[n-1], v]].
                
                # Doing this inside the loop is expensive (O(n^2) per swap).
                # We need a better way.
                
                # Alternative:
                # Delta = 2 * sum_{k != i, j} (a[i,k] + a[j,k]) * (b[p[k], v] - b[p[k], u]) 
                #         + 4 * a[i,j] * (b[v, u] - b[u, v])  # if a is symmetric
                #         + (a[i,i] + a[j,j]) * (b[v, v] + b[u, u] - 2*b[u, v]) # if b symmetric?
                
                # Let's assume a and b are symmetric (standard QAP).
                # If not symmetric, the formula is more complex.
                # wil100 is symmetric.
                
                # To avoid O(n^2) per swap, we can precompute the vector of differences.
                # diff_k = b[p[k], v] - b[p[k], u]
                # This still requires accessing b[p[k], v] for all k.
                
                # Given n=100, O(n^2) per swap is 10,000 ops.
                # 5000 swaps * 10,000 ops = 50,000,000 ops per pass.
                # In Python, this might be 5-10 seconds per pass.
                # 120s allows ~10-20 passes. This is acceptable.
                
                # Let's optimize the inner loop with NumPy.
                
                # Get vectors
                # b_col_v = b[:, v]
                # b_col_u = b[:, u]
                # We need b[p[k], v] which is b_col_v[p]
                # b[p[k], u] which is b_col_u[p]
                
                b_pv = b[:, v][p]
                b_pu = b[:, u][p]
                
                # Sum for i
                # sum_i = dot(a[i, :], b_pv) - a[i, i]*b[u, v] - a[i, j]*b[v, v]
                # sum_j = dot(a[j, :], b_pu) - a[j, j]*b[v, u] - a[j, i]*b[u, u]
                
                # Actually, simpler:
                # term_ik = np.dot(a[i, :], b_pv)
                # term_jk = np.dot(a[j, :], b_pu)
                
                # We need to subtract the i and j terms from the sum.
                # For i: k=i and k=j are included in dot.
                # dot includes a[i,i]*b[p[i], v] = a[i,i]*b[u, v] and a[i,j]*b[p[j], v] = a[i,j]*b[v, v].
                # We want sum_{k != i, j}. So subtract these two.
                
                sum_ik = np.dot(a[i, :], b_pv) - a[i, i]*b[u, v] - a[i, j]*b[v, v]
                sum_jk = np.dot(a[j, :], b_pu) - a[j, j]*b[v, u] - a[j, i]*b[u, u]
                
                # The change in cost for swapping i and j:
                # Original terms involving i or j:
                # Sum_{k} a[i,k] b[p[i], p[k]] + Sum_{k} a[j,k] b[p[j], p[k]]
                # = Sum_{k} a[i,k] b[u, p[k]] + Sum_{k} a[j,k] b[v, p[k]]
                
                # New terms:
                # Sum_{k} a[i,k] b[v, p[k]] + Sum_{k} a[j,k] b[u, p[k]]
                
                # Delta = (Sum_{k} a[i,k] b[v, p[k]] + Sum_{k} a[j,k] b[u, p[k]]) - (Sum_{k} a[i,k] b[u, p[k]] + Sum_{k} a[j,k] b[v, p[k]])
                
                # Note that k=i and k=j are included in these sums.
                # Let's just compute the full sums.
                
                new_sum_i = np.dot(a[i, :], b[:, v][p]) # This is Sum_{k} a[i,k] b[v, p[k]]
                new_sum_j = np.dot(a[j, :], b[:, u][p]) # This is Sum_{k} a[j,k] b[u, p[k]]
                
                old_sum_i = np.dot(a[i, :], b[:, u][p]) # This is Sum_{k} a[i,k] b[u, p[k]]
                old_sum_j = np.dot(a[j, :], b[:, v][p]) # This is Sum_{k} a[j,k] b[v, p[k]]
                
                delta = (new_sum_i + new_sum_j) - (old_sum_i + old_sum_j)
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # Restarting the inner loop is optional, but helps convergence
                    # break # If we break, we restart i loop. 
                    # If we don't break, we continue with next j.
                    # Let's not break to allow multiple swaps in one pass.
        
    return p
