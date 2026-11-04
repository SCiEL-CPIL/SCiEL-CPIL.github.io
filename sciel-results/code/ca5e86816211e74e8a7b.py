"""QAP tai256c solver: Iterated Local Search with First-Improvement 2-opt and Hungarian initialization."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0  # 2s safety margin

    def compute_cost(p):
        # p is a permutation of 0..n-1
        # Cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
        return np.sum(a * b[np.ix_(p, p)])

    def swap_delta(p, i, j):
        """
        Calculate the change in cost if we swap facilities at positions i and j.
        p is the current permutation.
        """
        # u and v are the facility IDs currently at positions i and j
        u = p[i]
        v = p[j]
        
        # We need to calculate the difference in cost.
        # Let C be the cost.
        # C = sum_{k,l} a[k,l] b[p[k], p[l]]
        # Swapping p[i] and p[j] changes terms involving i or j.
        
        # Terms involving i or j in the sum:
        # 1. k=i, l=i: a[i,i] b[u,u] -> a[i,i] b[v,v]
        # 2. k=i, l=j: a[i,j] b[u,v] -> a[i,j] b[v,u]
        # 3. k=j, l=i: a[j,i] b[v,u] -> a[j,i] b[u,v]
        # 4. k=j, l=j: a[j,j] b[v,v] -> a[j,j] b[u,u]
        # 5. k=i, l=k (k!=i,j): a[i,k] b[u,p[k]] -> a[i,k] b[v,p[k]]
        # 6. k=j, l=k (k!=i,j): a[j,k] b[v,p[k]] -> a[j,k] b[u,p[k]]
        # 7. k=k (k!=i,j), l=i: a[k,i] b[p[k],u] -> a[k,i] b[p[k],v]
        # 8. k=k (k!=i,j), l=j: a[k,j] b[p[k],v] -> a[k,j] b[p[k],u]
        
        # Let's group them:
        # Delta = 
        #   a[i,i](b[v,v] - b[u,u]) +
        #   a[i,j](b[v,u] - b[u,v]) +
        #   a[j,i](b[u,v] - b[v,u]) +
        #   a[j,j](b[u,u] - b[v,v]) +
        #   sum_{k!=i,j} a[i,k](b[v,p[k]] - b[u,p[k]]) +
        #   sum_{k!=i,j} a[j,k](b[u,p[k]] - b[v,p[k]]) +
        #   sum_{k!=i,j} a[k,i](b[p[k],v] - b[p[k],u]) +
        #   sum_{k!=i,j} a[k,j](b[p[k],u] - b[p[k],v])
        
        # We can compute the sums using dot products.
        # Note: p[k] for k!=i,j is just p with elements at i and j removed, but for dot product we can use full p and subtract the i,j terms if we include them, or just use the fact that b[v, p[i]] = b[v,u] and b[u, p[i]] = b[u,u].
        
        # Let's define vectors:
        # row_i = a[i, :]
        # row_j = a[j, :]
        # col_i = a[:, i]
        # col_j = a[:, j]
        
        # b_u_p = b[u, p]  # vector of b[u, p[k]] for all k
        # b_v_p = b[v, p]  # vector of b[v, p[k]] for all k
        # b_p_u = b[p, u]  # vector of b[p[k], u] for all k
        # b_p_v = b[p, v]  # vector of b[p[k], v] for all k
        
        b_u_p = b[u, p]
        b_v_p = b[v, p]
        b_p_u = b[p, u]
        b_p_v = b[p, v]
        
        # Sum 5: dot(a[i,:], b_v_p - b_u_p)
        # This includes k=i and k=j terms.
        # Term k=i: a[i,i] (b[v,u] - b[u,u])
        # Term k=j: a[i,j] (b[v,v] - b[u,v])
        # We need to subtract these from the sum and add the correct manual terms later, or just adjust the manual terms.
        # Actually, it's easier to compute the full dot product and then adjust for the i and j indices.
        
        s5 = np.dot(a[i, :], b_v_p - b_u_p)
        # Subtract terms for k=i and k=j
        s5 -= a[i, i] * (b[v, u] - b[u, u])
        s5 -= a[i, j] * (b[v, v] - b[u, v])
        
        # Sum 6: dot(a[j,:], b_u_p - b_v_p)
        s6 = np.dot(a[j, :], b_u_p - b_v_p)
        # Subtract terms for k=i and k=j
        s6 -= a[j, i] * (b[u, u] - b[v, u])
        s6 -= a[j, j] * (b[u, v] - b[v, v])
        
        # Sum 7: dot(a[:,i], b_p_v - b_p_u)
        s7 = np.dot(a[:, i], b_p_v - b_p_u)
        # Subtract terms for k=i and k=j
        s7 -= a[i, i] * (b[v, u] - b[u, u])
        s7 -= a[j, i] * (b[v, v] - b[u, v]) # Wait, b[p[j], v] - b[p[j], u] = b[v,v] - b[v,u]
        # Correction: b[p[k], v] - b[p[k], u]
        # For k=i: b[u, v] - b[u, u]
        # For k=j: b[v, v] - b[v, u]
        s7 -= a[i, i] * (b[u, v] - b[u, u])
        s7 -= a[j, i] * (b[v, v] - b[v, u])
        
        # Sum 8: dot(a[:,j], b_p_u - b_p_v)
        s8 = np.dot(a[:, j], b_p_u - b_p_v)
        # Subtract terms for k=i and k=j
        # For k=i: b[u, u] - b[u, v]
        # For k=j: b[v, u] - b[v, v]
        s8 -= a[i, j] * (b[u, u] - b[u, v])
        s8 -= a[j, j] * (b[v, u] - b[v, v])
        
        # Manual terms 1-4:
        t1 = a[i, i] * (b[v, v] - b[u, u])
        t2 = a[i, j] * (b[v, u] - b[u, v])
        t3 = a[j, i] * (b[u, v] - b[v, u])
        t4 = a[j, j] * (b[u, u] - b[v, v])
        
        return s5 + s6 + s7 + s8 + t1 + t2 + t3 + t4

    def local_search_first_improvement(p, cost):
        """
        First-improvement 2-opt local search.
        Scans all pairs (i, j) with i < j. If a swap improves cost, perform it and restart scan.
        """
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 2:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline - 2:
                        return p, cost
                    delta = swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break  # Restart scan
        return p, cost

    # Initialization
    # 1. Hungarian on linearized cost
    C = a @ b + a.T @ b.T
    row_ind, col_ind = linear_sum_assignment(C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    # 2. Rank-based initialization
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # 3. Random initialization
    p_rand = np.random.permutation(n)
    
    # Evaluate and pick best
    cost_hung = compute_cost(p_hung)
    cost_rank = compute_cost(p_rank)
    cost_rand = compute_cost(p_rand)
    
    if cost_hung <= cost_rank and cost_hung <= cost_rand:
        p = p_hung.copy()
        best_cost = cost_hung
    elif cost_rank <= cost_hung and cost_rank <= cost_rand:
        p = p_rank.copy()
        best_cost = cost_rank
    else:
        p = p_rand.copy()
        best_cost = cost_rand
        
    best_p = p.copy()
    
    # Initial Local Search
    p, cost = local_search_first_improvement(p, best_cost)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost
        
    # Iterated Local Search
    while time.time() < deadline - 5:
        # Perturb best solution
        p = best_p.copy()
        
        # Number of random swaps: start small, increase if stuck?
        # For now, fixed small number of swaps
        n_swaps = np.random.randint(2, 6)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            
        # Local Search
        p, cost = local_search_first_improvement(p, compute_cost(p))
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            
    return best_p.tolist()
