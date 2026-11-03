"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cost(p, a, b):
    """Compute total cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    pb = b[np.array(p)][:, np.array(p)]
    return int(np.dot(a.ravel(), pb.ravel()))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # --- 1. Build linear-cost matrix for Hungarian assignment ---
    # C[i, j] = sum_k a[i, k] * b[j, k]
    # Placing facility i at location j gives contribution sum_k a[i,k]*b[j, perm[k]]
    # For a fixed permutation p, cost = sum_i sum_k a[i,k]*b[p[i], p[k]]
    # The linear relaxation: assign facility i to location j minimizing sum_k a[i,k]*b[j,k]
    # is a reasonable initial solution that captures pairwise affinities.
    C = a @ b  # C[i,j] = sum_k a[i,k]*b[j,k], shape (n,n)
    
    # Hungarian algorithm to minimize total linear cost
    row_ind, col_ind = linear_sum_assignment(C)
    # col_ind[i] = location assigned to facility i
    p_list = col_ind.tolist()
    best_cost = _cost(p_list, a, b)
    best_p = p_list.copy()

    if time.time() >= deadline - 5:
        return best_p

    # --- 2. Local search: 2-swap (pairwise exchange) with fast delta ---
    def swap_delta(i, j, p_arr, b_arr):
        """Compute delta in cost when swapping p[i] and p[j]."""
        u = p_arr[i]
        v = p_arr[j]
        if u == v:
            return 0
        # delta = sum_k (a[i,k]*b[v,p_arr[k]] + a[j,k]*b[u,p_arr[k]] - a[i,k]*b[u,p_arr[k]] - a[j,k]*b[v,p_arr[k]])
        #      = dot(a[i,:] - a[j,:], b[v,:] - b[u,:])[p_arr] + corrections for k=i and k=j
        # For k != i,j: contribution is (a[i,k]-a[j,k])*(b[v,p_arr[k]] - b[u,p_arr[k]])
        # For k = i: a[i,i]*b[v,p_arr[i]] + a[j,i]*b[u,p_arr[i]] - a[i,i]*b[u,p_arr[i]] - a[j,i]*b[v,p_arr[i]]
        #   But p_arr[i] = u and p_arr[j] = v
        #   k=i: a[i,i]*b[v,u] + a[j,i]*b[u,u] - a[i,i]*b[u,u] - a[j,i]*b[v,v]
        #   k=j: a[i,j]*b[v,v] + a[j,j]*b[u,v] - a[i,j]*b[u,v] - a[j,j]*b[v,u]
        
        diff_a = a[i, :] - a[j, :]
        diff_b = b[v, :] - b[u, :]
        
        # Full sum over all k via the product, then subtract k=i and k=j and add correct terms
        # Actually simpler: 
        # delta = sum_k diff_a[k] * diff_b[p_arr[k]] 
        #         - diff_a[i]*diff_b[u] - diff_a[j]*diff_b[v]  (since p_arr[i]=u, p_arr[j]=v)
        #         + [a[i,i]*b[v,u] + a[j,i]*b[u,u] - a[i,i]*b[u,u] - a[j,i]*b[v,v]]
        #         + [a[i,j]*b[v,v] + a[j,j]*b[u,v] - a[i,j]*b[u,v] - a[j,j]*b[v,u]]
        # Note: diff_a[i]*diff_b[u] = (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        #       diff_a[j]*diff_b[v] = (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        
        full = np.dot(diff_a, diff_b[p_arr])
        sub_i = diff_a[i] * diff_b[u]
        sub_j = diff_a[j] * diff_b[v]
        
        corr_i = a[i,i]*b[v,u] + a[j,i]*b[u,u] - a[i,i]*b[u,u] - a[j,i]*b[v,v]
        corr_j = a[i,j]*b[v,v] + a[j,j]*b[u,v] - a[i,j]*b[u,v] - a[j,j]*b[v,u]
        
        return int(full - sub_i - sub_j + corr_i + corr_j)

    p_arr = np.array(best_p, dtype=np.int64)
    cost = best_cost
    
    # Greedy 2-opt local search
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            if time.time() >= deadline - 2:
                break
            for j in range(i + 1, n):
                delta = swap_delta(i, j, p_arr, b)
                if delta < 0:
                    p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                    cost += delta
                    improved = True
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p_arr.tolist()
                    break  # restart inner loop to re-scan from beginning for better convergence
    
    # --- 3. Second pass with shuffled order if time remains ---
    if time.time() < deadline - 3:
        p_arr = np.array(best_p, dtype=np.int64)
        cost = best_cost
        rng = np.random.default_rng(42)
        order = np.arange(n)
        for _pass in range(3):
            rng.shuffle(order)
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for idx in range(n):
                    i = order[idx]
                    if time.time() >= deadline - 2:
                        break
                    for idx2 in range(idx + 1, n):
                        j = order[idx2]
                        if i == j:
                            continue
                        delta = swap_delta(i, j, p_arr, b)
                        if delta < 0:
                            p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                            cost += delta
                            improved = True
                            if cost < best_cost:
                                best_cost = cost
                                best_p = p_arr.tolist()
                            break

    # --- 4. Final: try random perturbation + local search if time allows ---
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        p_arr = np.array(best_p, dtype=np.int64)
        # Perturb: swap 10 random pairs
        for _ in range(10):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
        cost = _cost(p_arr.tolist(), a, b)
        
        # Quick local search
        for _ls_pass in range(2):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(0, n, 2):
                    if time.time() >= deadline - 2:
                        break
                    j = (i + 1) % n
                    if i == j:
                        continue
                    delta = swap_delta(i, j, p_arr, b)
                    if delta < 0:
                        p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p_arr.tolist()
                        break

    return best_p
# EVOLVE-BLOCK-END
