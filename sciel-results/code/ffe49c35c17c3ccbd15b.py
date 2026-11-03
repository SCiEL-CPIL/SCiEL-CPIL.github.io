import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0
    
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    def calculate_cost(p):
        B_perm = B[np.ix_(p, p)]
        return np.sum(A * B_perm)

    def greedy_init():
        p = np.zeros(n, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        flow_sums = A.sum(axis=1)
        order = np.argsort(-flow_sums)
        
        for idx in order:
            best_j = -1
            best_cost = np.inf
            for j in range(n):
                if assigned[j]:
                    continue
                # Vectorized cost calculation for greedy step
                # Cost = sum_{k assigned} A[idx, k] B[j, p[k]] + A[k, idx] B[p[k], j]
                # Let's just use the loop for simplicity in greedy, N=100 is small
                cost = 0
                for k in range(n):
                    if assigned[k]:
                        cost += A[idx, k] * B[j, p[k]]
                        cost += A[k, idx] * B[p[k], j]
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            p[idx] = best_j
            assigned[best_j] = True
        return p

    def rank_matching_init():
        heavy = np.argsort(-A.sum(axis=1), kind="stable")
        central = np.argsort(B.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def il_search(p_init):
        p = p_init.copy()
        current_cost = calculate_cost(p)
        best_cost = current_cost
        best_p = p.copy()
        
        # Precompute B rows for p? No, p changes.
        
        def delta_swap(i, j):
            u = p[i]
            v = p[j]
            if u == v:
                return 0.0
            
            # Vectorized sums
            # S1 = sum_{k != i, j} A[i, k] B[v, p[k]]
            # S1_all = dot(A[i], B[v, p])
            # Note: B[v, p] creates an array of length n where element k is B[v, p[k]]
            
            # We can precompute B_rows_for_p? No, p changes.
            # But we can compute the dot products.
            
            # To avoid creating B[v, p] repeatedly, we can use matrix multiplication?
            # No, simple dot is fine.
            
            S1_all = np.dot(A[i], B[v, p])
            S2_all = np.dot(A[j], B[u, p])
            S3_all = np.dot(A[:, i], B[p, v]) # A[:, i] is column i. B[p, v] is array B[p[0], v], ...
            S4_all = np.dot(A[:, j], B[p, u])
            
            S5_all = np.dot(A[i], B[u, p])
            S6_all = np.dot(A[j], B[v, p])
            S7_all = np.dot(A[:, i], B[p, u])
            S8_all = np.dot(A[:, j], B[p, v])
            
            # Subtract i and j terms
            # For S1: k=i -> A[i,i] B[v, u]; k=j -> A[i,j] B[v, v]
            S1 = S1_all - A[i, i] * B[v, u] - A[i, j] * B[v, v]
            # For S2: k=i -> A[j,i] B[u, u]; k=j -> A[j,j] B[u, v]
            S2 = S2_all - A[j, i] * B[u, u] - A[j, j] * B[u, v]
            # For S3: k=i -> A[i,i] B[u, v]; k=j -> A[j,i] B[v, v]
            S3 = S3_all - A[i, i] * B[u, v] - A[j, i] * B[v, v]
            # For S4: k=i -> A[i,j] B[u, u]; k=j -> A[j,j] B[v, u]
            S4 = S4_all - A[i, j] * B[u, u] - A[j, j] * B[v, u]
            
            # For S5: k=i -> A[i,i] B[u, u]; k=j -> A[i,j] B[u, v]
            S5 = S5_all - A[i, i] * B[u, u] - A[i, j] * B[u, v]
            # For S6: k=i -> A[j,i] B[v, u]; k=j -> A[j,j] B[v, v]
            S6 = S6_all - A[j, i] * B[v, u] - A[j, j] * B[v, v]
            # For S7: k=i -> A[i,i] B[u, u]; k=j -> A[j,i] B[v, u]
            S7 = S7_all - A[i, i] * B[u, u] - A[j, i] * B[v, u]
            # For S8: k=i -> A[i,j] B[u, v]; k=j -> A[j,j] B[v, v]
            S8 = S8_all - A[i, j] * B[u, v] - A[j, j] * B[v, v]
            
            delta = S1 + S2 + S3 + S4 - (S5 + S6 + S7 + S8)
            
            # Pair terms
            delta += A[i, j] * B[v, u] + A[j, i] * B[u, v]
            delta -= A[i, j] * B[u, v] + A[j, i] * B[v, u]
            
            return delta

        for _ in range(10000):
            if time.time() > deadline:
                break
            
            best_delta = 0.0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    d = delta_swap(i, j)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_i == -1:
                # Perturb
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
                current_cost = calculate_cost(p)
            else:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        return best_p, best_cost

    candidates = [rank_matching_init(), greedy_init()]
    for _ in range(5):
        candidates.append(np.random.permutation(n))
    
    best_overall_p = None
    best_overall_cost = np.inf
    
    for p_init in candidates:
        if time.time() > deadline:
            break
        p_res, cost_res = il_search(p_init)
        if cost_res < best_overall_cost:
            best_overall_cost = cost_res
            best_overall_p = p_res
            
    if best_overall_p is not None and time.time() < deadline - 10.0:
        p_polish, cost_polish = il_search(best_overall_p)
        if cost_polish < best_overall_cost:
            best_overall_p = p_polish
            best_overall_cost = cost_polish

    if best_overall_p is None:
        return rank_matching_init().tolist()
        
    return best_overall_p.tolist()
