import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Ensure we work with arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    def calc_cost(p_list):
        # p_list is permutation
        # cost = sum a[i,j] b[p[i], p[j]]
        # Vectorized:
        # B_p = b[np.ix_(p_list, p_list)]
        # return np.sum(a * B_p)
        pass

    # Faster cost calculation
    def get_cost(p_list):
        # Use matrix multiplication trick?
        # C = sum a[i,j] b[p[i], p[j]]
        # Let P be permutation matrix. C = tr(A P B P^T)
        # For n=100, direct sum is fine.
        # a * b[p][:, p]
        # b_p = b[np.array(p_list)][:, np.array(p_list)]
        # return np.sum(a * b_p)
        p_arr = np.array(p_list)
        b_p = b[p_arr][:, p_arr]
        return np.sum(a * b_p)

    best_p = p.copy()
    best_cost = get_cost(p)
    current_cost = best_cost
    
    start_time = time.time()
    deadline = start_time + 110
    
    T = 500.0
    T_min = 1.0
    cooling = 0.999
    
    # Precompute row sums? No.
    
    while T > T_min and time.time() < deadline:
        best_delta = float('inf')
        best_i, best_j = -1, -1
        
        # Steepest Descent
        for i in range(n):
            k = p[i]
            Bk = b[k]
            Ai = a[:, i]
            Aii = a[i, i]
            Bkk = b[k, k]
            
            for j in range(i+1, n):
                l = p[j]
                Bl = b[l]
                Aj = a[:, j]
                Ajj = a[j, j]
                Bll = b[l, l]
                
                v = Bl - Bk
                w = v[p]
                
                sum_all = np.dot(Ai - Aj, w)
                
                term_i = (Ai[i] - Aj[i]) * (Bl[k] - Bk[k])
                term_j = (Ai[j] - Aj[j]) * (Bl[l] - Bk[l])
                
                sum_excl = sum_all - term_i - term_j
                
                diag_part = (Aii - Ajj) * (Bll - Bkk)
                
                delta = 2 * sum_excl + diag_part
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
            
        # Accept
        if best_delta < 0 or np.random.random() < np.exp(-best_delta / T):
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= cooling
        
    # Final Polish: Steepest Descent only
    # Run a few steps of pure descent if time remains
    while time.time() < deadline:
        improved = False
        for i in range(n):
            k = p[i]
            Bk = b[k]
            Ai = a[:, i]
            Aii = a[i, i]
            Bkk = b[k, k]
            for j in range(i+1, n):
                l = p[j]
                Bl = b[l]
                Aj = a[:, j]
                Ajj = a[j, j]
                Bll = b[l, l]
                v = Bl - Bk
                w = v[p]
                sum_all = np.dot(Ai - Aj, w)
                term_i = (Ai[i] - Aj[i]) * (Bl[k] - Bk[k])
                term_j = (Ai[j] - Aj[j]) * (Bl[l] - Bk[l])
                sum_excl = sum_all - term_i - term_j
                diag_part = (Aii - Ajj) * (Bll - Bkk)
                delta = 2 * sum_excl + diag_part
                if delta < 0:
                    p[best_i], p[best_j] = p[best_j], p[best_i] # Bug: used best_i from outer scope? No, i,j
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return best_p
