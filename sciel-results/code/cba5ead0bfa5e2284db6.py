import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute diagonal of a and b for fast delta calculation
    a_diag = np.diag(a)
    b_diag = np.diag(b)

    # Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        B_perm = b[np.ix_(p, p)]
        return np.sum(a * B_perm)
        
    current_p = p.copy()
    current_cost = calc_cost(current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    def local_search(p, cost, deadline):
        n = len(p)
        improved = True
        # Precompute rows of a? a is fixed.
        # We need a[i,:] - a[j,:]
        
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                ai = a[i, :]
                ai_diag = a_diag[i]
                Li = p[i]
                # b[L_i, p] is the row of b permuted by p
                # We can precompute b_rows_permuted = b[:, p] ? No, b[p[i], p]
                # b is (n, n). b[p[i], :] is a row.
                # We need b[p[i], p[k]] for all k.
                # This is b[p[i], p].
                
                for j in range(i + 1, n):
                    aj = a[j, :]
                    aj_diag = a_diag[j]
                    Lj = p[j]
                    
                    # Vector V = b[Lj, p] - b[Li, p]
                    # b[Lj, p] is b[p[j], p]
                    # b[Li, p] is b[p[i], p]
                    # Note: accessing b[p[j], p] creates a new array or view?
                    # In NumPy, b[p[j], p] is advanced indexing, creates a copy.
                    # This is O(n).
                    # We do this for every pair (i,j). O(n^3) total memory ops?
                    # Maybe we can precompute B_perm = b[:, p]? No, p changes.
                    # However, n=150 is small.
                    
                    V = b[p[j], p] - b[p[i], p]
                    U = ai - aj
                    
                    # Dot product
                    dot_prod = np.dot(U, V)
                    
                    # Subtract terms for k=i and k=j
                    # Term i: U[i] * V[i]
                    # U[i] = a[i,i] - a[j,i]
                    # V[i] = b[p[j], p[i]] - b[p[i], p[i]] = b[Lj, Li] - b[Li, Li]
                    term_i = (a[i, i] - a[j, i]) * (b[p[j], p[i]] - b[p[i], p[i]])
                    
                    # Term j: U[j] * V[j]
                    # U[j] = a[i,j] - a[j,j]
                    # V[j] = b[p[j], p[j]] - b[p[i], p[j]] = b[Lj, Lj] - b[Li, Lj]
                    term_j = (a[i, j] - a[j, j]) * (b[p[j], p[j]] - b[p[i], p[j]])
                    
                    sum_excl = dot_prod - term_i - term_j
                    
                    # Self term difference
                    # (a[i,i] - a[j,j]) * (b[Li, Li] - b[Lj, Lj])
                    self_diff = (a_diag[i] - a_diag[j]) * (b[p[i], p[i]] - b[p[j], p[j]])
                    
                    delta = 2 * sum_excl + self_diff
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                
        return p, cost

    while time.time() < deadline - 5:
        current_p, current_cost = local_search(current_p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
        
        if time.time() > deadline - 10:
            break
        
        # Random kick
        idx = np.random.choice(n, 3, replace=False)
        current_p = best_p.copy()
        current_p[idx[0]], current_p[idx[1]] = current_p[idx[1]], current_p[idx[0]]
        current_p[idx[1]], current_p[idx[2]] = current_p[idx[2]], current_p[idx[1]]
        current_cost = calc_cost(current_p)
        
    return best_p.tolist()
