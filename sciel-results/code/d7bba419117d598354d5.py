import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0
    
    # Initialization
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_facilities = np.argsort(-a_sums, kind="stable")
    central_locations = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # Precompute a_diff for cross terms?
    # a_cross[i, j] = a[i, j] - a[j, i]
    a_cross = a - a.T
    
    # a_diag[i] = a[i, i]
    a_diag = np.diag(a)
    
    # b_diag[i] = b[i, i]
    b_diag = np.diag(b)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
            
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            pi = p[i]
            ai = a[i]
            ai_diag = a_diag[i]
            
            for j in range(i + 1, n):
                pj = p[j]
                aj = a[j]
                aj_diag = a_diag[j]
                
                # Calculate Delta
                # S = sum_{all k} (ai[k] - aj[k]) * (b[pj, p[k]] - b[pi, p[k]])
                diff_a = ai - aj
                diff_b = b[pj] - b[pi]
                S = np.dot(diff_a, diff_b)
                
                # Correct for k=i and k=j
                # term_i = (ai[i] - aj[i]) * (b[pj, pi] - b[pi, pi])
                # term_j = (ai[j] - aj[j]) * (b[pj, pj] - b[pi, pj])
                
                term_i = (ai[i] - aj[i]) * (b[pj, pi] - b[pi, pi])
                term_j = (ai[j] - aj[j]) * (b[pj, pj] - b[pi, pj])
                
                sum_excl = S - term_i - term_j
                
                # Cross term: (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
                cross_term = a_cross[i, j] * (b[pj, pi] - b[pi, pj])
                
                # Diagonal term: (a[i,i] - a[j,j]) * (b[pj, pj] - b[pi, pi])
                diag_term = (ai_diag - aj_diag) * (b[pj, pj] - b[pi, pi])
                
                delta = sum_excl + cross_term + diag_term
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            # Apply best swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            improved = True
        else:
            break
            
    return p.tolist()
