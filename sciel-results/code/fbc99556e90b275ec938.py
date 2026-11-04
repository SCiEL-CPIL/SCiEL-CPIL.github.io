import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build tensor structure
    # alpha = i*m + j, beta = j*p + k, gamma = k*n + i
    valid_indices = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                valid_set.add((alpha, beta, gamma))
    
    # All possible index triples
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                all_indices.append((a, b, g))
    
    invalid_indices = [idx for idx in all_indices if idx not in valid_set]
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        r = U.shape[1]
        for (a, b, g) in valid_indices:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            if abs(val - 1.0) > 1e-9:
                return False
        for (a, b, g) in invalid_indices:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            if abs(val) > 1e-9:
                return False
        return True
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def random_search(r, max_attempts=500):
        vals = [-1, -0.5, 0, 0.5, 1]
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization with sparse half-integers
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for _ in range(n * m * r):
                a = np.random.randint(n * m)
                c = np.random.randint(r)
                if np.random.random() < 0.6:
                    U[a, c] = np.random.choice(vals)
            for _ in range(m * p * r):
                b = np.random.randint(m * p)
                c = np.random.randint(r)
                if np.random.random() < 0.6:
                    V[b, c] = np.random.choice(vals)
            
            # Solve for W: for each g, sum_c U[a,c]*V[b,c]*W[g,c] = T[a,b,g]
            W = np.zeros((p * n, r))
            valid = True
            
            for g in range(p * n):
                A_g = []
                b_g = []
                for a in range(n * m):
                    for b in range(m * p):
                        row = U[a, :] * V[b, :]
                        if np.all(row == 0):
                            continue
                        A_g.append(row)
                        target = 1.0 if (a, b, g) in valid_set else 0.0
                        b_g.append(target)
                
                if not A_g:
                    continue
                
                A_g = np.array(A_g)
                b_g = np.array(b_g)
                
                try:
                    w_g, residuals, rank_g, sv = np.linalg.lstsq(A_g, b_g, rcond=None)
                except:
                    valid = False
                    break
                
                if residuals is None or len(residuals) == 0 or residuals[0] > 1e-8:
                    valid = False
                    break
                    
                W[g, :] = w_g
            
            if not valid:
                continue
            
            # Snap to half-integers and verify
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            if verify(U_s, V_s, W_s):
                return (U_s, V_s, W_s)
        
        return None
    
    # Try structured approach: use schoolbook as base and try to merge
    # Also try pure random search at various ranks
    
    best_result = None
    
    # Try ranks from low to schoolbook
    for r in range(12, rank_school + 1):
        if time.time() > deadline:
            break
        
        result = random_search(r, max_attempts=200)
        if result is not None:
            U_s, V_s, W_s = result
            best_result = (U_s, V_s, W_s)
            break
    
    if best_result is not None:
        U_s, V_s, W_s = best_result
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
    
    # Fallback to schoolbook
    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }
