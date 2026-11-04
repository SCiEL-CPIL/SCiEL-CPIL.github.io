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
    
    # Strategy 1: Column pruning from schoolbook
    # Schoolbook has n*m*p = 36 columns
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
    
    # Try to merge columns by solving for combined columns
    # If we have two columns c1, c2, can we replace them with one column?
    # We need to find u, v, w such that for all (a,b,g):
    #   U[a,c1]*V[b,c1]*W[g,c1] + U[a,c2]*V[b,c2]*W[g,c2] = u[a]*v[b]*w[g]
    
    # More generally: try to find a rank-r decomposition by random search
    # with the constraint that entries are in {-1, 0, 1, 1/2, -1/2}
    
    def random_search(r, max_attempts=200):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random sparse initialization from half-integers
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(n * m)])
            V = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(m * p)])
            
            # Solve for W using linear systems
            W = np.zeros((p * n, r))
            valid = True
            
            for g in range(p * n):
                # Build constraint matrix for this gamma
                rows = []
                rhs = []
                for (a, b, target) in valid_indices:
                    if target == 1 and g == target:  # This is wrong, let me fix
                        pass
                # Better: group by gamma
                pass
            
            # Let me redo this properly
            # For each gamma g, constraints are:
            # sum_c U[a,c]*V[b,c]*W[g,c] = T[a,b,g]
            # where T[a,b,g] = 1 if (a,b,g) in valid_indices, else 0
            
            for g in range(p * n):
                A_g = []
                b_g = []
                for a in range(n * m):
                    for b in range(m * p):
                        row = U[a, :] * V[b, :]
                        A_g.append(row)
                        target = 1.0 if (a, b, g) in valid_set else 0.0
                        b_g.append(target)
                
                A_g = np.array(A_g)
                b_g = np.array(b_g)
                
                try:
                    w_g, residuals, rank, sv = np.linalg.lstsq(A_g, b_g, rcond=None)
                except:
                    valid = False
                    break
                
                if residuals is None or len(residuals) == 0 or residuals[0] > 1e-9:
                    valid = False
                    break
                    
                W[g, :] = w_g
            
            if not valid:
                continue
            
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            if verify(U_s, V_s, W_s):
                return (U_s, V_s, W_s)
        
        return None
    
    # Try ranks from low to schoolbook
    for r in range(20, rank_school + 1):
        if time.time() > deadline:
            break
        
        result = random_search(r, max_attempts=300)
        if result is not None:
            U_s, V_s, W_s = result
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
