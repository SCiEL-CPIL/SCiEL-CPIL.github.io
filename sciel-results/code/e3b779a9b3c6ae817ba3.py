import time
import numpy as np
from math import gcd

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    if time.time() - start_time > 5.0:
        return {"U": U_base, "V": V_base, "W": W_base}

    best_solution = None
    best_rank = rank_baseline
    deadline = start_time + time_limit - 5.0
    
    # Search for lower rank
    # We iterate R from 59 down to 45
    for R in range(rank_baseline - 1, 44, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        trials = 2000 if R > 55 else 1000
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Random U, V with entries from {-1, 0, 1}
            # Using a fixed seed for reproducibility in testing, but random in production
            # np.random.seed(t) # Uncomment for debugging
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            W_candidate = np.zeros((p * n, R), dtype=float)
            valid = True
            
            # Pre-compute M[c, i, k] = sum_j U[i*m+j, c] * V[j*p+k, c]
            M = np.zeros((R, n, p), dtype=int)
            for i in range(n):
                for k in range(p):
                    for c in range(R):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j, c] * V[j * p + k, c]
                        M[c, i, k] = s
            
            for i in range(n):
                for k in range(p):
                    M_vec = M[:, i, k]
                    
                    # Try to find a sparse integer solution for 2 * W
                    # sum_c M_vec[c] * W'_c = 2
                    # Try 1 non-zero
                    found = False
                    for c in range(R):
                        if M_vec[c] == 1:
                            W_candidate[k * n + i, c] = 1.0
                            found = True
                            break
                        elif M_vec[c] == -1:
                            W_candidate[k * n + i, c] = -1.0
                            found = True
                            break
                        elif M_vec[c] == 2:
                            W_candidate[k * n + i, c] = 0.5
                            found = True
                            break
                        elif M_vec[c] == -2:
                            W_candidate[k * n + i, c] = -0.5
                            found = True
                            break
                    
                    if not found:
                        # Try 2 non-zeros
                        for c1 in range(R):
                            if M_vec[c1] == 0:
                                continue
                            for c2 in range(c1 + 1, R):
                                if M_vec[c2] == 0:
                                    continue
                                m1 = M_vec[c1]
                                m2 = M_vec[c2]
                                # m1 * w1 + m2 * w2 = 2
                                # Check if solvable in integers
                                g = gcd(abs(m1), abs(m2))
                                if 2 % g == 0:
                                    # Find a solution
                                    # Use extended gcd
                                    def egcd(a, b):
                                        if b == 0:
                                            return a, 1, 0
                                        g, x1, y1 = egcd(b, a % b)
                                        return g, y1, x1 - (a // b) * y1
                                    
                                    g, x, y = egcd(m1, m2)
                                    # m1 * x + m2 * y = g
                                    # Multiply by 2/g
                                    w1 = x * (2 // g)
                                    w2 = y * (2 // g)
                                    # Check if half-integers (they are integers, so yes)
                                    W_candidate[k * n + i, c1] = w1 / 2.0
                                    W_candidate[k * n + i, c2] = w2 / 2.0
                                    found = True
                                    break
                            if found:
                                break
                    
                    if not found:
                        valid = False
                        break
                        
                if not valid:
                    break
            
            if valid:
                # Verify the full tensor equation
                valid_full = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U[i * m + j, c] * V[j * p + k, c] * W_candidate[k * n + i, c]
                            if abs(s - 1) > 1e-9:
                                valid_full = False
                                break
                        if not valid_full:
                            break
                    if not valid_full:
                        break
                
                if valid_full:
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_candidate.tolist()
                    }
                    best_rank = R
                    break

    if best_solution is not None:
        return best_solution
    else:
        return {"U": U_base, "V": V_base, "W": W_base}
