import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Precompute the target tensor T as a dictionary of non-zero entries
    # T[a, b, c] = 1 if (a, b, c) corresponds to (i*m+j, j*p+k, k*n+i)
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1

    def check_solution(U, V, W, r):
        # U: nm x r, V: mp x r, W: pn x r
        # Check all (a, b, c)
        # To be fast, we can compute the full tensor and compare
        # But full tensor is 24*12*8 = 2304 entries.
        # We can iterate over all a, b, c and check.
        # Or, we can use matrix multiplication tricks?
        # Let's just iterate.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = 0
                    for r_idx in range(r):
                        val += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
                    target = T.get((a, b, c), 0)
                    if val != target:
                        return False
        return True

    # Try to find a solution with rank < schoolbook_rank
    # We'll try ranks from schoolbook_rank - 1 down to 1
    # Use random search with small integer values
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    for r in range(schoolbook_rank - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(nm, r))
            V = np.random.randint(-2, 3, size=(mp, r))
            W = np.random.randint(-2, 3, size=(pn, r))
            
            # Check if it's a solution
            if check_solution(U, V, W, r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # If not, we could try local search, but for now just try next random init
            # A more effective search would adjust entries to reduce error
            # But exact integer search is hard.
            
    # If no better solution found, return schoolbook
    rank = schoolbook_rank
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
