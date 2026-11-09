import numpy as np
import random
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    # Initialize with schoolbook
    U_seed = np.zeros((n*m, rank_seed))
    V_seed = np.zeros((m*p, rank_seed))
    W_seed = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i*m + j, col] = 1
                V_seed[j*p + k, col] = 1
                W_seed[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_seed
    best_U = U_seed
    best_V = V_seed
    best_W = W_seed

    # Target tensor T (sparse)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    start_time = time.time()
    deadline = start_time + 140 # 10s buffer for safety
    
    # Precompute indices for target tensor (1s)
    # We need to check:
    # 1. All (i, j, k) where T is 1.
    # 2. All (a, b, c) where T is 0 (complementary set).
    
    # For T=1:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # We can iterate i, j, k.
    
    # For T=0:
    # Total entries = (n*m) * (m*p) * (p*n) = (n*m*p)^2 = 900
    # Non-zero entries = n*m*p = 90
    # Zero entries = 810
    # We can iterate all a, b, c and check if it's a 1-entry.
    # Or generate the set of 1-entries and check membership.
    
    # Let's use a set of tuples for 1-entries.
    one_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                one_entries.add((a, b, c))
    
    # List of all entries
    all_entries = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                all_entries.append((a, b, c))
                
    # Zero entries
    zero_entries = [e for e in all_entries if e not in one_entries]
    
    # Candidate ranks
    # Known lower bound for 3x5x6?
    # Schoolbook is 90.
    # Let's try to find something better.
    # Random search for integer solutions is very unlikely to find anything < 90 quickly.
    # But we must try.
    
    # Heuristic: Try ranks that are factors or known good values
    candidate_ranks = list(range(40, best_rank))
    random.shuffle(candidate_ranks)
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this R
        for attempt in range(20): # Increased attempts
            if time.time() > deadline:
                break
            
            # Random integer matrices with small entries
            # Use {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            W = np.random.randint(-1, 2, size=(p*n, R))
            
            # Efficient check using vectorization?
            # We need to check sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
            
            # This is hard to vectorize directly for all entries at once without creating huge arrays.
            # But the total number of entries is small (900).
            # We can loop over entries, but vectorize the sum over r.
            
            valid = True
            
            # Check 1s
            for a, b, c in one_entries:
                # val = sum(U[a, :] * V[b, :] * W[c, :])
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if val != 1:
                    valid = False
                    break
            
            if valid:
                # Check 0s
                for a, b, c in zero_entries:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if val != 0:
                        valid = False
                        break
            
            if valid:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                # Convert to list of lists
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Return best found (likely seed)
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
