"""
Rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.

The multiplication tensor condition:
  For all i in [n], j in [m], k in [p]:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

Strategy:
1. Start with schoolbook (rank 60) as guaranteed valid baseline.
2. Search for lower ranks using random sparse initialization + exact verification.
3. Use structured patterns to bias towards valid solutions.
4. Return the lowest-rank valid decomposition found within time budget.
"""

import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]


def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s
    
    # Schoolbook baseline (rank 60)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower-rank decompositions
    # Search ranks from 59 down to a reasonable minimum
    np.random.seed(42)
    random.seed(42)
    
    # Try various rank values
    for r in range(rank_school - 1, 20, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Strategy: sparse random initialization with values in {-1, 0, 1}
            # Each entry gets a nonzero value with some probability
            density = 0.3 + 0.3 * (rank_school - r) / (rank_school - 20)
            
            U_r = np.zeros((n * m, r), dtype=float)
            V_r = np.zeros((m * p, r), dtype=float)
            W_r = np.zeros((p * n, r), dtype=float)
            
            for idx in range(n * m):
                for col_idx in range(r):
                    if random.random() < density:
                        U_r[idx, col_idx] = random.choice([-1, 1])
            
            for idx in range(m * p):
                for col_idx in range(r):
                    if random.random() < density:
                        V_r[idx, col_idx] = random.choice([-1, 1])
            
            for idx in range(p * n):
                for col_idx in range(r):
                    if random.random() < density:
                        W_r[idx, col_idx] = random.choice([-1, 1])
            
            # Verify
            if verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break  # Found valid decomposition for this rank, move to next lower rank
        
        if best_rank == r:
            # Found a valid decomposition at rank r, try to go lower
            continue
    
    # Also try some structured approaches for specific ranks
    # Try rank 30: split into two 3x2x5 blocks
    if best_rank > 30 and time.time() < deadline - 30:
        # For each half of j (j=0,1 and j=2,3), try to find rank-15 for 3x2x5
        # Actually, let's try a different structured approach
        
        # Try: for each (i,k), we need sum over j and r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # If we set W[k*n+i, r] = 1 for all k,i,r and U[i*m+j, r] = 1, V[j*p+k, r] = 1,
        # then sum = r, so we'd need r=1, but that only works if all terms are 1.
        # That's not generally possible.
        
        # Try a cyclic pattern approach
        for r in [30, 25, 20]:
            if time.time() > deadline - 30:
                break
            for attempt in range(30):
                if time.time() > deadline - 30:
                    break
                
                U_r = np.zeros((n * m, r), dtype=float)
                V_r = np.zeros((m * p, r), dtype=float)
                W_r = np.zeros((p * n, r), dtype=float)
                
                # Structured: use modular arithmetic patterns
                for i in range(n):
                    for j in range(m):
                        for r_idx in range(r):
                            if (i * m + j + r_idx) % max(1, r // 2) == 0:
                                U_r[i * m + j, r_idx] = 1
                
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            if (j * p + k + r_idx) % max(1, r // 2) == 0:
                                V_r[j * p + k, r_idx] = 1
                
                for k in range(p):
                    for i in range(n):
                        for r_idx in range(r):
                            if (k * n + i + r_idx) % max(1, r // 2) == 0:
                                W_r[k * n + i, r_idx] = 1
                
                if verify_decomposition(U_r, V_r, W_r, n, m, p):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
    
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    # Verify
    n, m, p = DIMENSIONS
    valid = verify_decomposition(U, V, W, n, m, p)
    print(f"Valid: {valid}")
