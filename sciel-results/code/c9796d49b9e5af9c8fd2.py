import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    trivial_rank = n * m * p  # 24
    
    # Build the trivial solution as fallback
    U_trivial = np.zeros((nm, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((mp, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((pn, trivial_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
    
    best_rank = trivial_rank
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Helper: check if decomposition is valid (exact)
    def check_valid(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Helper: snap to nearest half-integer and recheck
    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s
    
    # Try to construct a rank-19 decomposition using Strassen-like techniques
    # The 2x3x4 multiplication tensor can be decomposed using rank-19
    # Strategy: use known algorithms for small matrix multiplication
    
    # For 2x3x4, we can use the following construction:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We'll define 19 rank-1 terms
    
    target_rank = 19
    U = np.zeros((nm, target_rank), dtype=np.float64)
    V = np.zeros((mp, target_rank), dtype=np.float64)
    W = np.zeros((pn, target_rank), dtype=np.float64)
    
    # Construction based on the tensor structure:
    # We view the 2x3x4 multiplication as a combination of:
    # - 2x2x2 type interactions (using Strassen-like combinations)
    # - 1x3x4 type interactions (trivial for single row/col)
    
    # Let's use a systematic construction:
    # For each (i,k) pair, we need sum_j A[i,j]*B[j,k]
    # We can share intermediate products across k values for fixed i,j
    
    # Define intermediate products M_r for r in range(target_rank)
    # Each M_r = U_r * V_r (outer product of row vectors from U and V)
    # Then C[i,k] = sum_r W[k*n+i, r] * M_r[j, ...] -- this isn't quite right
    
    # Actually, the tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for the multiplication tensor
    
    # Let me try a different approach: solve for the factors directly
    # using the known rank-19 algorithm structure
    
    # Based on literature, the rank of 2x3x4 is 19
    # Let me try to construct it using the following pattern:
    
    # For j=0: A[i,0]*B[0,k] for i in {0,1}, k in {0,1,2,3}
    # For j=1: A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2,3}
    # For j=2: A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2,3}
    
    # We can group terms to share computation:
    # Define S0 = A[0,0], S1 = A[0,1], S2 = A[0,2]
    # Define S3 = A[1,0], S4 = A[1,1], S5 = A[1,2]
    # Define T0 = B[0,0], T1 = B[0,1], T2 = B[0,2], T3 = B[0,3]
    # Define T4 = B[1,0], T5 = B[1,1], T6 = B[1,2], T7 = B[1,3]
    # Define T8 = B[2,0], T9 = B[2,1], T10 = B[2,2], T11 = B[2,3]
    
    # C[0,0] = S0*T0 + S1*T4 + S2*T8
    # C[0,1] = S0*T1 + S1*T5 + S2*T9
    # C[0,2] = S0*T2 + S1*T6 + S2*T10
    # C[0,3] = S0*T3 + S1*T7 + S2*T11
    # C[1,0] = S3*T0 + S4*T4 + S5*T8
    # C[1,1] = S3*T1 + S4*T5 + S5*T9
    # C[1,2] = S3*T2 + S4*T6 + S5*T10
    # C[1,3] = S3*T3 + S4*T7 + S5*T11
    
    # We can use Strassen-like combinations for the 2x2x2 subproblems
    # For example, for fixed k, we have a 2x3 multiplication which can be done in fewer products
    
    # Let me try a concrete rank-19 construction:
    
    # Terms 0-3: Handle j=0 contributions with sharing across k
    # M0 = S0 * T0 -> contributes to C[0,0]
    # M1 = S0 * T1 -> contributes to C[0,1]
    # M2 = S0 * T2 -> contributes to C[0,2]
    # M3 = S0 * T3 -> contributes to C[0,3]
    U[0, 0] = 1.0; V[0, 0] = 1.0; W[0, 0] = 1.0  # C[0,0] += S0*T0
    U[0, 1] = 1.0; V[1, 1] = 1.0; W[1, 1] = 1.0  # C[0,1] += S0*T1
    U[0, 2] = 1.0; V[2, 2] = 1.0; W[2, 2] = 1.0  # C[0,2] += S0*T2
    U[0, 3] = 1.0; V[3, 3] = 1.0; W[3, 3] = 1.0  # C[0,3] += S0*T3
    
    # Terms 4-7: Handle j=0 contributions for row 1
    U[3, 4] = 1.0; V[0, 4] = 1.0; W[0, 4] = 1.0  # C[1,0] += S3*T0
    U[3, 5] = 1.0; V[1, 5] = 1.0; W[1, 5] = 1.0  # C[1,1] += S3*T1
    U[3, 6] = 1.0; V[2, 6] = 1.0; W[2, 6] = 1.0  # C[1,2] += S3*T2
    U[3, 7] = 1.0; V[3, 7] = 1.0; W[3, 7] = 1.0  # C[1,3] += S3*T3
    
    # Terms 8-11: Handle j=1 contributions for row 0
    U[1, 8] = 1.0; V[4, 8] = 1.0; W[0, 8] = 1.0  # C[0,0] += S1*T4
    U[1, 9] = 1.0; V[5, 9] = 1.0; W[1, 9] = 1.0  # C[0,1] += S1*T5
    U[1, 10] = 1.0; V[6, 10] = 1.0; W[2, 10] = 1.0  # C[0,2] += S1*T6
    U[1, 11] = 1.0; V[7, 11] = 1.0; W[3, 11] = 1.0  # C[0,3] += S1*T7
    
    # Terms 12-15: Handle j=1 contributions for row 1
    U[4, 12] = 1.0; V[4, 12] = 1.0; W[0, 12] = 1.0  # C[1,0] += S4*T4
    U[4, 13] = 1.0; V[5, 13] = 1.0; W[1, 13] = 1.0  # C[1,1] += S4*T5
    U[4, 14] = 1.0; V[6, 14] = 1.0; W[2, 14] = 1.0  # C[1,2] += S4*T6
    U[4, 15] = 1.0; V[7, 15] = 1.0; W[3, 15] = 1.0  # C[1,3] += S4*T7
    
    # Terms 16-19: Handle j=2 contributions - but we only have 4 terms left
    # We need to handle: S2*T8, S2*T9, S2*T10, S2*T11 for C[0,*]
    # and S5*T8, S5*T9, S5*T10, S5*T11 for C[1,*]
    # That's 8 terms but we only have 4 slots. We need to share.
    
    # Let's use Strassen-like combinations for the j=2 part:
    # Define:
    # M16 = (S2 + S5) * (T8 + T10) = S2*T8 + S2*T10 + S5*T8 + S5*T10
    # M17 = (S2 + S5) * (T9 + T11) = S2*T9 + S2*T11 + S5*T9 + S5*T11
    # M18 = S2 * T8 - S5 * T8 = (S2 - S5) * T8
    # M19 = S2 * T10 - S5 * T10 = (S2 - S5) * T10
    
    # This is getting complicated. Let me try a different approach.
    
    # Actually, let me just try random search more aggressively with better initialization
    # and also try to use the structure to guide the search.
    
    # Reset and try random search with more attempts
    U.fill(0.0)
    V.fill(0.0)
    W.fill(0.0)
    
    # Try different rank values from 19 down to 12
    for target_rank in range(19, 11, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Multiple attempts per rank
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with small random values
            U = np.random.randint(-2, 3, size=(nm, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(mp, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(pn, target_rank)) * 0.5
            
            # Quick check
            if check_valid(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
            
            # Try snapping to half-integers
            U_s, V_s, W_s = snap_to_half_integer(U, V, W)
            if check_valid(U_s, V_s, W_s, target_rank):
                best_rank = target_rank
                best_U = U_s
                best_V = V_s
                best_W = W_s
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better solution, use it; otherwise use trivial
    if best_rank < trivial_rank:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U_trivial
        V_out = V_trivial
        W_out = W_trivial
    
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
