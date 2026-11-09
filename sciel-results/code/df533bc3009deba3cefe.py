"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum.
    
    T[i,j,jp,k,kp,ip] = sum_r U[i*m+j,r] * V[jp*p+k,r] * W[kp*n+ip,r]
    must equal delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)   # U3[i,j,r]
    V3 = V.reshape(m, p, R)   # V3[jp,k,r]
    W3 = W.reshape(p, n, R)   # W3[kp,ip,r]
    
    # T[i,j,jp,k,kp,ip] = sum_r U3[i,j,r] * V3[jp,k,r] * W3[kp,ip,r]
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    
    # Build target: delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    # Target shape: (n, m, m, p, p, n)
    # T[i,j,jp,k,kp,ip] should be 1 if i==ip and j==jp and k==kp
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _add_strassen_block(U, V, W, rank_offset, a_row_start, a_col_start, b_row_start, b_col_start, res_col_start, n, m, p):
    """
    Adds a Strassen decomposition for a 2x2 * 2x2 block to the global U, V, W matrices.
    
    A is 2x2 at [a_row_start:a_row_start+2, a_col_start:a_col_start+2]
    B is 2x2 at [b_row_start:b_row_start+2, b_col_start:b_col_start+2]
    Result is 2x2 at [res_col_start:res_col_start+2] (rows are 0..n-1, cols are res_col_start..)
    
    Note: The result of A*B is added to the main result.
    The Strassen intermediates M1..M7 are computed as:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A11-A10)(B11+B01)
    
    Result:
    R00 = M1 + M4 - M5 + M7
    R01 = M3 + M5
    R10 = M2 + M4
    R11 = M1 - M2 + M3 + M6
    """
    # U indices: i*m + j. i in [0,1], j in [0,1] relative to a_row_start, a_col_start
    # V indices: jp*p + k. jp in [0,1], k in [0,1] relative to b_row_start, b_col_start
    # W indices: kp*n + ip. kp in [0,1] (col in result), ip in [0,1] (row in result)
    
    # We need to map the local 2x2 indices to global indices.
    # A is at rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1
    # B is at rows b_row_start..b_row_start+1, cols b_col_start..b_col_start+1
    # Result is at rows 0..1 (since n=2), cols res_col_start..res_col_start+1
    
    # Strassen coefficients for U (A factors)
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A10=1
    # M6: A10=1, A00=-1
    # M7: A11=1, A10=-1
    
    u_coeffs = [
        [(1,0,1), (1,1,1)], # M1: (0,0)=1, (1,1)=1
        [(1,0,1), (1,1,1)], # M2: (1,0)=1, (1,1)=1
        [(0,0,1)],          # M3: (0,0)=1
        [(1,1,1)],          # M4: (1,1)=1
        [(0,0,1), (1,0,1)], # M5: (0,0)=1, (1,0)=1
        [(1,0,1), (0,0,-1)],# M6: (1,0)=1, (0,0)=-1
        [(1,1,1), (1,0,-1)] # M7: (1,1)=1, (1,0)=-1
    ]
    
    # Strassen coefficients for V (B factors)
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B10=1, B11=-1
    # M4: B01=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B10=1
    # M7: B11=1, B01=1
    
    v_coeffs = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(1,0,1), (1,1,-1)],# M3
        [(0,1,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (1,0,1)], # M6
        [(1,1,1), (0,1,1)]  # M7
    ]
    
    # Strassen coefficients for W (Result factors)
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # W index is kp*n + ip.
    # kp is the column index in the 2x2 result block (0 or 1).
    # ip is the row index in the 2x2 result block (0 or 1).
    # Global row in W is kp (since p=6, but wait. W shape is (p*n, rank). Index is kp*n + ip?
    # Task says: W[k*n+i, r]. k is col index of result (0..p-1), i is row index of result (0..n-1).
    # In our local 2x2 result, local col is lc (0,1), local row is lr (0,1).
    # Global col = res_col_start + lc.
    # Global row = lr (since n=2, rows are 0,1).
    
    w_coeffs = [
        # M1
        [(0,0,1), (1,1,1)], # R00 += M1, R11 += M1
        # M2
        [(1,0,1), (1,1,-1)], # R10 += M2, R11 -= M2
        # M3
        [(0,1,1), (1,1,1)],  # R01 += M3, R11 += M3
        # M4
        [(0,0,1), (1,0,1)],  # R00 += M4, R10 += M4
        # M5
        [(0,0,-1), (0,1,1)], # R00 -= M5, R01 += M5
        # M6
        [(1,1,1)],           # R11 += M6
        # M7
        [(0,0,1)]            # R00 += M7
    ]
    
    for m_idx in range(7):
        r = rank_offset + m_idx
        
        # Update U
        for (li, lj, val) in u_coeffs[m_idx]:
            gi = a_row_start + li
            gj = a_col_start + lj
            U[gi * m + gj, r] += val
            
        # Update V
        for (lj, lk, val) in v_coeffs[m_idx]:
            gj = b_row_start + lj
            gk = b_col_start + lk
            V[gj * p + gk, r] += val
            
        # Update W
        for (lk, li, val) in w_coeffs[m_idx]:
            # lk is local col, li is local row
            gk = res_col_start + lk
            gi = li # Global row is same as local row since n=2 and we start at 0
            W[gk * n + gi, r] += val

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS
    
    # Rank 42 solution using 6 Strassen blocks
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # A is 2x4. Split into A1 (cols 0-1) and A2 (cols 2-3).
    # B is 4x6. Split into 6 blocks of 2x2.
    # B rows 0-1 are B_top, rows 2-3 are B_bottom.
    # B cols 0-1, 2-3, 4-5.
    
    # Block 1: A1 * B_top[0:2, 0:2] -> Res[0:2, 0:2]
    # A1 is A[0:2, 0:2]. B_top[0:2, 0:2] is B[0:2, 0:2].
    _add_strassen_block(U, V, W, 0, 
                        a_row_start=0, a_col_start=0, 
                        b_row_start=0, b_col_start=0, 
                        res_col_start=0, n=n, m=m, p=p)
                        
    # Block 2: A2 * B_bottom[0:2, 0:2] -> Res[0:2, 0:2]
    # A2 is A[0:2, 2:4]. B_bottom[0:2, 0:2] is B[2:4, 0:2].
    _add_strassen_block(U, V, W, 7, 
                        a_row_start=0, a_col_start=2, 
                        b_row_start=2, b_col_start=0, 
                        res_col_start=0, n=n, m=m, p=p)
                        
    # Block 3: A1 * B_top[0:2, 2:4] -> Res[0:2, 2:4]
    # A1 is A[0:2, 0:2]. B_top[0:2, 2:4] is B[0:2, 2:4].
    _add_strassen_block(U, V, W, 14, 
                        a_row_start=0, a_col_start=0, 
                        b_row_start=0, b_col_start=2, 
                        res_col_start=2, n=n, m=m, p=p)
                        
    # Block 4: A2 * B_bottom[0:2, 2:4] -> Res[0:2, 2:4]
    # A2 is A[0:2, 2:4]. B_bottom[0:2, 2:4] is B[2:4, 2:4].
    _add_strassen_block(U, V, W, 21, 
                        a_row_start=0, a_col_start=2, 
                        b_row_start=2, b_col_start=2, 
                        res_col_start=2, n=n, m=m, p=p)
                        
    # Block 5: A1 * B_top[0:2, 4:6] -> Res[0:2, 4:6]
    # A1 is A[0:2, 0:2]. B_top[0:2, 4:6] is B[0:2, 4:6].
    _add_strassen_block(U, V, W, 28, 
                        a_row_start=0, a_col_start=0, 
                        b_row_start=0, b_col_start=4, 
                        res_col_start=4, n=n, m=m, p=p)
                        
    # Block 6: A2 * B_bottom[0:2, 4:6] -> Res[0:2, 4:6]
    # A2 is A[0:2, 2:4]. B_bottom[0:2, 4:6] is B[2:4, 4:6].
    _add_strassen_block(U, V, W, 35, 
                        a_row_start=0, a_col_start=2, 
                        b_row_start=2, b_col_start=4, 
                        res_col_start=4, n=n, m=m, p=p)
                        
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=float)
        V_s = np.zeros((m * p, rank_school), dtype=float)
        W_s = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
