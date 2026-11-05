"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses a structured decomposition:
    - Split m=4 into 2+2.
    - Split p=5 into 2+3.
    - This yields four sub-problems:
      1. 2x2x2 (Strassen, rank 7)
      2. 2x2x3 (Schoolbook, rank 12)
      3. 2x2x2 (Strassen, rank 7)
      4. 2x2x3 (Schoolbook, rank 12)
    Total rank = 38.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Total rank
    rank_total = 38
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col_offset = 0
    
    def add_strassen_2x2x2(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        """
        Add a Strassen decomposition for a 2x2x2 sub-problem.
        u_offset_i, u_offset_j: offsets for U indices (i, j)
        v_offset_j, v_offset_k: offsets for V indices (j, k)
        w_offset_k, w_offset_i: offsets for W indices (k, i)
        
        Note: U index is i*m + j. Here m=4.
        V index is j*p + k. Here p=5.
        W index is k*n + i. Here n=2.
        """
        nonlocal col_offset
        
        # Strassen steps
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c-b)(e+f)
        # M7 = (a-d)(g+h)
        
        # We define the linear forms for A (U) and B (V) and C (W)
        # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B indices: (0,0)=e, (0,1)=f, (1,0)=h, (1,1)=g
        
        # Step 1: M1
        # U: a+d -> U[0,0]=1, U[1,1]=1 (local)
        # V: e+h -> V[0,0]=1, V[1,0]=1 (local)
        # W: C11+=M1, C22+=M1 -> W[0,0]=1, W[1,1]=1 (local)
        
        # We need to map local (li, lj) to global (gi, gj)
        
        def map_u(li, lj):
            return (u_offset_i + li) * m + (u_offset_j + lj)
            
        def map_v(lj, lk):
            return (v_offset_j + lj) * p + (v_offset_k + lk)
            
        def map_w(lk, li):
            return (w_offset_k + lk) * n + (w_offset_i + li)
            
        # M1
        U[map_u(0,0), col_offset] = 1
        U[map_u(1,1), col_offset] = 1
        V[map_v(0,0), col_offset] = 1
        V[map_v(1,0), col_offset] = 1
        W[map_w(0,0), col_offset] = 1
        W[map_w(1,1), col_offset] = 1
        col_offset += 1
        
        # M2
        # U: c+d -> U[1,0]=1, U[1,1]=1
        # V: e -> V[0,0]=1
        # W: C21+=M2, C22-=M2 -> W[0,1]=1, W[1,1]=-1
        U[map_u(1,0), col_offset] = 1
        U[map_u(1,1), col_offset] = 1
        V[map_v(0,0), col_offset] = 1
        W[map_w(0,1), col_offset] = 1
        W[map_w(1,1), col_offset] = -1
        col_offset += 1
        
        # M3
        # U: a -> U[0,0]=1
        # V: f-h -> V[0,1]=1, V[1,1]=-1
        # W: C12+=M3, C22+=M3 -> W[1,0]=1, W[1,1]=1
        U[map_u(0,0), col_offset] = 1
        V[map_v(0,1), col_offset] = 1
        V[map_v(1,1), col_offset] = -1
        W[map_w(1,0), col_offset] = 1
        W[map_w(1,1), col_offset] = 1
        col_offset += 1
        
        # M4
        # U: d -> U[1,1]=1
        # V: g-f -> V[1,1]=1, V[0,1]=-1
        # W: C11+=M4, C22+=M4 -> W[0,0]=1, W[1,1]=1
        U[map_u(1,1), col_offset] = 1
        V[map_v(1,1), col_offset] = 1
        V[map_v(0,1), col_offset] = -1
        W[map_w(0,0), col_offset] = 1
        W[map_w(1,1), col_offset] = 1
        col_offset += 1
        
        # M5
        # U: a+b -> U[0,0]=1, U[0,1]=1
        # V: h -> V[1,0]=1
        # W: C11-=M5, C12+=M5 -> W[0,0]=-1, W[1,0]=1
        U[map_u(0,0), col_offset] = 1
        U[map_u(0,1), col_offset] = 1
        V[map_v(1,0), col_offset] = 1
        W[map_w(0,0), col_offset] = -1
        W[map_w(1,0), col_offset] = 1
        col_offset += 1
        
        # M6
        # U: c-b -> U[1,0]=1, U[0,1]=-1
        # V: e+f -> V[0,0]=1, V[0,1]=1
        # W: C21+=M6, C22+=M6 -> W[0,1]=1, W[1,1]=1
        U[map_u(1,0), col_offset] = 1
        U[map_u(0,1), col_offset] = -1
        V[map_v(0,0), col_offset] = 1
        V[map_v(0,1), col_offset] = 1
        W[map_w(0,1), col_offset] = 1
        W[map_w(1,1), col_offset] = 1
        col_offset += 1
        
        # M7
        # U: a-d -> U[0,0]=1, U[1,1]=-1
        # V: g+h -> V[1,1]=1, V[1,0]=1
        # W: C11+=M7, C22-=M7 -> W[0,0]=1, W[1,1]=-1
        U[map_u(0,0), col_offset] = 1
        U[map_u(1,1), col_offset] = -1
        V[map_v(1,1), col_offset] = 1
        V[map_v(1,0), col_offset] = 1
        W[map_w(0,0), col_offset] = 1
        W[map_w(1,1), col_offset] = -1
        col_offset += 1

    def add_schoolbook_2x2x3(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        """
        Add a Schoolbook decomposition for a 2x2x3 sub-problem.
        """
        nonlocal col_offset
        
        def map_u(li, lj):
            return (u_offset_i + li) * m + (u_offset_j + lj)
            
        def map_v(lj, lk):
            return (v_offset_j + lj) * p + (v_offset_k + lk)
            
        def map_w(lk, li):
            return (w_offset_k + lk) * n + (w_offset_i + li)
            
        # 12 terms: A[li, lj] * B[lj, lk] contributes to C[li, lk]
        for li in range(2):
            for lj in range(2):
                for lk in range(3):
                    U[map_u(li, lj), col_offset] = 1
                    V[map_v(lj, lk), col_offset] = 1
                    W[map_w(lk, li), col_offset] = 1
                    col_offset += 1

    # Sub-problem 1: A1 (cols 0-1) * B1,1 (rows 0-1, cols 0-1)
    # u_offset_i=0, u_offset_j=0
    # v_offset_j=0, v_offset_k=0
    # w_offset_k=0, w_offset_i=0
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # Sub-problem 2: A1 (cols 0-1) * B1,2 (rows 0-1, cols 2-4)
    # u_offset_i=0, u_offset_j=0
    # v_offset_j=0, v_offset_k=2
    # w_offset_k=2, w_offset_i=0
    add_schoolbook_2x2x3(0, 0, 0, 2, 2, 0)
    
    # Sub-problem 3: A2 (cols 2-3) * B2,1 (rows 2-3, cols 0-1)
    # u_offset_i=0, u_offset_j=2
    # v_offset_j=2, v_offset_k=0
    # w_offset_k=0, w_offset_i=0
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    
    # Sub-problem 4: A2 (cols 2-3) * B2,2 (rows 2-3, cols 2-4)
    # u_offset_i=0, u_offset_j=2
    # v_offset_j=2, v_offset_k=2
    # w_offset_k=2, w_offset_i=0
    add_schoolbook_2x2x3(0, 2, 2, 2, 2, 0)
    
    # Verify
    if col_offset != rank_total:
        # This should not happen
        pass
        
    # Check if entries are integers
    if not np.allclose(U, np.round(U)) or not np.allclose(V, np.round(V)) or not np.allclose(W, np.round(W)):
        # Fallback to schoolbook if something went wrong
        U_school = np.zeros((n*m, n*m*p))
        V_school = np.zeros((m*p, n*m*p))
        W_school = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
