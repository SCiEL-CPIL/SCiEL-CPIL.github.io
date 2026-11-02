import time
import sys

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)  
    # W shape: (p*n, rank) = (10, rank)
    
    # We'll build the decomposition using block-Strassen
    # Strategy:
    # Split B's columns: first 2 (B1) and last 3 (B2)
    # Split A's middle dim: first 2 (A_top) and last 2 (A_bot)
    #
    # C[:, 0:2] = A_top·B_top1 + A_bot·B_bot1  (each 2x2*2x2, Strassen rank 7)
    # C[:, 2:5] = A_top·B_top2 + A_bot·B_bot2  (each 2x2*2x3)
    #
    # For 2x2 * 2x3: split into 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = rank 11
    #
    # Total: 2*7 + 2*(7+4) = 14 + 22 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j  # A[i,j] -> row i*4+j in U
    
    def v_idx(j, k):
        return j * p + k  # B[j,k] -> row j*5+k in V
    
    def w_idx(k, i):
        return k * n + i  # C[i,k] -> row k*2+i in W (trace convention)
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (idx, coeff), etc."""
        nonlocal col
        for idx, coeff in u_entries:
            U[idx][col] += coeff
        for idx, coeff in v_entries:
            V[idx][col] += coeff
        for idx, coeff in w_entries:
            W[idx][col] += coeff
        col += 1
    
    def strassen_2x2x2(i0, j0, k0):
        """
        Add 7 Strassen terms for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        
        # m1 = (x11+x22)*(y11+y22)
        # z11 += m1, z22 += m1
        add_term(
            [(u_idx(i0, j0), 1), (u_idx(i0+1, j0+1), 1)],
            [(v_idx(j0, k0), 1), (v_idx(j0+1, k0+1), 1)],
            [(w_idx(k0, i0), 1), (w_idx(k0+1, i0+1), 1)]
        )
        
        # m2 = (x21+x22)*y11
        # z21 += m2, z22 -= m2
        add_term(
            [(u_idx(i0+1, j0), 1), (u_idx(i0+1, j0+1), 1)],
            [(v_idx(j0, k0), 1)],
            [(w_idx(k0, i0+1), 1), (w_idx(k0+1, i0+1), -1)]
        )
        
        # m3 = x11*(y12-y22)
        # z12 += m3, z22 += m3
        add_term(
            [(u_idx(i0, j0), 1)],
            [(v_idx(j0, k0+1), 1), (v_idx(j0+1, k0+1), -1)],
            [(w_idx(k0+1, i0), 1), (w_idx(k0+1, i0+1), 1)]
        )
        
        # m4 = x22*(y21-y11)
        # z11 += m4, z21 += m4
        add_term(
            [(u_idx(i0+1, j0+1), 1)],
            [(v_idx(j0+1, k0), 1), (v_idx(j0, k0), -1)],
            [(w_idx(k0, i0), 1), (w_idx(k0, i0+1), 1)]
        )
        
        # m5 = (x11+x12)*y22
        # z11 -= m5, z12 += m5
        add_term(
            [(u_idx(i0, j0), 1), (u_idx(i0, j0+1), 1)],
            [(v_idx(j0+1, k0+1), 1)],
            [(w_idx(k0, i0), -1), (w_idx(k0+1, i0), 1)]
        )
        
        # m6 = (x21-x11)*(y11+y12)
        # z22 += m6
        add_term(
            [(u_idx(i0+1, j0), 1), (u_idx(i0, j0), -1)],
            [(v_idx(j0, k0), 1), (v_idx(j0, k0+1), 1)],
            [(w_idx(k0+1, i0+1), 1)]
        )
        
        # m7 = (x12-x22)*(y21+y22)
        # z11 += m7
        add_term(
            [(u_idx(i0, j0+1), 1), (u_idx(i0+1, j0+1), -1)],
            [(v_idx(j0+1, k0), 1), (v_idx(j0+1, k0+1), 1)],
            [(w_idx(k0, i0), 1)]
        )
    
    def schoolbook_2x2x1(i0, j0, k0):
        """
        Add 4 schoolbook terms for C[i0:i0+2, k0] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0]
        """
        nonlocal col
        # C[i0, k0] += A[i0, j0] * B[j0, k0]
        add_term(
            [(u_idx(i0, j0), 1)],
            [(v_idx(j0, k0), 1)],
            [(w_idx(k0, i0), 1)]
        )
        # C[i0, k0] += A[i0, j0+1] * B[j0+1, k0]
        add_term(
            [(u_idx(i0, j0+1), 1)],
            [(v_idx(j0+1, k0), 1)],
            [(w_idx(k0, i0), 1)]
        )
        # C[i0+1, k0] += A[i0+1, j0] * B[j0, k0]
        add_term(
            [(u_idx(i0+1, j0), 1)],
            [(v_idx(j0, k0), 1)],
            [(w_idx(k0, i0+1), 1)]
        )
        # C[i0+1, k0] += A[i0+1, j0+1] * B[j0+1, k0]
        add_term(
            [(u_idx(i0+1, j0+1), 1)],
            [(v_idx(j0+1, k0), 1)],
            [(w_idx(k0, i0+1), 1)]
        )
    
    # Block 1: C[:, 0:2] = A_top * B_top1 + A_bot * B_bot1
    # A_top = A[0:2, 0:2], B_top1 = B[0:2, 0:2], C[:, 0:2]
    strassen_2x2x2(i0=0, j0=0, k0=0)
    # A_bot = A[0:2, 2:4], B_bot1 = B[2:4, 0:2], C[:, 0:2]
    strassen_2x2x2(i0=0, j0=2, k0=0)
    
    # Block 2: C[:, 2:5] = A_top * B_top2 + A_bot * B_bot2
    # For each 2x2 * 2x3:
    #   Split 3 cols into 2+1: first 2 cols use Strassen, last 1 col uses schoolbook
    
    # A_top * B_top2: A[0:2, 0:2] * B[0:2, 2:5]
    # Strassen for B[0:2, 2:4] (2x2 part)
    strassen_2x2x2(i0=0, j0=0, k0=2)
    # Schoolbook for B[0:2, 4] (2x1 part)
    schoolbook_2x2x1(i0=0, j0=0, k0=4)
    
    # A_bot * B_bot2: A[0:2, 2:4] * B[2:4, 2:5]
    # Strassen for B[2:4, 2:4] (2x2 part)
    strassen_2x2x2(i0=0, j0=2, k0=2)
    # Schoolbook for B[2:4, 4] (2x1 part)
    schoolbook_2x2x1(i0=0, j0=2, k0=4)
    
    if time.time() - start_time > 100:
        pass
    
    return {"U": U, "V": V, "W": W}
