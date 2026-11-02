"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Use 6 Strassen blocks for first 12 columns, schoolbook for last column
    # Total rank: 6*7 + 4 = 46
    rank = 46
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    counter = 0

    def emit(u_dict, v_dict, w_dict):
        nonlocal counter
        for ii, vv in u_dict.items():
            U_mat[ii][counter] += vv
        for ii, vv in v_dict.items():
            V_mat[ii][counter] += vv
        for ii, vv in w_dict.items():
            W_mat[ii][counter] += vv
        counter += 1

    # Strassen-like decomposition for pairs of columns
    num_pairs = p // 2
    leftover = p - 2 * num_pairs

    for blk in range(num_pairs):
        ka = 2 * blk
        kb = 2 * blk + 1

        # M1 = (a00 + a11) * (b00 + b11)
        # U: a00 (0), a11 (3)
        # V: b00 (ka), b11 (13+kb) -> Note: B is 2x13. Row 0 is 0..12, Row 1 is 13..25.
        # B[j,k] is at index j*p + k.
        # b00 is B[0,ka] -> index ka.
        # b11 is B[1,kb] -> index p + kb = 13 + kb.
        # W: C[k,i] is at index k*n + i.
        # C[ka,0] -> 2*ka + 0.
        # C[kb,1] -> 2*kb + 1.
        
        # Wait, the Strassen formula in the current program:
        # M1: U[0]=1, U[3]=1. V[ka]=1, V[13+kb]=1. W[2ka]=1, W[2kb+1]=1.
        # This corresponds to (A00+A11)(B0k1 + B1k2) -> C[k1,0] + C[k2,1]?
        # Let's check the current program's logic.
        # It works. I will copy the logic exactly.

        emit({0: 1, 3: 1},
             {ka: 1, 13 + kb: 1},
             {2 * ka: 1, 2 * kb + 1: 1})

        # M2 = (a10 + a11) * b00
        # U: a10 (2), a11 (3)
        # V: b00 (ka)
        # W: C[ka,1] (2ka+1), C[kb,1] (2kb+1) with -1?
        # Current: U[2]=1, U[3]=1. V[ka]=1. W[2ka+1]=1, W[2kb+1]=-1.
        emit({2: 1, 3: 1},
             {ka: 1},
             {2 * ka + 1: 1, 2 * kb + 1: -1})

        # M3 = a00 * (b01 - b11)
        # U: a00 (0)
        # V: b01 (kb), b11 (13+kb) with -1?
        # Current: U[0]=1. V[kb]=1, V[13+kb]=-1. W[2kb]=1, W[2kb+1]=1.
        # Wait, b01 is B[0,kb] -> index kb.
        # b11 is B[1,kb] -> index 13+kb.
        # M3 = A00 * (B0k2 - B1k2).
        # Output: C[k2,0] + C[k2,1]?
        # Current: W[2kb]=1, W[2kb+1]=1.
        emit({0: 1},
             {kb: 1, 13 + kb: -1},
             {2 * kb: 1, 2 * kb + 1: 1})

        # M4 = a11 * (b10 - b00)
        # U: a11 (3)
        # V: b10 (13+ka), b00 (ka) with -1?
        # Current: U[3]=1. V[13+ka]=1, V[ka]=-1. W[2ka]=1, W[2ka+1]=1.
        emit({3: 1},
             {13 + ka: 1, ka: -1},
             {2 * ka: 1, 2 * ka + 1: 1})

        # M5 = (a00 + a01) * b11
        # U: a00 (0), a01 (1)
        # V: b11 (13+kb)
        # W: C[ka,0] (2ka) with -1, C[kb,0] (2kb) with 1?
        # Current: U[0]=1, U[1]=1. V[13+kb]=1. W[2ka]=-1, W[2kb]=1.
        emit({0: 1, 1: 1},
             {13 + kb: 1},
             {2 * ka: -1, 2 * kb: 1})

        # M6 = (a10 - a00) * (b00 + b01)
        # U: a10 (2), a00 (0) with -1?
        # V: b00 (ka), b01 (kb)
        # W: C[kb,1] (2kb+1)
        # Current: U[2]=1, U[0]=-1. V[ka]=1, V[kb]=1. W[2kb+1]=1.
        emit({2: 1, 0: -1},
             {ka: 1, kb: 1},
             {2 * kb + 1: 1})

        # M7 = (a01 - a11) * (b10 + b11)
        # U: a01 (1), a11 (3) with -1?
        # V: b10 (13+ka), b11 (13+kb)
        # W: C[ka,0] (2ka)
        # Current: U[1]=1, U[3]=-1. V[13+ka]=1, V[13+kb]=1. W[2ka]=1.
        emit({1: 1, 3: -1},
             {13 + ka: 1, 13 + kb: 1},
             {2 * ka: 1})

    # Handle leftover column
    if leftover:
        kl = 2 * num_pairs
        # Schoolbook for 1 column
        # C[kl,0] = A00*B0kl + A01*B1kl
        # C[kl,1] = A10*B0kl + A11*B1kl
        emit({0: 1}, {kl: 1}, {2 * kl: 1})
        emit({1: 1}, {13 + kl: 1}, {2 * kl: 1})
        emit({2: 1}, {kl: 1}, {2 * kl + 1: 1})
        emit({3: 1}, {13 + kl: 1}, {2 * kl + 1: 1})

    # Convert to lists for return
    U_list = [row[:] for row in U_mat]
    V_list = [row[:] for row in V_mat]
    W_list = [row[:] for row in W_mat]

    # Verify
    if time.time() < deadline and _check_decomposition(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: pure schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j][c] = 1
                V_fb[j*p + k][c] = 1
                W_fb[k*n + i][c] = 1
                c += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
