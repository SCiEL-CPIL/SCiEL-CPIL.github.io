"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_2x2():
    """
    Return the Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Each of the 7 products M1..M7 is a rank-1 term.
    For X (2x2) * Y (2x2) = Z (2x2):
    X indices: (i, jl) where i in {0,1}, jl in {0,1}
    Y indices: (jh, kl) where jh in {0,1}, kl in {0,1}
    Z indices: (i, kl)
    
    In our global indexing:
    - A[i, j] where j = 2*jh + jl, so A index = i*4 + 2*jh + jl
    - B[j, k] where j = 2*jh + jl, k = 2*kh + kl, so B index = (2*jh+jl)*8 + 2*kh + kl
    - C[i, k] where k = 2*kh + kl, so C index = (2*kh+kl)*2 + i
    """
    # Strassen products for X (2x2) * Y (2x2) = Z (2x2)
    # X = [[a, b], [c, d]] where a=X[0,0], b=X[0,1], c=X[1,0], d=X[1,1]
    # Y = [[e, f], [g, h]] where e=Y[0,0], f=Y[0,1], g=Y[1,0], h=Y[1,1]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # Z[0,0] = M1+M4-M5+M7
    # Z[0,1] = M3+M5
    # Z[1,0] = M2+M4
    # Z[1,1] = M1-M2+M3+M6
    
    # For each Strassen product, we need to express it as a product of
    # linear forms in A, B, and C.
    # Each product M_s = (sum of A terms) * (sum of B terms) contributes to
    # specific C entries.
    
    # Let me define for a given (jh, kh) block:
    # A indices: i in {0,1}, jl in {0,1} -> A[i, 2*jh + jl]
    # B indices: jh in {0,1}, kl in {0,1} -> B[2*jh + jl, 2*kh + kl]
    # Wait, I need to be more careful.
    
    # Actually for the sub-problem at (jh, kh):
    # We're computing contribution to C[i, 2*kh + kl] from A[i, 2*jh + jl] * B[2*jh + jl, 2*kh + kl]
    # for i, jl, kl in {0,1}
    
    # So the 2x2 X matrix has X[i, jl] = A[i, 2*jh + jl]
    # The 2x2 Y matrix has Y[jl, kl] = B[2*jh + jl, 2*kh + kl]
    # The 2x2 Z matrix has Z[i, kl] = C[i, 2*kh + kl] (contribution from this jh block)
    
    # Strassen terms:
    # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
    #     = (A[i,2*jh+0] + A[i,2*jh+1]) -- wait, this doesn't work directly
    #     because X[0,0] = A[0, 2*jh] and X[1,1] = A[1, 2*jh+1]
    #     These are different A indices, so we can't factor it as a single product.
    
    # Hmm, I think I need to reconsider. Strassen works on the full 2x2 matrices,
    # so each M_s is a product of a linear combination of X entries and a linear
    # combination of Y entries. In the tensor decomposition, each M_s becomes
    # a rank-1 term where:
    # - U part: coefficients for A entries involved
    # - V part: coefficients for B entries involved  
    # - W part: coefficients for C entries that receive this contribution
    
    # Let me define the 7 Strassen products more carefully.
    # For X (2x2) * Y (2x2):
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # In the tensor decomposition, each M_s is a rank-1 term:
    # M_s = (linear form in X) * (linear form in Y)
    # This contributes to Z entries with certain coefficients.
    
    # For the global tensor, for a fixed (jh, kh):
    # x00 = A[0, 2*jh+0], x01 = A[0, 2*jh+1]
    # x10 = A[1, 2*jh+0], x11 = A[1, 2*jh+1]
    # y00 = B[2*jh+0, 2*kh+0], y01 = B[2*jh+0, 2*kh+1]
    # y10 = B[2*jh+1, 2*kh+0], y11 = B[2*jh+1, 2*kh+1]
    # z00 = C[0, 2*kh+0], z01 = C[0, 2*kh+1]
    # z10 = C[1, 2*kh+0], z11 = C[1, 2*kh+1]
    
    # Each M_s contributes to specific z entries.
    
    # Let me build the full decomposition.
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    col = 0
    for jh in range(2):
        for kh in range(4):
            # For this (jh, kh) block, we have 7 Strassen terms
            # x00 = A[0, 2*jh], x01 = A[0, 2*jh+1]
            # x10 = A[1, 2*jh], x11 = A[1, 2*jh+1]
            # y00 = B[2*jh, 2*kh], y01 = B[2*jh, 2*kh+1]
            # y10 = B[2*jh+1, 2*kh], y11 = B[2*jh+1, 2*kh+1]
            
            # M1 = (x00 + x11) * (y00 + y11)
            # U: A[0, 2*jh] += 1, A[1, 2*jh+1] += 1
            # V: B[2*jh, 2*kh] += 1, B[2*jh+1, 2*kh+1] += 1
            # W: C[0, 2*kh] += 1, C[1, 2*kh+1] += 1 (from Z00 and Z11)
            # Wait, Z00 = M1 + ... and Z11 = M1 + ...
            # So M1 contributes +1 to Z00 and +1 to Z11
            # Z00 = C[0, 2*kh], Z11 = C[1, 2*kh+1]
            U[0 * m + (2*jh + 0)][col] += 1   # A[0, 2*jh]
            U[1 * m + (2*jh + 1)][col] += 1   # A[1, 2*jh+1]
            V[(2*jh + 0) * p + (2*kh + 0)][col] += 1  # B[2*jh, 2*kh]
            V[(2*jh + 1) * p + (2*kh + 1)][col] += 1  # B[2*jh+1, 2*kh+1]
            W[(2*kh + 0) * n + 0][col] += 1   # C[0, 2*kh]
            W[(2*kh + 1) * n + 1][col] += 1   # C[1, 2*kh+1]
            col += 1
            
            # M2 = (x10 + x11) * y00
            # U: A[1, 2*jh] += 1, A[1, 2*jh+1] += 1
            # V: B[2*jh, 2*kh] += 1
            # W: Z10 = C[1, 2*kh] += 1
            U[1 * m + (2*jh + 0)][col] += 1   # A[1, 2*jh]
            U[1 * m + (2*jh + 1)][col] += 1   # A[1, 2*jh+1]
            V[(2*jh + 0) * p + (2*kh + 0)][col] += 1  # B[2*jh, 2*kh]
            W[(2*kh + 0) * n + 1][col] += 1   # C[1, 2*kh]
            col += 1
            
            # M3 = x00 * (y01 - y11)
            # U: A[0, 2*jh] += 1
            # V: B[2*jh, 2*kh+1] += 1, B[2*jh+1, 2*kh+1] -= 1
            # W: Z01 = C[0, 2*kh+1] += 1, Z11 = C[1, 2*kh+1] += 1
            U[0 * m + (2*jh + 0)][col] += 1   # A[0, 2*jh]
            V[(2*jh + 0) * p + (2*kh + 1)][col] += 1  # B[2*jh, 2*kh+1]
            V[(2*jh + 1) * p + (2*kh + 1)][col] -= 1  # B[2*jh+1, 2*kh+1]
            W[(2*kh + 1) * n + 0][col] += 1   # C[0, 2*kh+1]
            W[(2*kh + 1) * n + 1][col] += 1   # C[1, 2*kh+1]
            col += 1
            
            # M4 = x11 * (y10 - y00)
            # U: A[1, 2*jh+1] += 1
            # V: B[2*jh+1, 2*kh] += 1, B[2*jh, 2*kh] -= 1
            # W: Z00 = C[0, 2*kh] += 1, Z10 = C[1, 2*kh] += 1
            U[1 * m + (2*jh + 1)][col] += 1   # A[1, 2*jh+1]
            V[(2*jh + 1) * p + (2*kh + 0)][col] += 1  # B[2*jh+1, 2*kh]
            V[(2*jh + 0) * p + (2*kh + 0)][col] -= 1  # B[2*jh, 2*kh]
            W[(2*kh + 0) * n + 0][col] += 1   # C[0, 2*kh]
            W[(2*kh + 0) * n + 1][col] += 1   # C[1, 2*kh]
            col += 1
            
            # M5 = (x00 + x01) * y11
            # U: A[0, 2*jh] += 1, A[0, 2*jh+1] += 1
            # V: B[2*jh+1, 2*kh+1] += 1
            # W: Z01 = C[0, 2*kh+1] += 1, Z00 = C[0, 2*kh] -= 1
            U[0 * m + (2*jh + 0)][col] += 1   # A[0, 2*jh]
            U[0 * m + (2*jh + 1)][col] += 1   # A[0, 2*jh+1]
            V[(2*jh + 1) * p + (2*kh + 1)][col] += 1  # B[2*jh+1, 2*kh+1]
            W[(2*kh + 1) * n + 0][col] += 1   # C[0, 2*kh+1]
            W[(2*kh + 0) * n + 0][col] -= 1   # C[0, 2*kh]
            col += 1
            
            # M6 = (x10 - x00) * (y00 + y01)
            # U: A[1, 2*jh] += 1, A[0, 2*jh] -= 1
            # V: B[2*jh, 2*kh] += 1, B[2*jh, 2*kh+1] += 1
            # W: Z11 = C[1, 2*kh+1] += 1
            U[1 * m + (2*jh + 0)][col] += 1   # A[1, 2*jh]
            U[0 * m + (2*jh + 0)][col] -= 1   # A[0, 2*jh]
            V[(2*jh + 0) * p + (2*kh + 0)][col] += 1  # B[2*jh, 2*kh]
            V[(2*jh + 0) * p + (2*kh + 1)][col] += 1  # B[2*jh, 2*kh+1]
            W[(2*kh + 1) * n + 1][col] += 1   # C[1, 2*kh+1]
            col += 1
            
            # M7 = (x01 - x11) * (y10 + y11)
            # U: A[0, 2*jh+1] += 1, A[1, 2*jh+1] -= 1
            # V: B[2*jh+1, 2*kh] += 1, B[2*jh+1, 2*kh+1] += 1
            # W: Z00 = C[0, 2*kh] += 1
            U[0 * m + (2*jh + 1)][col] += 1   # A[0, 2*jh+1]
            U[1 * m + (2*jh + 1)][col] -= 1   # A[1, 2*jh+1]
            V[(2*jh + 1) * p + (2*kh + 0)][col] += 1  # B[2*jh+1, 2*kh]
            V[(2*jh + 1) * p + (2*kh + 1)][col] += 1  # B[2*jh+1, 2*kh+1]
            W[(2*kh + 0) * n + 0][col] += 1   # C[0, 2*kh]
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at i={i}, j={j}, k={k}: val={val}")
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Use Strassen-based decomposition with rank 56
    result = _strassen_2x2_2x2()
    
    return result
# EVOLVE-BLOCK-END
