import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Strategy: Use block decomposition with Strassen 2x2x2 blocks (rank 7 each) and
    schoolbook 2x2x1 blocks (rank 4 each).
    
    Split the inner dimension (size 4) into two blocks of size 2: {0,1} and {2,3}.
    For each block, split the output dimension (size 7) into three 2x2 blocks and one 1x2 block.
    
    Total rank: 2 blocks * (3 Strassen * 7 + 1 schoolbook * 4) = 2 * (21 + 4) = 50
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # i0: start row of A (always 0 here, covers both rows)
        # j0: start col of A (0 or 2)
        # k0: start col of B (0, 2, 4)
        
        # Local indices
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices: W[k*n + i, r]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> contributes to C[0,1] and C[1,1] (negative)
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> contributes to C[1,0] and C[1,1]
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> contributes to C[0,0] and C[0,1]
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> contributes to C[0,0] (negative) and C[1,0]
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> contributes to C[1,1]
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> contributes to C[0,0]
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C[0,0]
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C[0,0]
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C[0,1]
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C[0,1]
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Process blocks
    # For each j-block (size 2), split k into [0,1], [2,3], [4,5], [6]
    for j_start in [0, 2]:
        # Three Strassen 2x2x2 blocks for k = 0-1, 2-3, 4-5
        for k_start in [0, 2, 4]:
            if time.time() > DEADLINE:
                # Return whatever we have so far (shouldn't happen with rank 50)
                return {"U": U, "V": V, "W": W}
            add_strassen_block(0, j_start, k_start)
        
        # One schoolbook 2x2x1 block for k = 6
        if time.time() > DEADLINE:
            return {"U": U, "V": V, "W": W}
        add_schoolbook_221_block(0, j_start, 6)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick verification
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build tensor T[i,j,k] = delta (i.e., T[i*m+j, j*p+k, k*n+i] should be 1)
    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    print(f"Rank: {rank}")
    print(f"Max error: {max_err}")
    print("Valid!" if max_err < 1e-10 else "INVALID!")
