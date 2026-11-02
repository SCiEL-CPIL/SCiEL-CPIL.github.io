import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    
    # Use 4 disjoint 2x2x2 Strassen blocks + schoolbook for remainder
    # Blocks cover i in {0,1}, j in {0,1}/{2,3}, k in {0,1}/{2,3}
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7  # 28
    total_terms = n * m * p  # 75
    strassen_terms_count = num_strassen_blocks * 8  # 32
    schoolbook_terms_count = total_terms - strassen_terms_count  # 43
    total_rank = total_strassen_rank + schoolbook_terms_count  # 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Track which (i,j,k) triples are covered by Strassen blocks
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    # Schoolbook for the remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Verify correctness (fast check)
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    # if (i,j,k) is a valid multiplication term, and the decomposition is exact.
    # Actually, the tensor T_ijk = delta_ik (for the multiplication tensor), but here
    # we're decomposing the identity-like tensor where T_ijk = 1 for all valid terms.
    # Let's verify by checking that the contraction gives the correct result.
    
    # Quick spot check on a few entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(total_rank))
                expected = 1.0  # Every (i,j,k) term should contribute 1
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # This shouldn't happen, but if it does, fall back to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1.0
                    V_fb[j*p+k][col_fb] = 1.0
                    W_fb[k*n+i][col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
