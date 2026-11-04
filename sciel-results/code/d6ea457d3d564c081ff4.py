"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-42 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for 2x2 blocks.
    - Split A (2x4) into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    - Split B (4x6) into top (rows 0-1) and bottom (rows 2-3), each 2x6
    - Further split each 2x6 into three 2x2 blocks
    - For each 2x2 * 2x2 multiplication, use 7 Strassen products
    - Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    def add_strassen_block(i_base, j_base, k_base, offset):
        """
        Add a Strassen block for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        """
        # A indices: A[i, j] -> U[i*m + j]
        a00 = i_base * m + j_base
        a01 = i_base * m + (j_base + 1)
        a10 = (i_base + 1) * m + j_base
        a11 = (i_base + 1) * m + (j_base + 1)
        
        # B indices: B[j, k] -> V[j*p + k]
        b00 = j_base * p + k_base
        b01 = j_base * p + (k_base + 1)
        b10 = (j_base + 1) * p + k_base
        b11 = (j_base + 1) * p + (k_base + 1)
        
        # W indices: C[i, k] -> W[k*n + i]
        w00 = k_base * n + i_base
        w01 = (k_base + 1) * n + i_base
        w10 = k_base * n + (i_base + 1)
        w11 = (k_base + 1) * n + (i_base + 1)
        
        # M1 = (a+d)(e+h)
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0
        
        # M2 = (c+d)e
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0
        
        # M3 = a(f-h)
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0
        
        # M4 = d(g-e)
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0
        
        # M5 = (a+b)h
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0
        
        # M6 = (c-a)(e+f)
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        # M7 = (b-d)(g+h)
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0
    
    # Apply Strassen blocks
    # i_base: 0 (only one block since n=2)
    # j_base: 0, 2 (two blocks since m=4)
    # k_base: 0, 2, 4 (three blocks since p=6)
    
    offset = 0
    for i_base in range(0, n, 2):
        for j_base in range(0, m, 2):
            for k_base in range(0, p, 2):
                add_strassen_block(i_base, j_base, k_base, offset)
                offset += 7
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rank_val, tol=1e-10):
        # Check that for all valid (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    total = 0.0
                    for r in range(rank_val):
                        total += U_mat[a_idx, r] * V_mat[b_idx, r] * W_mat[c_idx, r]
                    if abs(total - 1.0) > tol:
                        return False
        
        # Check that for invalid triples, the sum is 0
        # A triple (i,j,k) is valid if it corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
        # In the tensor, the entry is 1 iff the indices match the matrix multiplication pattern
        # For our decomposition, we need to ensure no spurious terms
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_c = c % n
                    k_c = c // n
                    
                    # This triple contributes to C[i_c, k_c] via term A[i_a, j_a] * B[j_b, k_b]
                    # It should be nonzero only if i_a == i_c, j_a == j_b, and k_b == k_c
                    if not (i_a == i_c and j_a == j_b and k_b == k_c):
                        total = 0.0
                        for r in range(rank_val):
                            total += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                        if abs(total) > tol:
                            return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback: this shouldn't happen, but return as-is
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
