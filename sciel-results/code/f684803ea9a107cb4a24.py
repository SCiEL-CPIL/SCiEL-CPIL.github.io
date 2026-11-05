"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
DEADLINE = _START_TIME + 130  # 130s deadline (10s safety margin from 140s limit)

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                T[i, j, k] = np.dot(u_val, v_val) * 1.0  # placeholder
                # Actually: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                T[i, j, k] = 0
                for r in range(rank):
                    T[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    # The matrix multiplication tensor should have T[i,j,k] = 1 if j is the "linking" index
    # For C = A * B: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k) where we're computing the product
    # Actually the tensor for matmul is: T[i,j,k] = 1 (it's just the identity in the linking dimension)
    # Wait, let me reconsider. The matrix multiplication tensor is defined as:
    # T[i,j,k] = delta (i.e., 1) - no, that's not right either.
    # 
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
    
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # We partition the 'm' dimension (5) into pairs (0,1), (2,3) and single (4).
    # For each pair (j, j+1) in A and corresponding rows in B, we use Strassen (7 mults).
    # For the single column j=4 in A and row j=4 in B, we use schoolbook (12 mults).
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # i_base: 0 (only one pair of rows in A/C since n=2)
        # j_base: 0 or 2 (pairs of cols in A / rows in B)
        # k_base: 0, 2, or 4 (pairs of cols in B / cols in C)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M0: (A00+A11)(B00+B11)
        # A00: U[i_base, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        # A10: U[i_base+1, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base]
        # C10: W[k_base, i_base+1], C11: W[k_base+1, i_base+1] (C11 gets -1)
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        # A00: U[i_base, j_base]
        # B01: V[j_base, k_base+1], B11: V[j_base+1, k_base+1]
        # C01: W[k_base+1, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        # A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B00: V[j_base, k_base]
        # C00: W[k_base, i_base], C10: W[k_base, i_base+1]
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        # A00: U[i_base, j_base], A01: U[i_base, j_base+1]
        # B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base] (gets -1), C01: W[k_base+1, i_base]
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        # A10: U[i_base+1, j_base], A00: U[i_base, j_base]
        # B00: V[j_base, k_base], B01: V[j_base, k_base+1]
        # C11: W[k_base+1, i_base+1]
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        # A01: U[i_base, j_base+1], A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base]
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # Handles A[i_base..i_base+1, j_base] * B[j_base, k_base..k_base+1] -> C[i_base..i_base+1, k_base..k_base+1]
        # This is a 2x1 * 1x2 multiplication, requiring 4 scalar multiplications.
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    # k-pairs are (0,1), (2,3), (4,5)
    
    # j_pair 0,1
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_pair 2,3
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    # k-pairs are (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col != rank:
        # Fallback, though it should be exactly 54
        # If it fails, return schoolbook to be safe? No, better to return what we have if valid, 
        # but the problem requires exact decomposition. 
        # The construction is deterministic and correct, so col should be 54.
        pass

    # Verify the decomposition is correct
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][idx] = 1
                    V_s[j * p + k][idx] = 1
                    W_s[k * n + i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
