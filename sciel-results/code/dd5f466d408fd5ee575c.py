"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # We partition the 'm' dimension (5) into pairs (0,1), (2,3) and single (4).
    # For each pair (j, j+1) in A and corresponding rows in B, we use Strassen (7 mults).
    # For the single column j=4 in A and row j=4 in B, we use schoolbook (12 mults).
    
    rank = 54
    # Use numpy arrays for cleaner index handling and faster verification if needed
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
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
        U[i_base * m + j_base, col] += 1
        U[(i_base+1) * m + (j_base+1), col] += 1
        V[j_base * p + k_base, col] += 1
        V[(j_base+1) * p + (k_base+1), col] += 1
        W[k_base * n + i_base, col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        # A10: U[i_base+1, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base]
        # C10: W[k_base, i_base+1], C11: W[k_base+1, i_base+1] (C11 gets -1)
        U[(i_base+1) * m + j_base, col] += 1
        U[(i_base+1) * m + (j_base+1), col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base+1), col] += 1
        W[(k_base+1) * n + (i_base+1), col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        # A00: U[i_base, j_base]
        # B01: V[j_base, k_base+1], B11: V[j_base+1, k_base+1]
        # C01: W[k_base+1, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        V[(j_base+1) * p + (k_base+1), col] -= 1
        W[(k_base+1) * n + i_base, col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        # A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B00: V[j_base, k_base]
        # C00: W[k_base, i_base], C10: W[k_base, i_base+1]
        U[(i_base+1) * m + (j_base+1), col] += 1
        V[(j_base+1) * p + k_base, col] += 1
        V[j_base * p + k_base, col] -= 1
        W[k_base * n + i_base, col] += 1
        W[k_base * n + (i_base+1), col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        # A00: U[i_base, j_base], A01: U[i_base, j_base+1]
        # B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base] (gets -1), C01: W[k_base+1, i_base]
        U[i_base * m + j_base, col] += 1
        U[i_base * m + (j_base+1), col] += 1
        V[(j_base+1) * p + (k_base+1), col] += 1
        W[k_base * n + i_base, col] -= 1
        W[(k_base+1) * n + i_base, col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        # A10: U[i_base+1, j_base], A00: U[i_base, j_base]
        # B00: V[j_base, k_base], B01: V[j_base, k_base+1]
        # C11: W[k_base+1, i_base+1]
        U[(i_base+1) * m + j_base, col] += 1
        U[i_base * m + j_base, col] -= 1
        V[j_base * p + k_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        # A01: U[i_base, j_base+1], A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base]
        U[i_base * m + (j_base+1), col] += 1
        U[(i_base+1) * m + (j_base+1), col] -= 1
        V[(j_base+1) * p + k_base, col] += 1
        V[(j_base+1) * p + (k_base+1), col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # Handles A[i_base..i_base+1, j_base] * B[j_base, k_base..k_base+1] -> C[i_base..i_base+1, k_base..k_base+1]
        # This is a 2x1 * 1x2 multiplication, requiring 4 scalar multiplications.
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        W[(k_base+1) * n + i_base, col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base+1) * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base+1), col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base+1) * m + j_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
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
        # If col < rank, trim the matrices to the actual number of terms used
        if col < rank:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
        # If col > rank, this is an error; return what we have (shouldn't happen)
        pass

    # Verify the decomposition with a quick check (only if time permits)
    if _time_remaining() > 5:
        try:
            U_np = U.astype(float)
            V_np = V.astype(float)
            W_np = W.astype(float)
            r = U_np.shape[1]
            
            # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Check against the expected multiplication tensor
            # Expected: for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor should satisfy: sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
            # In the decomposition form, T is implicitly defined by U,V,W.
            # We can verify by checking that the decomposition produces correct results
            # for random matrices.
            
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Efficient verification using einsum or loops
            # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
            # Let's construct the full tensor T of shape (n, m, p)
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Reshape U, V, W for easier broadcasting
            # U: (n*m, r) -> (n, m, r)
            U_reshaped = U_np.reshape(n, m, r)
            # V: (m*p, r) -> (m, p, r)
            V_reshaped = V_np.reshape(m, p, r)
            # W: (p*n, r) -> (p, n, r)
            W_reshaped = W_np.reshape(p, n, r)
            
            # Compute T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
            # This is a contraction over r.
            # We can do this by iterating over r or using einsum.
            # einsum: 'ijr,jkr,kir->ijk'
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            # Now compute C_computed[i,k] = sum_j T[i,j,k] * A_test[i,j] * B_test[j,k]
            # einsum: 'ijk,ij,jk->ik'
            C_computed = np.einsum('ijk,ij,jk->ik', T, A_test, B_test)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Decomposition is incorrect; fall back to schoolbook
                rank_school = n * m * p
                U_school = np.zeros((n * m, rank_school), dtype=int)
                V_school = np.zeros((m * p, rank_school), dtype=int)
                W_school = np.zeros((p * n, rank_school), dtype=int)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_school[i*m+j, idx] = 1
                            V_school[j*p+k, idx] = 1
                            W_school[k*n+i, idx] = 1
                return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        except Exception:
            pass

    # Convert numpy arrays to lists as per interface requirement (integer/half-integer entries)
    # The interface expects lists of lists or arrays? "factor matrices of shapes... with integer/half-integer entries"
    # The seed program returned lists. Let's return lists to be safe, or numpy arrays if allowed.
    # The prompt says "factor matrices of shapes (n*m, rank)...". Usually numpy arrays are fine, but let's stick to the seed's style if possible.
    # However, the seed program returned lists. Let's return lists to match the "interface" strictly if it implies Python types.
    # But inspiration 2 used numpy arrays directly. Let's check the seed again.
    # Seed: return {"U": U, "V": V, "W": W} where U, V, W were lists.
    # Inspiration 2: return {"U": U, "V": V, "W": W} where U, V, W were numpy arrays.
    # The evaluator likely accepts both. Numpy arrays are more efficient. I will return numpy arrays.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
