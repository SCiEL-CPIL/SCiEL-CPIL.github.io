import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen 2x2x2 factors (rank 7)
    # U[i,j,r], V[i,j,r], W[i,j,r]
    # Indices for 2x2: 0, 1
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)

    # r=0: (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1

    # r=1: (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1

    # r=2: A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1

    # r=3: A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1

    # r=4: (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1

    # r=5: (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1

    # r=6: (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Vectorized construction of 4x4x4 factors from 2x2x2
    # The rank index R = 7*s + r.
    # We want U4[i, j, R] where i=2*i1+i0, j=2*j1+j0.
    # U4 is the tensor product (Kronecker) of U2 with itself.
    
    # We can create the 4x4x49 array by iterating over the 49 ranks, 
    # but we can also use broadcasting. 
    # Let's reshape U2 to (2, 2, 7). 
    # We want to map (i1, i0, j1, j0, s, r) -> (i, j, R).
    
    # A simpler vectorized way: 
    # Create indices for the high-level (2x2) and low-level (2x2) parts.
    # i1, i0 in 0..1. i = 2*i1 + i0.
    
    # For each rank component r_low in 0..6 and s_high in 0..6:
    # R = 7*s_high + r_low
    # U4[:, :, R] = kron(U2[:,:,s_high], U2[:,:,r_low]) -- No, Kronecker is not exactly this simple for tensors.
    # Actually, for matrix multiplication factors, the Kronecker product works such that:
    # A = kron(A1, A0) where A1 is 2x2 block and A0 is 2x2 element?
    # The tensor decomposition for $2 \otimes 2$ is indeed the product of the ranks.
    # U4[i,j,R] = U2[i_hi, j_hi, s_hi] * U2[i_lo, j_lo, r_lo]
    
    # We can build U4 using broadcasting:
    # i_hi, i_lo, j_hi, j_lo, s_hi, r_lo
    # i = 2*i_hi + i_lo
    # j = 2*j_hi + j_lo
    # R = 7*s_hi + r_lo
    
    # Let's create arrays for the indices
    i_lo = np.array([0, 1])
    i_hi = np.array([0, 1])
    j_lo = np.array([0, 1])
    j_hi = np.array([0, 1])
    r_lo = np.arange(7)
    r_hi = np.arange(7)
    
    # Meshgrid for 4D spatial + 2D rank
    # This might be memory heavy (2*2*2*2*7*7 = 392 elements, trivial)
    
    # Let's just loop over the 49 ranks but vectorize the 4x4 part inside the loop.
    # This is much faster than nested loops.
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
    # Pre-calculate the mapping from linear R to (s_hi, r_lo)
    rank_indices = np.unravel_index(np.arange(49), (7, 7))
    s_hi_arr, r_lo_arr = rank_indices # Both shape (49,)
    
    for R in range(49):
        s_hi = s_hi_arr[R]
        r_lo = r_lo_arr[R]
        
        # Kronecker product of 2x2 matrices? 
        # No, the tensor components multiply element-wise for the specific indices.
        # U4[i, j, R] = U2[i//2, j//2, s_hi] * U2[i%2, j%2, r_lo]
        
        # Vectorized construction for the 4x4 slice
        # We can use np.kron on the 2x2 slices U2[:,:,s_hi] and U2[:,:,r_lo]?
        # Let M1 = U2[:,:,s_hi] (2x2), M0 = U2[:,:,r_lo] (2x2).
        # The Kronecker product M1 ⊗ M0 gives a 4x4 matrix K where K[i,j] = M1[i//2, j//2] * M0[i%2, j%2].
        # Yes, this is exactly the structure required.
        
        M1_u = U2[:, :, s_hi]
        M0_u = U2[:, :, r_lo]
        U4[:, :, R] = np.kron(M1_u, M0_u)
        
        M1_v = V2[:, :, s_hi]
        M0_v = V2[:, :, r_lo]
        V4[:, :, R] = np.kron(M1_v, M0_v)
        
        M1_w = W2[:, :, s_hi]
        M0_w = W2[:, :, r_lo]
        W4[:, :, R] = np.kron(M1_w, M0_w)
        
        if time.time() > deadline - 5.0:
            break

    # Build 4x4x8 factors
    # Rank 98: Two independent 4x4x4 blocks for the 8 columns of B.
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Flatten U4, V4, W4 to 2D
    # U4 shape (4,4,49) -> (16, 49)
    U4_flat = U4.reshape(16, 49)
    V4_flat = V4.reshape(16, 49)
    W4_flat = W4.reshape(16, 49)
    
    # Fill first half (k=0..3)
    U[:, :49] = U4_flat
    V[:, :49] = V4_flat
    W[:, :49] = W4_flat
    
    # Fill second half (k=4..7) with same pattern
    # Note: The problem is A(4x4) * B(4x8).
    # The output C(4x8) has 8 columns.
    # The decomposition is essentially two parallel 4x4x4 multiplications.
    # For k=4..7, we use the same factors as k=0..3.
    
    U[:, 49:] = U4_flat
    V[:, 49:] = V4_flat
    W[:, 49:] = W4_flat
    
    # Verification
    if time.time() < deadline - 10.0:
        U3 = U.reshape(4, 4, 98)
        V3 = V.reshape(4, 8, 98)
        W3 = W.reshape(8, 4, 98)
        
        # Use einsum to verify
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        
        # The tensor for matrix multiplication C_ik = sum_j A_ij B_jk is represented by T_ijk
        # Here our indices are C(i,k) from A(i,j) B(j,k).
        # The convention in the prompt says output factor follows W[k*n+i, r].
        # Let's check if the reconstruction matches the identity tensor for MM.
        # For random A, B, C should be A*B.
        # Our factors satisfy sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta? 
        # Actually, the tensor decomposition of MM_{n,m,p} is T_ijk = 1 for all i,j,k? No.
        # T_ijk represents the coefficient of the monomial a_ij b_jk in the polynomial for C_ik.
        # For standard MM, T_ijk = 1 if the indices match the definition?
        # The tensor is $T_{i,j,k} = 1$ where the indices are free?
        # Let's just check if the factors produce the standard MM tensor.
        # The standard MM tensor has T[i,j,k] = 1.
        # Wait, the tensor is T(i,j,k) = 1 for all i,j,k? 
        # Let's re-read the tensor decomposition definition.
        # MM_{n,m,p} tensor has entries 1 at (i, j, k).
        # So we expect result[i,k] to be something related to identity?
        # Actually, sum_r U[i,j,r] V[j,k,r] W[k,i,r] should equal 1 for all i,j,k?
        # Let's test with n=2,m=2,p=2 Strassen.
        # Sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # If this sum is 1 for all i,j,k, then it represents the tensor with all 1s.
        # Does MM have all 1s?
        # C_ik = sum_j A_ij B_jk. The coefficient of A_ij B_jk is 1.
        # Yes, the tensor entries are all 1.
        
        if not np.allclose(result, 1.0, atol=1e-10):
             # Fallback to schoolbook
             rank_school = n * m * p
             U_s = np.zeros((n * m, rank_school))
             V_s = np.zeros((m * p, rank_school))
             W_s = np.zeros((p * n, rank_school))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U_s[i * m + j, col] = 1
                         V_s[j * p + k, col] = 1
                         W_s[k * n + i, col] = 1
                         col += 1
             return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
