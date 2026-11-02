"""Matrix multiplication 2,3,8 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # We want to decompose the 2x3 * 3x8 matrix multiplication tensor.
    # The tensor T has indices (i, j, k) where i in [0,1], j in [0,2], k in [0,7].
    # T[i,j,k] = sum over the multiplication... wait, let's be precise.
    # The matrix multiplication tensor for C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i, j, k] = 1 if we are computing the product A[i,j] * B[j,k].
    # Actually, the standard tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_{i,k}^{(C)} ⊗ e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)}
    # So T has three modes: (i,k), (i,j), (j,k).
    # The factor matrices are:
    # U: shape (n*m, rank) -> indices (i*m+j, r)
    # V: shape (m*p, rank) -> indices (j*p+k, r)
    # W: shape (p*n, rank) -> indices (k*n+i, r)  [Note: task says W[k*n+i,r]]
    
    # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # This means: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But T[i,j,k] is 1 if the indices are valid (which they always are in the full tensor),
    # and the "value" of the tensor entry corresponds to the coefficient in the multiplication.
    # Actually, for matrix multiplication, the tensor is:
    # T_{i,j,k} = 1 (for all valid i,j,k)
    # And the reconstruction is: C[i,k] = sum_{j} A[i,j] B[j,k]
    # The rank-r decomposition satisfies:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that's not right either. Let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T_{n,m,p} is defined by:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not standard. Let me think again.
    
    # The standard definition: The matrix multiplication tensor M_{n,m,p} has:
    # M_{i,j,k} = 1 if we are multiplying A[i,j] by B[j,k]
    # So M is a 3-tensor with entries 1 at positions (i,j,k).
    # The rank-r decomposition: M = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # where u_r has components u_{r,ij}, v_r has components v_{r,jk}, w_r has components w_{r,ik}
    # The condition is: for all i,j,k: sum_r u_{r,ij} * v_{r,jk} * w_{r,ik} = 1
    
    # In the factor matrix notation:
    # U[i*m+j, r] = u_{r,ij}
    # V[j*p+k, r] = v_{r,jk}
    # W[k*n+i, r] = w_{r,ik}
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # The schoolbook decomposition has rank n*m*p = 48.
    # We want to reduce this.
    
    # Strategy: Split the m=3 dimension.
    # Write A as [A1, A2] where A1 is 2x1 and A2 is 2x2.
    # Write B as [B1; B2] where B1 is 1x8 and B2 is 2x8.
    # Then C = A1*B1 + A2*B2.
    
    # A1*B1: 2x1 * 1x8 -> 2x8. This is an outer product, rank 2 (one per output element... no).
    # Actually, A1*B1 where A1 is 2x1 and B1 is 1x8:
    # C1[i,k] = A1[i,0] * B1[0,k]
    # This is a rank-2 tensor (in the sense of tensor rank for this specific multiplication).
    # The tensor for 2x1 * 1x8 has rank 2 (one term per i, since k is free).
    # Wait: T_{i,0,k} = 1. We need sum_r u_{r,i0} v_{r,0k} w_{r,ik} = 1.
    # If we set R=2: for r=0 (i=0): u_{0,00}=1, v_{0,0k}=1 for all k, w_{0,k0}=1 for all k.
    # For r=1 (i=1): u_{1,10}=1, v_{1,0k}=1 for all k, w_{1,k1}=1 for all k.
    # Then sum_r u_{r,i0} v_{r,0k} w_{r,ik} = u_{i,0} * 1 * 1 = 1. Yes, rank 2.
    
    # A2*B2: 2x2 * 2x8 -> 2x8.
    # This is 2x2 * 2x8. We can use Strassen for the 2x2 part, but B has 8 columns.
    # Actually, we can view this as: for each column k of B2, we compute A2 * B2[:,k].
    # A2 is 2x2, B2[:,k] is 2x1. So each column is a 2x2 * 2x1 multiplication.
    # The tensor for 2x2 * 2x1: T_{i,j,k} where i in [0,2), j in [0,2), k in [0,1).
    # Rank of 2x2*2x1: This is equivalent to multiplying a 2x2 matrix by a vector.
    # C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # The tensor has entries T_{i,j,0} = 1.
    # Rank: We need sum_r u_{r,ij} v_{r,j0} w_{r,i0} = 1.
    # Schoolbook: 2*2*1 = 4. Can we do better?
    # For 2x2*2x1, the rank is actually 4 (it's equivalent to 2x2*2x1, which is the same as 2x2*1x2 transposed... no).
    # Actually, 2x2 * 2x1: the output is 2x1. The tensor is T_{i,j,k} with k always 0.
    # This is a 2x2x1 tensor. The rank of a 2x2x1 tensor... 
    # Let me check: we need u_{r,ij} v_{r,j} w_{r,i} such that sum_r u_{r,ij} v_{r,j} w_{r,i} = 1 for all i,j.
    # This is like decomposing the identity in some sense.
    # For i=0,j=0: sum_r u_{r,00} v_{r,0} w_{r,0} = 1
    # For i=0,j=1: sum_r u_{r,01} v_{r,1} w_{r,0} = 1
    # For i=1,j=0: sum_r u_{r,10} v_{r,0} w_{r,1} = 1
    # For i=1,j=1: sum_r u_{r,11} v_{r,1} w_{r,1} = 1
    # With R=4: schoolbook works. Can we do R=3?
    # This is the same as the rank of the 2x2 identity-like tensor. I believe the rank is 4.
    
    # So for A2*B2 (2x2 * 2x8):
    # We can process each column independently, but that gives 8 * 4 = 32.
    # Alternatively, we can use the structure: B2 is 2x8, so we have 8 independent 2x1 vectors.
    # But we can also think of it as: C = A2 * B2 where A2 is 2x2 and B2 is 2x8.
    # This is equivalent to 8 independent 2x2*2x1 multiplications, but they share A2.
    # The tensor for 2x2*2x8: T_{i,j,k} = 1 for i in [0,2), j in [0,2), k in [0,8).
    # Rank: schoolbook is 2*2*8 = 32.
    # Can we do better? 
    # We can use Strassen-like techniques. The 2x2*2x8 can be seen as 4 independent 1x1*2x8? No.
    # Let's use the known result: rank of 2x2*2xp.
    # For p=1, rank=4. For p=2, rank=8? Or can we do better?
    # Actually, 2x2*2x2 has rank 7 (Strassen).
    # 2x2*2x8: We can split the 8 columns into 4 groups of 2.
    # Each 2x2*2x2 block has rank 7 (Strassen).
    # So 4 * 7 = 28.
    # Total: 2 (for A1*B1) + 28 (for A2*B2) = 30.
    
    # But wait, the knowledge says: "Reduced rank from schoolbook 48 to 44 by splitting m=3 into 1+2 and using Strassen for the 2x2x2 blocks of the 2x2x8 part"
    # So they got 44. Let me re-examine.
    # A1*B1: 2x1 * 1x8. Rank = 2? Let me verify.
    # T_{i,0,k} = 1. We need sum_r u_{r,i0} v_{r,0k} w_{r,ik} = 1.
    # If R=2: r=0 for i=0, r=1 for i=1.
    # u_{0,00}=1, u_{1,10}=1, others 0.
    # v_{0,0k}=1 for all k, v_{1,0k}=1 for all k.
    # w_{0,k0}=1 for all k, w_{1,k1}=1 for all k.
    # Check: for i=0, j=0, any k: sum_r u_{r,00} v_{r,0k} w_{r,k0} = 1*1*1 + 0 = 1. ✓
    # For i=1, j=0, any k: sum_r u_{r,10} v_{r,0k} w_{r,k1} = 0 + 1*1*1 = 1. ✓
    # So rank 2 for A1*B1.
    
    # A2*B2: 2x2 * 2x8.
    # Split 8 columns into 4 groups of 2: B2 = [B2_0, B2_1, B2_2, B2_3] where each B2_t is 2x2.
    # C = A2 * B2 = [A2*B2_0, A2*B2_1, A2*B2_2, A2*B2_3]
    # Each A2*B2_t is 2x2*2x2, which has rank 7 (Strassen).
    # So total for A2*B2: 4 * 7 = 28.
    # Grand total: 2 + 28 = 30.
    
    # But the knowledge says they got 44. So maybe my analysis is wrong, or they did something different.
    # Let me re-read: "splitting m=3 into 1+2 and using Strassen for the 2x2x2 blocks of the 2x2x8 part"
    # Maybe they didn't split the 8 columns into groups of 2?
    # Or maybe the Strassen decomposition for 2x2*2x2 doesn't directly apply when the output indices are interleaved?
    
    # Let me think about this more carefully. The issue is the index mapping.
    # For A2*B2_t (2x2 * 2x2), the output is 2x2.
    # In the full problem, the output index is k*n+i = k*2+i.
    # For B2_t (columns 2t, 2t+1), the output indices are (2t)*2+i and (2t+1)*2+i.
    # So the output indices for block t are: 4t, 4t+1, 4t+2, 4t+3.
    # This is a contiguous block of 4 output indices, which is fine.
    
    # So my analysis should give rank 30. Let me implement it and see.
    
    # Actually, wait. Let me reconsider the Strassen decomposition for 2x2*2x2.
    # Strassen: C = A*B where A,B,C are 2x2.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # The 7 terms are the rank-1 components.
    # Each term M_r is a scalar (product of two linear combinations).
    # In the tensor decomposition, each term corresponds to a rank-1 tensor.
    
    # For our problem, we need to construct U, V, W for the 2x2*2x2 subproblem.
    # Then we can embed it into the full 2x3*3x8 problem.
    
    # Let me implement the full decomposition with rank 30.
    
    # Actually, let me first verify that rank 30 is achievable by constructing it explicitly.
    
    # A1*B1 part (rank 2):
    # i in [0,2), j=0, k in [0,8)
    # Term r=0: i=0. U[0*3+0, 0]=1, V[0*8+k, 0]=1 for all k, W[k*2+0, 0]=1 for all k.
    # Term r=1: i=1. U[1*3+0, 1]=1, V[0*8+k, 1]=1 for all k, W[k*2+1, 1]=1 for all k.
    
    # A2*B2 part (rank 28):
    # i in [0,2), j in [1,3) -> j_local = j-1 in [0,2), k in [0,8)
    # Split k into 4 groups of 2: k = 2t + s, t in [0,4), s in [0,2)
    # For each t, we have a 2x2*2x2 multiplication.
    # Strassen terms for block t:
    
    # Let's define the Strassen decomposition for 2x2*2x2.
    # A is 2x2: A[i,j] for i,j in [0,2)
    # B is 2x2: B[j,k] for j,k in [0,2)
    # C is 2x2: C[i,k] for i,k in [0,2)
    
    # The 7 Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In the tensor decomposition, each M_r contributes to the U, V, W factors.
    # For a given Strassen term r, we have:
    # u_{r,ij} = coefficient of A[i,j] in the first factor of M_r
    # v_{r,jk} = coefficient of B[j,k] in the second factor of M_r
    # w_{r,ik} = coefficient of C[i,k] in the expression for C[i,k] involving M_r
    
    # Let's list them:
    # M1: first factor = A00+A11, second factor = B00+B11
    #   u: (0,0)->1, (1,1)->1
    #   v: (0,0)->1, (1,1)->1
    #   w: (0,0)->1, (1,1)->1  [since C00 += M1, C11 += M1]
    
    # M2: first factor = A10+A11, second factor = B00
    #   u: (1,0)->1, (1,1)->1
    #   v: (0,0)->1
    #   w: (1,0)->1  [since C10 += M2]
    
    # M3: first factor = A00, second factor = B01-B11
    #   u: (0,0)->1
    #   v: (0,1)->1, (1,1)->-1
    #   w: (0,1)->1, (1,1)->1  [since C01 += M3, C11 += M3]
    
    # M4: first factor = A11, second factor = B10-B00
    #   u: (1,1)->1
    #   v: (1,0)->1, (0,0)->-1
    #   w: (0,0)->1, (1,0)->1  [since C00 += M4, C10 += M4]
    
    # M5: first factor = A00+A10, second factor = B11
    #   u: (0,0)->1, (1,0)->1
    #   v: (1,1)->1
    #   w: (0,0)->-1, (0,1)->1  [since C00 -= M5, C01 += M5]
    
    # M6: first factor = A01-A11, second factor = B10+B11
    #   u: (0,1)->1, (1,1)->-1
    #   v: (1,0)->1, (1,1)->1
    #   w: (1,1)->1  [since C11 += M6]
    
    # M7: first factor = A10-A00, second factor = B00+B01
    #   u: (1,0)->1, (0,0)->-1
    #   v: (0,0)->1, (0,1)->1
    #   w: (0,0)->1  [since C00 += M7]
    
    # Now, let's construct the full U, V, W matrices.
    # n=2, m=3, p=8
    # U shape: (2*3, 30) = (6, 30)
    # V shape: (3*8, 30) = (24, 30)
    # W shape: (8*2, 30) = (16, 30)
    
    rank = 30
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(16)]
    
    # A1*B1 part: columns 0-1
    # Column 0: i=0
    U[0*3+0][0] = 1  # U[0,0]
    for k in range(8):
        V[0*8+k][0] = 1  # V[k,0]
        W[k*2+0][0] = 1  # W[k*2,0]
    # Column 1: i=1
    U[1*3+0][1] = 1  # U[3,1]
    for k in range(8):
        V[0*8+k][1] = 1  # V[k,1]
        W[k*2+1][1] = 1  # W[k*2+1,1]
    
    # A2*B2 part: columns 2-29
    # For each block t (0 to 3), we have 7 Strassen terms.
    # Block t corresponds to k = 2t, 2t+1.
    # The local indices: i_local in [0,2), j_local in [0,2) (j = j_local + 1), k_local in [0,2) (k = 2t + k_local)
    
    # Strassen coefficients:
    # Each entry is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i_local, j_local, coeff)
    # v_coeffs: list of (j_local, k_local, coeff)
    # w_coeffs: list of (i_local, k_local, coeff)
    
    strassen_terms = [
        # M1
        ([(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)],
         [(0,0,1)],
         [(1,0,1)]),
        # M3
        ([(0,0,1)],
         [(0,1,1), (1,1,-1)],
         [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)],
         [(1,0,1), (0,0,-1)],
         [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (1,0,1)],
         [(1,1,1)],
         [(0,0,-1), (0,1,1)]),
        # M6
        ([(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)],
         [(1,1,1)]),
        # M7
        ([(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)],
         [(0,0,1)]),
    ]
    
    col = 2
    for t in range(4):
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            for (i_loc, j_loc, coeff) in u_coeffs:
                i = i_loc
                j = j_loc + 1  # j_local + 1 since A2 starts at j=1
                U[i*3+j][col] = coeff
            for (j_loc, k_loc, coeff) in v_coeffs:
                j = j_loc + 1
                k = 2*t + k_loc
                V[j*8+k][col] = coeff
            for (i_loc, k_loc, coeff) in w_coeffs:
                i = i_loc
                k = 2*t + k_loc
                W[k*2+i][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
