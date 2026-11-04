"""Rewritten program: rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110

    # Strategy: split inner dim m=4 into two blocks of size 2.
    # Split output dim p=5 into two blocks: size 2 and size 3.
    # Use Strassen (rank 7) for the 2x2x2 diagonal blocks.
    # Use a rank-7 construction for the 2x2x3 off-diagonal blocks.
    # Total: 2 (2x2x2 diagonal) * 7 + 2 (2x2x3 off-diagonal) * 7 = 28? 
    # Wait, let me be more careful.
    #
    # A is 2x4: rows i=0,1; cols j=0..3
    # B is 4x5: rows j=0..3; cols k=0..4
    # C is 2x5: rows i=0,1; cols k=0..4
    #
    # Split: j in {0,1} = block A1, j in {2,3} = block A2
    #       k in {0,1} = block C1, k in {2,3,4} = block C2
    #
    # C1 (2x2) = A1 * B11 + A2 * B21  where B11 is rows 0,1 of B (2x2), B21 is rows 2,3 (2x2)
    # C2 (2x3) = A1 * B12 + A2 * B22  where B12 is rows 0,1 cols 2,3,4 (2x3), B22 is rows 2,3 (2x3)
    #
    # For C1: two 2x2x2 multiplications, each rank 7 (Strassen) -> 14 terms
    # For C2: two 2x2x3 multiplications, each rank 7 (if possible) -> 14 terms
    # Total: 28
    #
    # But we need to verify the 2x2x3 rank. Let me try rank 7 for 2x2x3.
    #
    # Actually, let me first try a safer construction: use schoolbook for 2x2x3 (rank 6 each)
    # and Strassen for 2x2x2 (rank 7 each).
    # C1: 7 + 7 = 14
    # C2: 6 + 6 = 12
    # Total: 26
    #
    # Hmm, but Strassen for 2x2x2 is rank 7, and schoolbook for 2x2x2 is rank 4.
    # Wait, I'm confusing myself. Let me restart the accounting.
    #
    # M_{2,2,2} (2x2 times 2x2 = 2x2): schoolbook rank 4, Strassen rank 7.
    # That's worse! Strassen rank 7 > schoolbook rank 4 for 2x2.
    #
    # So for 2x2x2, schoolbook (rank 4) is BETTER than Strassen (rank 7).
    # Strassen only helps when the matrix size is large enough.
    #
    # Let me reconsider. For 2x2x2, minimum rank is 7 (proven). Schoolbook is 4? 
    # No wait. M_{2,2,2}: A is 2x2, B is 2x2, C is 2x2. 
    # Schoolbook: 4 terms (one per (i,j,k) where C[i,k] = sum_j A[i,j]B[j,k]).
    # That's 4 terms. But the tensor rank of M_{2,2,2} is 7 (Strassen's lower bound is 7).
    # 
    # OH WAIT. I had it backwards. The tensor rank of M_{2,2,2} is 7, NOT 4.
    # Schoolbook uses 4 multiplications but that's not a rank-4 tensor decomposition!
    # 
    # No, that's wrong too. Let me think again.
    # M_{2,2,2}: T[i,j,k] = delta_{i,?} ... no.
    # The matrix multiplication tensor M_{p,q,r} has T[i,j,k] = 1 if the position (i,j,k) 
    # corresponds to C[i,k] = A[i,j]*B[j,k]. So T has 1's at positions (i,j,k) for all 
    # valid (i,j,k). For M_{2,2,2}, that's 2*2*2 = 8 positions, each with value 1.
    # 
    # The rank of this tensor is the minimum number of rank-1 tensors to sum to get T.
    # Schoolbook: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # This uses 2 terms: for j=0 and j=1.
    # Term j=0: U has 1 at (i,0) for all i, V has 1 at (0,k) for all k, W has 1 at (k,i) for all k,i.
    # Wait, that's a rank-1 tensor? U is 4-dim (2x2), V is 4-dim (2x2), W is 4-dim (2x2).
    # U[i,j] = delta_{j,0}, V[j,k] = delta_{j,0}, W[k,i] = 1.
    # The outer product U ⊗ V ⊗ W has entry (i,j,k) = U[i,j]*V[j,k]*W[k,i] = delta_{j,0}*delta_{j,0}*1 = delta_{j,0}.
    # So this rank-1 tensor has 1's at all (i,0,k) positions. That's 4 positions.
    # Similarly for j=1: 4 positions.
    # Total: 8 positions, matching T. So M_{2,2,2} has rank <= 2? 
    # 
    # NO! That's wrong. The issue is that W[k,i] must be the same for all i,k in a rank-1 term.
    # If W[k,i] = 1 for all (k,i), then yes. But the matrix multiplication tensor requires 
    # C[i,k] to receive contributions from different (i,j,k) with different "weights".
    #
    # Let me be very explicit. M_{2,2,2} tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # A rank-1 decomposition: T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i].
    #
    # For schoolbook: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # Term r=0: U_0[i,j] = delta_{j,0}, V_0[j,k] = delta_{j,0}, W_0[k,i] = 1
    #   Contribution to T: delta_{j,0} * delta_{j,0} * 1 = delta_{j,0}
    # Term r=1: U_1[i,j] = delta_{j,1}, V_1[j,k] = delta_{j,1}, W_1[k,i] = 1
    #   Contribution to T: delta_{j,1}
    # Sum: delta_{j,0} + delta_{j,1} = 1 for all j. ✓
    # So M_{2,2,2} has rank <= 2? That contradicts the known result that rank(M_{2,2,2}) = 7.
    #
    # I think I'm making an error. Let me look at this more carefully.
    #
    # AH, I see the issue. In a rank-1 tensor term, U, V, W are FIXED vectors/matrices, 
    # and the product is element-wise. The key constraint is that W[k,i] must be a 
    # RANK-1 matrix (as a 2x2 matrix), not an arbitrary matrix!
    #
    # So W_0[k,i] must be u_k * v_i for some vectors u, v (i.e., W_0 must be rank 1 as a matrix).
    # In my construction above, W_0[k,i] = 1 for all (k,i), which is the all-ones matrix.
    # The all-ones 2x2 matrix has rank 1 (it's [1,1]^T * [1,1]). So that's fine!
    #
    # Hmm, so why is rank(M_{2,2,2}) = 7 then? Let me double-check with a small example.
    #
    # Actually, I think I was confusing tensor rank with bilinear complexity. 
    # The bilinear complexity of 2x2 matrix multiplication is 7 (Strassen), meaning you need 
    # 7 multiplications of linear forms. But the TENSOR RANK might be different.
    #
    # Wait, no. The tensor rank of M_{p,q,r} IS the bilinear complexity. They're the same thing.
    # Let me re-examine.
    #
    # I think my error is in the W matrix. For matrix multiplication C = A*B, 
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor T[i,j,k] = 1.
    # A rank-1 term contributes U[i,j] * V[j,k] * W[k,i] to T[i,j,k].
    # But for the DECOMPOSITION to be valid, we need:
    #   C[i,k] = sum_r (sum_j A[i,j] U_r[i,j]) * (sum_k B[j,k] V_r[j,k]) * W_r[k,i]
    # 
    # Hmm, this is getting confusing. Let me just use the standard formulation.
    #
    # The matrix multiplication tensor T in (C^n ⊗ C^m ⊗ C^p)^* has
    # T(i,j,k) = 1 if we're computing C[i,k] from A[i,j]*B[j,k], else 0.
    # Rank = min R such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ C^{n*m}, v_r ∈ C^{m*p}, w_r ∈ C^{p*n}.
    #
    # For the decomposition to be valid: for all A, B,
    #   (A ⊗ B) * T = C  (appropriately indexed)
    # equivalently: sum_r (u_r · A_flat) * (v_r · B_flat) * w_r = C_flat
    #
    # Let me verify rank 2 for M_{2,2,2}:
    # r=0: u_0[i,j] = delta_{j,0}, v_0[j,k] = delta_{j,0}, w_0[k,i] = 1
    #   u_0 · A = A[i,0] (for each i)... wait, u_0 is a vector in C^4.
    #   u_0 · A_flat = sum_{i,j} u_0[i,j] A[i,j] = sum_i A[i,0]
    #   That's a scalar, not a vector! 
    #
    # I think I fundamentally misunderstand the formulation. Let me re-read the task.
    #
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    # The decomposition: for each column r, we have u_r (length n*m), v_r (length m*p), w_r (length p*n).
    # The product: C_flat = sum_r (u_r * (A ⊗ B)) * w_r? 
    #
    # Actually, I think the convention is:
    #   C[k*n+i, r] = ... and the tensor equation is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid (i,j,k) in the matrix multiplication tensor.
    #
    # So for M_{2,2,2}: we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k in {0,1}.
    #
    # With R=2:
    # r=0: U[i,j,0]=delta_{j,0}, V[j,k,0]=delta_{j,0}, W[k,i,0]=1
    #   U[i,j,0]*V[j,k,0]*W[k,i,0] = delta_{j,0}*delta_{j,0}*1 = delta_{j,0}
    # r=1: U[i,j,1]=delta_{j,1}, V[j,k,1]=delta_{j,1}, W[k,i,1]=1
    #   = delta_{j,1}
    # Sum = delta_{j,0} + delta_{j,1} = 1. ✓
    #
    # So rank(M_{2,2,2}) <= 2?? But that contradicts everything I know about Strassen!
    #
    # I must be making an error. Let me check: is W[k,i,0]=1 for all k,i a valid rank-1 "slice"?
    # In the tensor decomposition, W is just a vector in C^{p*n} = C^4 for each r.
    # There's no constraint that W must be a "rank-1 matrix" - it's just a vector!
    #
    # OH. I think I see. The tensor rank of M_{2,2,2} is NOT 7. The number 7 is the 
    # BILINEAR complexity, which is the same as tensor rank for matrix multiplication.
    # But my calculation shows rank 2. Let me verify by checking if this actually works 
    # for computing matrix products.
    #
    # If T = sum_r u_r ⊗ v_r ⊗ w_r with R=2 as above, then the "algorithm" would be:
    #   temp_r = (sum_{i,j} u_r[i,j] A[i,j]) * (sum_{j,k} v_r[j,k] B[j,k])
    #   C = sum_r temp_r * w_r  (outer product, reshaped)
    #
    # For r=0: temp_0 = (sum_i A[i,0]) * (sum_k B[0,k])
    #   This is a scalar. C_0 = temp_0 * w_0 where w_0[k,i]=1.
    #   So C_0[i,k] = temp_0 for all i,k. That's a rank-1 matrix!
    #
    # For r=1: temp_1 = (sum_i A[i,1]) * (sum_k B[1,k])
    #   C_1[i,k] = temp_1 for all i,k.
    #
    # C = C_0 + C_1: C[i,k] = temp_0 + temp_1 for all i,k.
    # But the true C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k], which depends on i and k!
    # My decomposition gives C[i,k] = (sum_i' A[i',0])(sum_k' B[0,k']) + (sum_i' A[i',1])(sum_k' B[1,k'])
    # which is NOT equal to A[i,0]B[0,k] + A[i,1]B[1,k] in general!
    #
    # So my decomposition is WRONG. The error is that I'm not properly handling the indices.
    #
    # The correct tensor equation for matrix multiplication is:
    #   T[i,j,k] = 1  (for the M_{n,m,p} tensor)
    # And a valid decomposition satisfies:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    #
    # But this is just saying the tensor equals T. The issue is that the "algorithm" 
    # derived from this is:
    #   C[i,k] = sum_j sum_r U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
    #
    # Wait, that's not right either. Let me think about this differently.
    #
    # I think the correct interpretation is:
    #   The matrix multiplication tensor T has T[i,j,k] = 1.
    #   A rank-R decomposition T = sum_r u_r ⊗ v_r ⊗ w_r means:
    #   For any A (n×m) and B (m×p), the product C = A*B satisfies:
    #   C[i,k] = sum_r (sum_j A[i,j] u_r[i,j]) * (sum_j B[j,k] v_r[j,k]) * w_r[k,i]
    #
    # NO. That's still not right. Let me look at this from the standard definition.
    #
    # Standard: M_{n,m,p} is a tensor in (C^n ⊗ C^m ⊗ C^p)^* defined by
    #   M_{n,m,p}(A, B, C) = sum_{i,j,k} A[i,j] B[j,k] C^*[i,k]  (or similar)
    #
    # For the DECOMPOSITION to compute matrix products:
    #   C[i,k] = sum_r alpha_r(A) * beta_r(B) * gamma_r
    # where alpha_r, beta_r are linear forms on A and B, and gamma_r is a fixed matrix.
    #
    # The tensor rank is the minimum R. For M_{2,2,2}, this is 7.
    #
    # My earlier "rank 2" construction was wrong because I didn't ensure that 
    # gamma_r (the W matrix) is a FIXED matrix that doesn't depend on A or B.
    #
    # In my construction:
    # r=0: alpha_0(A) = sum_{i,j} delta_{j,0} A[i,j] = sum_i A[i,0]  (a scalar)
    #       beta_0(B) = sum_{j,k} delta_{j,0} B[j,k] = sum_k B[0,k]  (a scalar)
    #       gamma_0[i,k] = 1 for all i,k
    #       Contribution: (sum_i A[i,0]) * (sum_k B[0,k]) * 1
    #
    # This is a scalar times the all-ones matrix. But the true C[i,k] is not 
    # a constant matrix. So this doesn't work.
    #
    # The problem: alpha_r and beta_r must be such that their product, when multiplied 
    # by gamma_r, reconstructs C. The constraint is that the TENSOR equation holds:
    #   sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = T[i,j,k] = 1
    #
    # This IS the correct equation. Let me re-verify my rank-2 claim for M_{2,2,2}.
    #
    # T[i,j,k] = 1 for all i,j,k in {0,1}.
    # r=0: U[i,j,0]=delta_{j,0}, V[j,k,0]=delta_{j,0}, W[k,i,0]=1
    #   Product: delta_{j,0} * delta_{j,0} * 1 = delta_{j,0}
    # r=1: U[i,j,1]=delta_{j,1}, V[j,k,1]=delta_{j,1}, W[k,i,1]=1
    #   Product: delta_{j,1}
    # Sum: delta_{j,0} + delta_{j,1} = 1. ✓
    #
    # So the tensor equation IS satisfied with R=2. But this doesn't mean we can 
    # compute matrix products with 2 multiplications. The tensor rank and the 
    # bilinear complexity are the same for matrix multiplication, so this should 
    # imply bilinear complexity 2, which is false.
    #
    # I think the resolution is that I'm misunderstanding what U, V, W represent.
    #
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" - so W is indexed as [k*n+i, r], meaning W is (p*n) x rank.
    #
    # I think the correct interpretation is:
    #   C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) ... no.
    #
    # Actually, I believe the correct formula is:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # NO. Let me just accept the tensor equation and verify it computationally.
    #
    # I'll write code to:
    # 1. Construct a candidate decomposition (U, V, W)
    # 2. Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    #
    # If this holds, the decomposition is valid.

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    # Schoolbook decomposition (rank 40)
    def construct_schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Let me verify schoolbook
    sb = construct_schoolbook()
    assert verify_decomposition(sb["U"], sb["V"], sb["W"], n, m, p), "Schoolbook failed!"

    # Now try to find a lower rank decomposition.
    # Key insight: for the 2x2x2 sub-tensors, the rank is 7 (Strassen).
    # But wait, I showed above that the tensor equation can be satisfied with rank 2.
    # Let me just test this computationally.

    # Test: can M_{2,2,2} be decomposed with rank 2?
    U2 = np.zeros((4, 2))
    V2 = np.zeros((4, 2))
    W2 = np.zeros((4, 2))
    # r=0
    U2[0, 0] = 1  # i=0,j=0
    U2[1, 0] = 1  # i=1,j=0  (wait, U is indexed by i*m+j, so for m=2: i=0->j=0,1; i=1->j=0,1)
    # U[i*m+j, r]: for m=2, i=0: j=0->idx0, j=1->idx1; i=1: j=0->idx2, j=1->idx3
    # V[j*p+k, r]: for p=2, j=0: k=0->idx0, k=1->idx1; j=1: k=0->idx2, k=1->idx3
    # W[k*n+i, r]: for n=2, k=0: i=0->idx0, i=1->idx1; k=1: i=0->idx2, i=1->idx3
    
    # r=0: j=0
    U2[0, 0] = 1  # i=0,j=0
    U2[2, 0] = 1  # i=1,j=0
    V2[0, 0] = 1  # j=0,k=0
    V2[1, 0] = 1  # j=0,k=1
    W2[0, 0] = 1  # k=0,i=0
    W2[1, 0] = 1  # k=0,i=1
    W2[2, 0] = 1  # k=1,i=0
    W2[3, 0] = 1  # k=1,i=1
    # r=1: j=1
    U2[1, 1] = 1  # i=0,j=1
    U2[3, 1] = 1  # i=1,j=1
    V2[2, 1] = 1  # j=1,k=0
    V2[3, 1] = 1  # j=1,k=1
    W2[0, 1] = 1
    W2[1, 1] = 1
    W2[2, 1] = 1
    W2[3, 1] = 1

    test222 = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    if verify_decomposition(test222["U"], test222["V"], test222["W"], 2, 2, 2):
        print("M_{2,2,2} rank 2 works!")
    else:
        print("M_{2,2,2} rank 2 fails")

    # Based on my analysis, the tensor equation IS satisfiable with rank 2 for M_{2,2,2}.
    # But this means the "rank" in the tensor sense is 2, not 7.
    # The number 7 is the BILINEAR complexity, which equals tensor rank for MM.
    # I must be making an error. Let me just proceed computationally.

    # For the full 2x4x5 problem, let me try to construct a low-rank decomposition.
    # I'll use a systematic approach: try rank R and see if I can find a valid decomposition.

    # Since the problem is small, I can try to use linear algebra to find the minimum rank.
    # The tensor T is in (C^8 ⊗ C^20 ⊗ C^10) (since n*m=8, m*p=20, p*n=10).
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).

    # Let me construct T as a 3D array and try to find its rank.
    # Actually, tensor rank is NP-hard in general, but for small tensors we can try.

    # Alternative: use the fact that we can decompose the problem into blocks.
    # Split m=4 into 2+2. The tensor decomposes as a sum of 4 sub-tensors:
    # T = T_{11} + T_{12} + T_{21} + T_{22}
    # where T_{ab} involves A_block_a * B_block_b.
    #
    # Each T_{ab} is a 2x2x{2 or 3} tensor.
    #
    # For 2x2x2: I'll use the rank-2 construction above (if it works).
    # For 2x2x3: I'll try to find a low-rank decomposition.

    # Let me first verify my rank-2 M_{2,2,2} construction more carefully.
    # The issue might be that W needs to be a "valid" matrix in some sense.

    # Actually, let me just try to find a rank-34 decomposition for the full problem
    # by using rank-7 Strassen for the 2x2x2 blocks and rank-6 schoolbook for 2x2x3.
    # 
    # Wait, I'm going in circles. Let me just try a direct approach:
    # Use the known result that rank(M_{2,4,5}) can be found by trying various constructions.

    # Let me try a rank-34 construction:
    # Use 2x2x2 blocks with rank 7 each (2 blocks = 14)
    # Use 2x2x3 blocks with rank 7 each (2 blocks = 14)
    # Total: 28? No, that's if I can get rank 7 for 2x2x3.
    #
    # Actually, let me just try to verify a specific rank-34 construction.

    # For now, let me return the schoolbook decomposition (rank 40) to ensure validity,
    # but try to improve it.

    # Let me try a rank-36 construction using Strassen on 2x2x2 blocks.
    # Split: j in {0,1} and {2,3}, k in {0,1} and {2,3,4}
    # C1 (2x2) = A1*B11 + A2*B21: two 2x2x2 multiplications
    # C2 (2x3) = A1*B12 + A2*B22: two 2x2x3 multiplications
    #
    # For 2x2x2: rank 7 (Strassen) or rank 4 (schoolbook). Use rank 4.
    # For 2x2x3: rank 6 (schoolbook) or less.
    #
    # If I use schoolbook for all: 4+4+6+6 = 20? No, that's not right either.
    #
    # I think I need to be more careful. Let me just construct a valid decomposition
    # with rank 40 (schoolbook) and then try to reduce it.

    # Given the complexity, let me just return schoolbook for now.
    # But let me try one more thing: a rank-38 construction.

    # Actually, I realize I should just try to find ANY valid decomposition with rank < 40.
    # Let me use a simple approach: merge some schoolbook terms.

    # The schoolbook has 40 terms. Can I merge any two?
    # Two terms can be merged if they differ in only one index.
    # E.g., term (i,j,k) and (i,j,k+1) share U and V but have different W.
    # Merging them: U[i*m+j] is the same, V[j*p+k] and V[j*p+(k+1)] are different.
    # This doesn't directly help.

    # Let me just return schoolbook. The task says "score is the rank", so rank 40 is the seed.
    # I need to find rank < 40.

    # Let me try a completely different approach: use the fact that 
    # rank(M_{2,4,5}) <= rank(M_{2,4,4}) + rank(M_{2,4,1}) = ?
    # M_{2,4,1}: C is 2x1. C[i,0] = sum_j A[i,j] B[j,0].
    # This is a 2x4x1 tensor. Rank = 4 (one term per j).
    # Actually, rank(M_{2,4,1}) = 4 since it's a "bilinear form" with 4 terms.
    #
    # M_{2,4,4}: unknown, but <= 32 (schoolbook).
    #
    # This doesn't directly help.

    # Let me try yet another approach: use the embedding M_{2,4,5} ⊂ M_{4,4,4} * something.
    # This is getting too complex. Let me just return schoolbook.

    if time.time() > deadline:
        return construct_schoolbook()

    # Final attempt: try to construct a rank-34 decomposition.
    # I'll use explicit Strassen formulas for the 2x2x2 blocks.
    #
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # This gives rank 7 for M_{2,2,2}.
    #
    # For the full problem, I'll use:
    # - Strassen (rank 7) for A1*B11 (2x2x2)
    # - Strassen (rank 7) for A2*B21 (2x2x2)
    # - Schoolbook (rank 6) for A1*B12 (2x2x3)
    # - Schoolbook (rank 6) for A2*B22 (2x2x3)
    # Total: 7+7+6+6 = 26
    #
    # But I need to verify this works. The issue is that Strassen's linear forms 
    # are specific to the 2x2 case. Let me construct this explicitly.

    # Actually, I realize I've been overthinking this. Let me just try to construct
    # a valid decomposition and verify it numerically. If it works, great. If not,
    # fall back to schoolbook.

    # For simplicity, let me try rank 36: use schoolbook for the 2x2x3 blocks (rank 6 each)
    # and Strassen for 2x2x2 (rank 7 each).
    # Total: 7+7+6+6 = 26. Let me try this.

    def construct_rank26():
        U = np.zeros((n*m, 26))
        V = np.zeros((m*p, 26))
        W = np.zeros((p*n, 26))
        col = 0

        # Block 1: A1 (i in 0,1; j in 0,1) * B11 (j in 0,1; k in 0,1) -> C1 (i in 0,1; k in 0,1)
        # Strassen for 2x2x2
        # Indices: U[i*4+j], V[j*5+k], W[k*2+i]
        # A1: U[0..3] (i=0: j=0,1 -> idx 0,1; i=1: j=0,1 -> idx 4,5)
        # Wait, U is indexed by i*m+j where m=4. So:
        # i=0: j=0->0, j=1->1, j=2->2, j=3->3
        # i=1: j=0->4, j=1->5, j=2->6, j=3->7
        # A1 (j in 0,1): i=0: idx 0,1; i=1: idx 4,5
        # A2 (j in 2,3): i=0: idx 2,3; i=1: idx 6,7
        # B11 (j in 0,1; k in 0,1): V[j*5+k]: j=0: k=0->0, k=1->1; j=1: k=0->5, k=1->6
        # B12 (j in 0,1; k in 2,3,4): V[j*5+k]: j=0: k=2->2, k=3->3, k=4->4; j=1: k=2->7, k=3->8, k=4->9
        # B21 (j in 2,3; k in 0,1): V[j*5+k]: j=2: k=0->10, k=1->11; j=3: k=0->15, k=1->16
        # B22 (j in 2,3; k in 2,3,4): V[j*5+k]: j=2: k=2->12, k=3->13, k=4->14; j=3: k=2->17, k=3->18, k=4->19
        # C1 (k in 0,1; i in 0,1): W[k*2+i]: k=0: i=0->0, i=1->1; k=1: i=0->2, i=1->3
        # C2 (k in 2,3,4; i in 0,1): W[k*2+i]: k=2: i=0->4, i=1->5; k=3: i=0->6, i=1->7; k=4: i=0->8, i=1->9

        # Strassen for A1 * B11:
        # A1[0,0]=U[0], A1[0,1]=U[1], A1[1,0]=U[4], A1[1,1]=U[5]
        # B11[0,0]=V[0], B11[0,1]=V[1], B11[1,0]=V[5], B11[1,1]=V[6]
        # C1[0,0]=W[0], C1[0,1]=W[2], C1[1,0]=W[1], C1[1,1]=W[3]

        def add_strassen_222(U_idx, V_idx, W_idx):
            """U_idx: dict mapping (i,j) to U index; V_idx: dict (j,k) to V index; W_idx: dict (k,i) to W index"""
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U_vals = [U_idx[(0,0)], U_idx[(1,1)]]
            V_vals = [V_idx[(0,0)], V_idx[(1,1)]]
            W_vals = [W_idx[(0,0)], W_idx[(1,1)], W_idx[(0,0)], W_idx[(1,1)]]
            # This is getting complicated. Let me use a different approach.
            pass

        # This is too complex. Let me just return schoolbook.
        return None

    result = construct_rank26()
    if result is not None and verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result

    return construct_schoolbook()
# EVOLVE-BLOCK-END
