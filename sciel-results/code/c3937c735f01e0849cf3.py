"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]
import time

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ... 
                # Actually the decomposition is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # Tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For matrix mult, T[i,j,k] = 1 if this is the "multiplication" tensor
                # Wait, the standard form: M_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
                # But our W is indexed [k*n+i, r] which is [k,i]
                # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == delta_{j,k} * something?
                # No. The matrix multiplication tensor M has M[i,j,k] = 1 if we view it as
                # (A ⊗ B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
                # The tensor is T[i,j,k] such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
                # Actually T[i,j,k] = delta_{j,k} is not right either.
                # Standard: M_{ijk} = 1 if the indices match the multiplication pattern.
                # For A (n x m) * B (m x p) = C (n x p):
                # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
                # So the tensor M has M[i,j,k] = 1 for all i,j,k? No.
                # The tensor M is defined by M[i,j,k] = 1 if j is the "inner" index.
                # Actually, the multiplication tensor has entries:
                # M[i,j,k] = 1 if we are computing C[i,k] using A[i,j] and B[j,k].
                # Since for each (i,k), we sum over j, the tensor entry M[i,j,k] is 1
                # for all valid i,j,k. Wait, that's the identity tensor.
                
                # Let me re-read the contract: "exactly decompose the matrix multiplication tensor"
                # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in the valid range.
                # And the decomposition is: T[i,j,k] = sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Wait, that's a rank-1 decomposition of a 3rd order tensor.
                # If T[i,j,k] = 1 for all i,j,k, then we need:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                
                # Let's test this interpretation.
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Rank 15 decomposition for 2x3x3 matrix multiplication
    # This is based on known results that rank(M_{2,3,3}) = 15
    
    # The decomposition uses the structure that 2x3x3 can be decomposed
    # by exploiting the fact that one dimension is 2.
    
    # Known rank-15 construction:
    # We can write 2x3x3 as a combination of smaller multiplications.
    # One approach: use the fact that 2x3 = 2x2 + 2x1, etc.
    
    # Let me use a specific known construction.
    # For M_{2,3,3}, a rank-15 decomposition exists.
    
    # I'll construct it using a systematic approach.
    # The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # We need 15 rank-1 tensors whose sum equals T.
    
    # A known construction uses the following pattern:
    # Think of the 2x3 matrix A and 3x3 matrix B.
    # C = A*B is 2x3.
    
    # We can decompose this as:
    # - Multiply 2x2 part of A with 2x2 part of B (using rank 7 or similar)
    # - Handle the remaining terms
    
    # Actually, let me use a direct algebraic construction.
    
    # The rank of M_{m,n,p} satisfies various bounds.
    # For M_{2,3,3}, the rank is known to be 15.
    
    # Let me try to construct it explicitly.
    
    # One way: use the fact that we can "share" some terms.
    
    # Let me try a different approach: use a known explicit decomposition.
    
    # For 2x3x3, we can use:
    # - 2x2x2 multiplication (rank 7) for the top-left 2x2x2 block
    # - Plus additional terms for the remaining elements
    
    # Actually, let me just hardcode a verified rank-15 decomposition.
    
    # Here's a construction:
    # We'll use 15 terms. The key insight is that for 2x3x3,
    # we can save 3 terms compared to schoolbook (18) by exploiting
    # the small size of the first dimension.
    
    # Let me use a systematic construction based on the following:
    # For each pair (i,k), we need to compute sum_j A[i,j] B[j,k]
    # There are 2*3 = 6 such outputs, each requiring 3 multiplications.
    # Schoolbook: 18 multiplications.
    # We need to reduce to 15.
    
    # One known method: use "addition chains" or "shared subexpressions".
    
    # Let me try to find a valid rank-15 decomposition by construction.
    
    # Actually, I'll use a known result: the rank of M_{2,3,3} is 15,
    # and a specific decomposition can be given.
    
    # Let me construct it using the following approach:
    # Split the 3x3 matrix B into 3 columns, each of size 3x1.
    # Then A (2x3) times each column (3x1) gives a 2x1 vector.
    # So we need to do 3 matrix-vector multiplications of size 2x3 * 3x1.
    
    # For 2x3 * 3x1, the rank is 6 (schoolbook).
    # 3 such multiplications would be rank 18.
    # But we can share some computations.
    
    # Alternative: think of it as 2x3 * 3x3.
    # We can write B as a sum of 3 rank-1 matrices: B = sum_l b_l * c_l^T
    # Then A*B = sum_l (A*c_l) * b_l^T
    # Each A*c_l is a 2x1 vector, requiring 6 multiplications (2x3 * 3x1).
    # 3 such: 18 multiplications. Still schoolbook.
    
    # To get rank 15, we need a more clever decomposition.
    
    # Let me use a known explicit construction from the literature.
    
    # Here's a rank-15 decomposition for M_{2,3,3}:
    
    # I'll use the following 15 terms (U, V, W matrices):
    
    # Term 1-3: Handle the first column of B with all rows of A
    # Term 4-6: Handle the second column of B with all rows of A  
    # Term 7-9: Handle the third column of B with all rows of A
    # But that's 9 terms, not 15.
    
    # Let me think differently.
    
    # Actually, let me just verify that a rank-15 decomposition exists
    # and use a computational approach to find one.
    
    # For now, let me return a schoolbook rank-18 decomposition as a fallback,
    # and try to find a rank-15 one.
    
    # Let me try a specific construction:
    
    # We'll use the following 15 bilinear forms:
    
    # For i in {0,1}, k in {0,1,2}:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # We can write this as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # To reduce the rank, we can combine some of these.
    
    # One approach: define new variables that are sums of A's or B's.
    
    # For example, let:
    # S0 = A[0,0] + A[1,0]
    # S1 = A[0,1] + A[1,1]  
    # S2 = A[0,2] + A[1,2]
    
    # And:
    # T0 = B[0,0] + B[1,0] + B[2,0]
    # T1 = B[0,1] + B[1,1] + B[2,1]
    # T2 = B[0,2] + B[1,2] + B[2,2]
    
    # But this doesn't directly help.
    
    # Let me try a different strategy: use a known rank-15 construction.
    
    # After research, I recall that for M_{2,3,3}, a rank-15 decomposition
    # can be constructed using the following pattern:
    
    # We use 15 bilinear forms. Here's one such construction:
    
    # Let me just hardcode a valid rank-15 decomposition that I can verify.
    
    # Actually, let me use a computational search approach within the time limit.
    
    # For now, let me return the schoolbook rank-18 to ensure correctness,
    # and the score will be 18/18 = 1.0 (same as current).
    
    # To improve, I need to find a valid rank-15 decomposition.
    
    # Let me try to construct one using the following idea:
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need to write T as a sum of 15 rank-1 tensors.
    
    # A rank-1 tensor is u_i * v_j * w_k.
    
    # So we need: sum_{r=1}^{15} u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    # This is equivalent to finding 15 triples (u_r, v_r, w_r) such that
    # their tensor sum equals the all-ones tensor.
    
    # One simple way: use the fact that 1 = 1*1*1, and we can "split" this.
    
    # For example, we can use:
    # (1,1,1) + (1,0,0) + (0,1,0) + (0,0,1) - ... no, this gets complicated.
    
    # Let me use a more systematic approach.
    
    # For 2x3x3, we can use the following construction:
    
    # Term 1: u=[1,1], v=[1,0,0], w=[1,0,0]  -> contributes to T[0,0,0] and T[1,0,0]
    # Term 2: u=[1,0], v=[0,1,0], w=[1,0,0]  -> contributes to T[0,1,0]
    # Term 3: u=[0,1], v=[0,1,0], w=[1,0,0]  -> contributes to T[1,1,0]
    # Term 4: u=[1,0], v=[0,0,1], w=[1,0,0]  -> contributes to T[0,2,0]
    # Term 5: u=[0,1], v=[0,0,1], w=[1,0,0]  -> contributes to T[1,2,0]
    # ... and so on
    
    # But this gives 2*3*3 = 18 terms if we do it naively.
    
    # To reduce to 15, we need to combine some terms.
    
    # Let me try:
    
    # For k=0:
    # We need T[i,j,0] = 1 for all i,j.
    # This is a 2x3 all-ones matrix.
    # We can write this as: [1,1; 1,1; 1,1] = [1;1] * [1,1,1] (rank 1)
    # So for k=0, we can use: u=[1,1], v=[1,1,1], w=[1,0,0]
    # This gives T[i,j,0] = 1 for all i,j. (1 term)
    
    # For k=1:
    # Similarly, u=[1,1], v=[1,1,1], w=[0,1,0]
    # This gives T[i,j,1] = 1 for all i,j. (1 term)
    
    # For k=2:
    # Similarly, u=[1,1], v=[1,1,1], w=[0,0,1]
    # This gives T[i,j,2] = 1 for all i,j. (1 term)
    
    # Total: 3 terms! But wait, this gives T[i,j,k] = 1 for all i,j,k,
    # which is exactly what we want.
    
    # But the rank is 3, not 15. That can't be right.
    
    # Oh, I see the issue. The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Note that U is indexed by (i,j), V by (j,k), W by (k,i).
    # So the indices are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    
    # This is not a standard tensor decomposition where each factor depends
    # on a single index. The factors share indices.
    
    # So my previous analysis was wrong.
    
    # Let me reconsider.
    
    # The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a "structured" tensor decomposition where the factors
    # have shared indices.
    
    # For schoolbook: U[i*m+j, r] = delta_{i,j,k} for the r-th term,
    # V[j*p+k, r] = delta_{i,j,k}, W[k*n+i, r] = delta_{i,j,k}
    # where r = i*m*p + j*p + k.
    
    # To reduce the rank, we need to find a set of (U, V, W) such that
    # the sum equals 1 for all i,j,k.
    
    # This is a non-trivial problem.
    
    # Let me just return the schoolbook rank-18 for now.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
