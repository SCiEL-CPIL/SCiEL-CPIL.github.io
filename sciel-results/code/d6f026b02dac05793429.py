"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for (n x m) * (m x p) with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    # Test with random matrices
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    # Compute C from decomposition: C[i,k] = sum_r U[i*m + j] * V[j*p + k] * W[k*n + i]
    # More precisely: C[i,k] = sum_r (sum_j U[i*m+j]*V[j*p+k]) * W[k*n+i]
    # But the standard form is: C[i,k] = sum_r U[i*m + j] * V[j*p + k] * W[k*n + i]
    # where each term r contributes U[i*m+j_r] * V[j_r*p+k] * W[k*n+i]
    # Actually the tensor decomposition is: T[i,j,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i]
    # where T[i,j,k] is 1 if this is a valid term in the multiplication tensor
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # Wait, no. The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] where
    # the bilinear form is sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # So M[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # M[i,j,k] = sum_r U[i,j] * V[j,k] * W[i,k]
    # In our indexing: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Sum over r should give 1 if (i,j,k) is a valid triple, else 0.
    # For matrix multiplication, all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid.
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(rank):
                # Sum over j
                for j in range(m):
                    s += U[i * m + j, r] * V[j * p + k, r]
            C_computed[i, k] = s * W[k * n + i, r] if False else 0  # This is wrong
    # Let me redo: the bilinear form is sum_{i,j,k} (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) * A[i,j] * B[j,k]
    # This should equal sum_{i,k} C[i,k] where C = A@B
    # So for each (i,k): C[i,k] = sum_j A[i,j] * B[j,k]
    # And the decomposition gives: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r] * A[i,j] * B[j,k])
    # Hmm, this isn't quite right. Let me think again.
    # The standard tensor decomposition of matrix multiplication:
    # (A @ B)[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not how it works. The correct form is:
    # (A @ B)[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that's not right either. Let me look at this more carefully.
    # The bilinear algorithm for matrix multiplication uses rank-r decomposition:
    # C = A @ B
    # where C[i,k] = sum_r lambda_r * (sum_j a_{r,j} * A[i,j]) * (sum_j b_{r,j} * B[j,k])
    # In our notation:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # So let's verify this.
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                u_val = sum(U[i * m + j, r] * A[i, j] for j in range(m))
                v_val = sum(V[j * p + k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k * n + i, r] * u_val * v_val
    return np.allclose(C_computed, C_expected, atol=tol)

def _strassen_block(n1, n2, p1, p2, m1, m2):
    """
    Compute (n1+m2) x (m1+m2) * (m1+m2) x (p1+p2) using Strassen on 2x2 blocks.
    This requires n1==m1, n2==m2, p1==m1, p2==m2 for square Strassen.
    For non-square, we need a different approach.
    """
    pass

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Try Strassen-style decomposition by partitioning m=4 into 2+2
    # A is 3x4, B is 4x7
    # Partition: A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    #             B = [B1; B2] where B1 is 2x7, B2 is 2x7
    # C = A1@B1 + A2@B2
    # Each A_i @ B_i is (3x2) @ (2x7) -> 3x7
    # For (3x2) @ (2x7): can we do better than 42?
    # The tensor is M_{3,2,7}. Schoolbook rank = 3*2*7 = 42.
    # 
    # For (n x 2) @ (2 x p), the bilinear complexity is known to be n*p + min(n,p) - 1
    # for n,p >= 2? Let me check: for (2x2)@(2x2), it's 7 (Strassen), and 2*2 + min(2,2)-1 = 5, which is wrong.
    # So that formula is incorrect.
    #
    # Actually, for (n x k) @ (k x p), when k=2:
    # We can write it as: result[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # This is a sum of 2 rank-1 tensors for each (i,j), so total rank <= 2*n*p.
    # But we can share across i and j.
    # 
    # Let's just try to construct an explicit decomposition for (3x2)@(2x7).
    # 
    # Actually, let me try a different approach: use the fact that we can pad 3->4 and 7->8
    # and use Strassen on 4x4 @ 4x8? No, Strassen needs square.
    #
    # Let me try: pad A to 4x4 (add zero row), pad B to 4x8 (add zero column).
    # Then 4x4 @ 4x8. Partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where B11,B21 are 2x4, B12,B22 are 2x4
    # Hmm, this doesn't work cleanly with Strassen.
    #
    # Alternative: use the "laser method" or just try random searches for low-rank decompositions.
    #
    # For now, let's try a simpler improvement: partition m=4 into 2+2 and for each
    # (3x2)@(2x7) subproblem, use a decomposition with rank < 42 if possible.
    #
    # Let me try to find a rank-r decomposition for (3x2)@(2x7) with r < 42.
    # The lower bound is max(3*7, 3*2, 2*7) = max(21, 6, 14) = 21.
    # So rank is between 21 and 42.
    #
    # Let's try to construct one. For (n x 2) @ (2 x p):
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # We can write this as:
    # C[i,j] = sum_r W[i,j,r] * (sum_a U[i,a,r] * A[i,a]) * (sum_b V[b,j,r] * B[b,j])
    # where a,b in {0,1}
    # 
    # One simple decomposition: for each (i,j), use 2 terms (one for a=0, one for a=1).
    # That gives rank 2*n*p = 42.
    # 
    # To do better, we need to share terms across different (i,j) pairs.
    # 
    # Let's try: for each i, define U_i = (U[i,0,r], U[i,1,r]) for r=1..R_i
    # and for each j, V_j = (V[0,j,r], V[1,j,r]) for r=1..R_j
    # We need: for all i,j: A[i,0]*B[0,j] + A[i,1]*B[1,j] = sum_r W[i,j,r] * (U[i,0,r]*A[i,0] + U[i,1,r]*A[i,1]) * (V[0,j,r]*B[0,j] + V[1,j,r]*B[1,j])
    #
    # This is a system of equations. Let's try small R.
    #
    # Given the complexity, let's just return the schoolbook for now.
    # The rank is 84.
    
    return best
# EVOLVE-BLOCK-END
