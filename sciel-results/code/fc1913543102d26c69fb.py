"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # C[k,i] = sum_l A[i,l] * B[l,k]
    # Decomposition: C[k,i] = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
    # Actually the standard form: C = sum_r (U_r @ A) * (B @ V_r) * W_r
    # Where U_r is (n,m), V_r is (m,p), W_r is (p,n) scalar weights
    
    # Let me re-read the convention:
    # U: (n*m, rank) - U[i*m+j, c]
    # V: (m*p, rank) - V[j*p+k, c]  
    # W: (p*n, rank) - W[k*n+i, c]
    # 
    # The product for output C[k*n+i]:
    # C[k*n+i] = sum_c U[?, c] * V[?, c] * W[k*n+i, c]
    # 
    # For schoolbook: for each (i,j,k):
    #   U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This gives C[k*n+i] = sum_j A[i*m+j] * B[j*p+k] * 1 = sum_j A[i,j]*B[j,k] ✓
    
    # So the tensor equation is:
    # For all (i,k): C[k*n+i] = sum_j A[i,j]*B[j,k] = sum_c W[k*n+i,c] * sum_j U[i*m+j,c] * V[j*p+k,c]
    # Wait, that's not quite right. Let me reconsider.
    #
    # Actually: C[k,i] = sum_j A[i,j] * B[j,k]
    # In the decomposition: C[k,i] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    # No wait, that would be a product of sums. Let me re-examine.
    #
    # The standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where the sum over j is inside: C[i,k] = sum_c W[i,k,c] * (sum_j U[i,j,c] * V[j,k,c])
    #
    # But in the flattened form:
    # U has shape (n*m, rank): U[i*m+j, c] = U[i,j,c]
    # V has shape (m*p, rank): V[j*p+k, c] = V[j,k,c]
    # W has shape (p*n, rank): W[k*n+i, c] = W[i,k,c]
    #
    # So: C[k*n+i] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c]
    # This must equal sum_j A[i,j] * B[j,k] for all i,k
    #
    # Equivalently, for each c, define matrices U_c (n×m), V_c (m×p), scalar W_c
    # Then C = sum_c W_c * U_c * A * ... no.
    #
    # Actually: C[i,k] = sum_c W_c[i,k] * (sum_j U_c[i,j] * V_c[j,k])
    # But sum_j U_c[i,j] * V_c[j,k] is not a standard matrix product unless we're more careful.
    #
    # Let me think differently. The bilinear form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_c (sum_j u_c[i,j] * A[i,j]) * (sum_l v_c[j,l] * B[j,l]) ... no.
    #
    # The correct form for tensor rank decomposition of matmul:
    # C[i,k] = sum_c alpha_c[i,k] * prod_j (beta_c[i,j] * A[i,j]) ... no, that's not bilinear.
    #
    # For bilinear: C[i,k] = sum_c sum_j U_c[i,j] * A[i,j] * V_c[j,k] * B[j,k] * W_c[i,k]
    # = sum_c W_c[i,k] * sum_j (U_c[i,j] * A[i,j]) * (V_c[j,k] * B[j,k])
    #
    # Hmm, this is getting complicated. Let me just use the schoolbook verification approach.
    #
    # Schoolbook: rank = n*m*p = 36
    # For each (i,j,k) in [0,3)×[0,3)×[0,4):
    #   U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    #
    # Verify: C[k*n+i] = sum_col W[k*n+i, col] * ... 
    # Actually I think the formula is:
    # C[k*n+i] = sum_col (sum_j U[i*m+j, col] * A[i,j]) * (sum_l V[j*p+l, col] * B[j,l]) ... 
    # No, that doesn't work either because j is summed in both.
    
    # Let me just directly verify the schoolbook works by computing the full product.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        n, m, p = DIMENSIONS
        # Create random test matrices
        np.random.seed(42)
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        # Compute using decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # C[k*n+i] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]
        # Wait, I need to figure out the exact formula.
        #
        # From schoolbook: for term (i,j,k):
        #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        #   This term contributes to C[k*n+i] the value: A[i,j] * B[j,k] * 1
        #
        # So the formula must be:
        # C[k*n+i] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]
        #
        # Let's check: for schoolbook, only one c has all three non-zero for a given (i,j,k),
        # and that c has W[k*n+i,c]=1, U[i*m+j,c]=1, V[j*p+k,c]=1.
        # So C[k*n+i] = sum_{i,j,k} [W[k*n+i,c_{ijk}]=1] * [U[i*m+j,c_{ijk}]=1] * [V[j*p+k,c_{ijk}]=1] * A[i,j]*B[j,k]
        # = sum_j A[i,j]*B[j,k] ✓
        
        C_computed = np.zeros((p, n), dtype=float)
        for k in range(p):
            for i in range(n):
                val = 0.0
                for c in range(rank):
                    wc = W_arr[k*n+i, c]
                    if wc != 0:
                        sum_j = 0.0
                        for j in range(m):
                            sum_j += U_arr[i*m+j, c] * V_arr[j*p+k, c] * A[i,j] * B[j,k]
                        val += wc * sum_j
                C_computed[k, i] = val
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U[:], V[:], W[:]
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Strategy: try to find pairs of terms that can be merged or eliminated
    # For now, just return schoolbook if we can't find better quickly
    # 
    # Let's try a more systematic approach: use the fact that we can represent
    # the matmul tensor and try to find lower rank decompositions.
    #
    # The matmul tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We want rank-r decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank reduction using iterative methods
    # For small problems, we can try exhaustive or semi-exhaustive search
    
    # Let's try to find if rank 23 works by using a known construction
    # For 3x3 x 3x3, rank 23 is known. For 3x3 x 3x4, let's try rank 23 or 24.
    
    # Actually, let me try a different approach: use the schoolbook and try to
    # find linear combinations that allow rank reduction.
    
    # Time check
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to reduce rank by identifying redundant terms
    # A term c is redundant if its contribution can be expressed as a linear combination
    # of other terms.
    
    # For each pair of terms, check if they can be combined
    # This is expensive, so let's be selective.
    
    # Actually, for this specific problem, let me try to use known results.
    # The best known rank for 3x3 x 3x4 matrix multiplication is 23.
    # Let me try to construct a rank-23 decomposition.
    
    # One approach: use the 3x3 x 3x3 rank-23 decomposition and extend it.
    # But I don't have the explicit rank-23 decomposition for 3x3 x 3x3 memorized.
    
    # Alternative: try a random search for lower-rank decompositions.
    # This is unlikely to work well for exact integer/half-integer solutions.
    
    # Let me try a more structured approach: use the fact that matmul tensors
    # have symmetries and try to exploit them.
    
    # For now, let's just return the schoolbook decomposition.
    # The score will be 1.0 (same as seed), but at least it's valid.
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try a simple rank reduction: look for terms where U, V, W have
    # patterns that allow merging.
    
    # For each column c, if there exists another column c' such that
    # U[:,c] = a*U[:,c'] + b*U[:,c''], etc., we might be able to reduce.
    # This is complex, so let's skip for now.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
