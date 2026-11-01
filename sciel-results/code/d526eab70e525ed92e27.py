"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k in the standard basis, but actually:
    # The multiplication tensor has T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # In the decomposition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{i,k}... no.
    
    # Actually, the matrix multiplication tensor M[i,j,k] = 1 for all i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]B[j,k] in (AB)[i,k] is 1.
    
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's verify by checking a few random instances
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB = A @ B
        # Compute using decomposition
        result = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Reconstruct A factor: uc[i*m+j] gives coefficient for A[i,j]
            # Reconstruct B factor: vc[j*p+k] gives coefficient for B[j,k]
            # Reconstruct C factor: wc[k*n+i] gives coefficient for C[i,k]
            A_c = np.zeros((n, m))
            B_c = np.zeros((m, p))
            C_c = np.zeros((n, p))
            for i in range(n):
                for j in range(m):
                    A_c[i, j] = uc[i * m + j]
            for j in range(m):
                for k in range(p):
                    B_c[j, k] = vc[j * p + k]
            for k in range(p):
                for i in range(n):
                    C_c[i, k] = wc[k * n + i]
            result += np.outer(np.dot(A, A_c), np.dot(B_c.T, B)) if False else 0
            
            # Actually, the decomposition works as:
            # C = sum_c (A * u_c) * (v_c * B) where u_c is reshaped from uc, v_c from vc
            # More precisely: C[i,k] = sum_j sum_c A[i,j] * uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
            # Let me just directly compute the tensor contraction
            pass
        
        # Direct tensor check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > tol:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 2*2*10 = 40
    
    # Start with schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Strategy: Look for linear dependencies among the rank-1 terms
    # Each rank-1 term c is defined by vectors u_c (in R^{n*m}), v_c (in R^{m*p}), w_c (in R^{p*n})
    # We want to find coefficients alpha_c such that sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0
    # This is a tensor equation. We can vectorize it.
    
    # Alternative approach: Use the fact that for 2x2x10, we can try to find a lower rank
    # by exploiting structure. The 2x2x2 tensor has rank 7 (Strassen).
    # 2x2x10 = 2x2x2 ⊗ 1x1x5, so naive tensor product gives rank 7*5=35.
    # Can we do better?
    
    # Let's try a direct search: start from the 35-term decomposition (Strassen for 2x2x2 tensor product with 5x identity)
    # and look for linear dependencies.
    
    # Strassen's 7-term decomposition for 2x2x2:
    # The 7 rank-1 terms for 2x2 matrix multiplication:
    strassen_U = [
        [1, 0, 0, 1, 0, 1, 0],  # A[0,0]
        [0, 1, 0, 0, 1, 0, 1],  # A[0,1]
        [0, 0, 1, 0, 0, 1, 0],  # A[1,0]
        [1, 0, 0, 0, 0, 0, 1],  # A[1,1]
    ]
    strassen_V = [
        [1, 0, 1, 0, 0, 0, 0],  # B[0,0]
        [0, 0, 0, 0, 1, 0, 0],  # B[0,1]
        [0, 1, 0, 0, 0, 0, 1],  # B[1,0]
        [0, 0, 0, 1, 0, 1, 0],  # B[1,1]
    ]
    strassen_W = [
        [1, 0, 0, 0, 0, 0, 0],  # C[0,0]
        [0, 1, 0, 0, 0, 0, 0],  # C[0,1]
        [0, 0, 1, 0, 0, 0, 0],  # C[1,0]
        [0, 0, 0, 1, 1, 1, 1],  # C[1,1]
    ]
    
    # Verify Strassen decomposition for 2x2x2
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] should equal 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(strassen_U[i*2+j][c] * strassen_V[j*2+k][c] * strassen_W[k*2+i][c] for c in range(7))
                if val != 1:
                    # Fallback to schoolbook
                    pass
    
    # Build 35-term decomposition: tensor product of Strassen (7 terms) and identity (5 terms)
    # For the 1x1x5 part, the identity tensor has 5 rank-1 terms (just 1 for each k)
    # So total rank = 7 * 5 = 35
    
    rank35 = 35
    U35 = [[0] * rank35 for _ in range(n * m)]
    V35 = [[0] * rank35 for _ in range(m * p)]
    W35 = [[0] * rank35 for _ in range(p * n)]
    
    for s in range(7):
        for t in range(5):
            col = s * 5 + t
            for i in range(2):
                for j in range(2):
                    U35[i*m+j][col] = strassen_U[i*2+j][s]
            for j in range(2):
                for k in range(10):
                    # For the 1x1x5 part: B[j,k] where j in {0,1}, k in {0..9}
                    # The 5 identity terms correspond to k groups? 
                    # Actually, 1x1x5 means multiplying 1x1 by 1x5, so B is 1x5.
                    # But we have 2x10. So the structure is different.
                    pass
    
    # This tensor product approach is getting complicated. Let me try a different strategy.
    
    # Direct approach: Use the schoolbook 40-term decomposition and try to eliminate one term
    # by finding a linear combination.
    
    # For 2x2x10, the tensor lives in R^{4} ⊗ R^{20} ⊗ R^{20}
    # The 40 schoolbook terms are e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    # Let's try to find if there's a rank 39 decomposition by solving a linear system.
    # Assume we can express one term as a linear combination of the others.
    
    # Actually, let's just try to find any linear dependency among the 40 tensor products.
    # Vectorize: each term c is a vector in R^{4*20*20} = R^{1600}
    # Term c: u_c ⊗ v_c ⊗ w_c where u_c = e_{i_c, j_c} (4-dim), v_c = e_{j_c, k_c} (20-dim), w_c = e_{k_c, i_c} (20-dim)
    
    # Build the 40 vectors
    vectors = []
    for i in range(2):
        for j in range(2):
            for k in range(10):
                vec = np.zeros(4 * 20 * 20)
                # u_c is at position (i,j) in the 4-dim space
                u_idx = i * 2 + j
                # v_c is at position (j,k) in the 20-dim space (2*10)
                v_idx = j * 10 + k
                # w_c is at position (k,i) in the 20-dim space (10*2)
                w_idx = k * 2 + i
                # Tensor product position: u_idx * 20*20 + v_idx * 20 + w_idx
                pos = u_idx * 400 + v_idx * 20 + w_idx
                vec[pos] = 1.0
                vectors.append(vec)
    
    vectors_np = np.array(vectors)  # Shape (40, 1600)
    
    # Check rank of this matrix
    rank = np.linalg.matrix_rank(vectors_np, tol=1e-10)
    
    if rank < 40:
        # There's a linear dependency! Find null space
        _, s, vh = np.linalg.svd(vectors_np)
        null_space = vh[rank:]  # Rows corresponding to zero singular values
        # The null space vector gives coefficients for the dependency
        # But we need to reconstruct a valid decomposition from this
        
        # This is complex. For now, just return the schoolbook if no clear improvement.
        pass
    
    # Given the complexity, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
