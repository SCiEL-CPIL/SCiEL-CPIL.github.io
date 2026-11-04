"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Return the 2x2x2 Strassen decomposition.
    U, V, W have shape (4, 7) each.
    T[i*2+j, j*2+k, k*2+i] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # The 7 products are:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Now we need to map this to the tensor form.
    # The tensor for 2x2x2 has indices:
    # U index: i*2 + j where i is row of A, j is col of A (which is row of B)
    # V index: j*2 + k where j is row of B, k is col of B
    # W index: k*2 + i where k is col of B, i is row of C
    
    # Let's construct U, V, W directly.
    # For each product c, we have a linear combination of A entries times a linear combination of B entries.
    # U[i*2+j, c] is the coefficient of A[i,j] in product c
    # V[j*2+k, c] is the coefficient of B[j,k] in product c
    # W[k*2+i, c] is 1 for all (since C entries are just sums of products)
    
    # Actually, W is not always 1. Let me reconsider.
    # The tensor is T[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k in {0,1}
    # A decomposition is: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # So T[i*2+j, j*2+k, k*2+i] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    
    # For Strassen, the 7 products give us:
    # C[r,c] = sum_s A[r,s] * B[s,c]
    # In the tensor form: T[r*2+s, s*2+c, c*2+r] = 1
    # So we need: sum_p U[r*2+s, p] * V[s*2+c, p] * W[c*2+r, p] = 1
    
    # Let me define the 7 products more carefully.
    # Product p corresponds to: (linear combo of A) * (linear combo of B)
    # Then C entries are linear combos of these products.
    
    # Let me use a different approach. Define the decomposition directly.
    # For 2x2x2, the Strassen decomposition has rank 7.
    
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    # U index: r*2 + s (row r, col s of A)
    # V index: s*2 + c (row s, col c of B)  
    # W index: c*2 + r (col c of B, row r of C)
    
    # The 7 products:
    # p0: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # p1: (A[1,0] + A[1,1]) * B[0,0]
    # p2: A[0,0] * (B[0,1] - B[1,1])
    # p3: A[1,1] * (B[1,0] - B[0,0])
    # p4: (A[0,0] + A[0,1]) * B[1,1]
    # p5: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # p6: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C entries:
    # C[0,0] = p0 + p3 - p4 + p6
    # C[0,1] = p2 + p4
    # C[1,0] = p1 + p3
    # C[1,1] = p0 - p1 + p2 + p5
    
    # So the decomposition is:
    # T[r*2+s, s*2+c, c*2+r] = sum_p U[r*2+s, p] * V[s*2+c, p] * W[c*2+r, p]
    
    # For each product p, U[:,p] is the coefficient vector for A entries,
    # V[:,p] is the coefficient vector for B entries,
    # and W[:,p] must be such that the sum gives the correct C coefficients.
    
    # But wait, this isn't quite right. In the tensor decomposition,
    # each term is a rank-1 tensor U_p ⊗ V_p ⊗ W_p.
    # The product of A and B matrices gives C, and the tensor encodes this.
    
    # Let me think again. The matrix multiplication tensor for n x m x p is:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # This means: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # A decomposition T = sum_c U_c ⊗ V_c ⊗ W_c means:
    # (A*B)[i,k] = sum_c (sum_j U_c[i*m+j] * A[i,j]) * (sum_k' V_c[j'*p+k'] * B[j',k']) * W_c[k*n+i]
    
    # Hmm, this is getting confusing. Let me use the standard form.
    # The multiplication tensor is:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # A rank-r decomposition is:
    # T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    
    # This means: for all i,j,k:
    # 1 = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # For the Strassen decomposition of 2x2x2, we can verify it works.
    # Let me just define the matrices directly.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 0: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # u0: A[0,0] coeff = 1, A[1,1] coeff = 1 -> U[0,0]=1, U[3,0]=1
    # v0: B[0,0] coeff = 1, B[1,1] coeff = 1 -> V[0,0]=1, V[3,0]=1
    # This product contributes to C[0,0] (+1), C[0,1] (0), C[1,0] (0), C[1,1] (+1)
    # So w0: W[0,0]=1 (C[0,0]), W[3,0]=1 (C[1,1])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 1: (A[1,0] + A[1,1]) * B[0,0]
    # u1: A[1,0] coeff = 1, A[1,1] coeff = 1 -> U[2,1]=1, U[3,1]=1
    # v1: B[0,0] coeff = 1 -> V[0,1]=1
    # Contributes to C[1,0] (+1), C[1,1] (-1)
    # So w1: W[2,1]=1 (C[1,0]), W[3,1]=-1 (C[1,1])
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Product 2: A[0,0] * (B[0,1] - B[1,1])
    # u2: A[0,0] coeff = 1 -> U[0,2]=1
    # v2: B[0,1] coeff = 1, B[1,1] coeff = -1 -> V[1,2]=1, V[3,2]=-1
    # Contributes to C[0,0] (0), C[0,1] (+1), C[1,0] (0), C[1,1] (+1)
    # So w2: W[1,2]=1 (C[0,1]), W[3,2]=1 (C[1,1])
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 3: A[1,1] * (B[1,0] - B[0,0])
    # u3: A[1,1] coeff = 1 -> U[3,3]=1
    # v3: B[1,0] coeff = 1, B[0,0] coeff = -1 -> V[2,3]=1, V[0,3]=-1
    # Contributes to C[0,0] (+1), C[0,1] (0), C[1,0] (+1), C[1,1] (0)
    # So w3: W[0,3]=1 (C[0,0]), W[2,3]=1 (C[1,0])
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 4: (A[0,0] + A[0,1]) * B[1,1]
    # u4: A[0,0] coeff = 1, A[0,1] coeff = 1 -> U[0,4]=1, U[1,4]=1
    # v4: B[1,1] coeff = 1 -> V[3,4]=1
    # Contributes to C[0,0] (-1), C[0,1] (+1), C[1,0] (0), C[1,1] (0)
    # So w4: W[0,4]=-1 (C[0,0]), W[1,4]=1 (C[0,1])
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 5: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # u5: A[1,0] coeff = 1, A[0,0] coeff = -1 -> U[2,5]=1, U[0,5]=-1
    # v5: B[0,0] coeff = 1, B[0,1] coeff = 1 -> V[0,5]=1, V[1,5]=1
    # Contributes to C[0,0] (0), C[0,1] (0), C[1,0] (0), C[1,1] (+1)
    # So w5: W[3,5]=1 (C[1,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Product 6: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # u6: A[0,1] coeff = 1, A[1,1] coeff = -1 -> U[1,6]=1, U[3,6]=-1
    # v6: B[1,0] coeff = 1, B[1,1] coeff = 1 -> V[2,6]=1, V[3,6]=1
    # Contributes to C[0,0] (+1), C[0,1] (0), C[1,0] (0), C[1,1] (0)
    # So w6: W[0,6]=1 (C[0,0])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen 2x2x2 decomposition."""
    U, V, W = _strassen_2x2x2()
    T = _make_tensor(2, 2, 2)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=1e-9):
        print("Strassen verification failed!")
        print("T_rec:", T_rec)
        print("T:", T)
        return False
    return True

def _try_strassen_block(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by partitioning into blocks
    and using Strassen-like 7-product base case where applicable.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    T = _make_tensor(n, m, p)
    
    # For 4x5x5, try splitting into blocks where we can use Strassen
    # Split A (4x5) into 2x2 blocks vertically: top 2x5, bottom 2x5
    # Split B (5x5) into 2x2 blocks: need 5 = 2+3 or other splits
    
    # This is complex. Let's try a simpler approach:
    # Use the fact that we can embed smaller Strassen decompositions
    
    # For now, return None to fall back to schoolbook
    return None

def construct():
    deadline = time.time() + 120 - 10  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook decomposition
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
    
    best_result = {'U': U, 'V': V, 'W': W}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decomposition
    result = _try_strassen_block(n, m, p, deadline)
    if result is not None:
        U_try = np.array(result['U'])
        rank_try = U_try.shape[1]
        if rank_try < best_rank:
            # Verify
            T = _make_tensor(n, m, p)
            if _verify_decomposition(U_try, np.array(result['V']), np.array(result['W']), T):
                best_result = result
                best_rank = rank_try
    
    return best_result
# EVOLVE-BLOCK-END
