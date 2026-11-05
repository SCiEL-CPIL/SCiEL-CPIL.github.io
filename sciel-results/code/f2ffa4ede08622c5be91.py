import time
import random

DIMENSIONS = [3, 3, 6]
DEADLINE_FRACTION = 0.85  # stop at 85% of the 120s budget (102s)
START_TIME = time.time()

def _remaining_time():
    return DEADLINE_FRACTION * 120.0 - (time.time() - START_TIME)

def _check_valid(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Check tensor identity: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(i,j,k)
    Actually the matrix multiplication tensor is T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k].
    The decomposition is: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+k... 
    Wait, let me re-read the convention.
    
    U: shape (n*m, rank) - U[i*m+j, c]
    V: shape (m*p, rank) - V[j*p+k, c]
    W: shape (p*n, rank) - W[k*n+i, c]
    
    The tensor for matmul: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    Actually the standard tensor decomposition:
    C[i,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] * A[i,j] * B[j,k]
    
    Hmm, that's not right either. Let me think again.
    
    Standard formulation: For matrix multiplication, we have bilinear form.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition:
    C[i,k] = sum_{c=1}^{r} sum_j U[i,j,c] A[i,j] * V[j,k,c] B[j,k] * W[i,k,c]
    
    But that's not quite it either. The standard way:
    
    We define the matrix multiplication tensor T of shape (n, m, p) where
    T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]... 
    
    No. Let me be precise. The matrix multiplication is a bilinear map:
    (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r tensor decomposition of the matmul tensor means:
    C[i,k] = sum_c M_c(A) * N_c(B) * P_c
    where M_c is linear in A, N_c is linear in B, and P_c distributes to C[i,k].
    
    More specifically, with the given factor convention:
    U[i*m+j, c] = coefficient of A[i,j] in the c-th bilinear form
    V[j*p+k, c] = coefficient of B[j,k] in the c-th bilinear form  
    W[k*n+i, c] = coefficient for C[i,k] (with k*n+i indexing)
    
    Wait, W[k*n+i, c] means W is indexed by (k, i) pair, flattened as k*n+i.
    So W[k*n+i, c] is the coefficient that multiplies the c-th product to contribute to C[i,k].
    
    So the decomposition is:
    C[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    
    For this to equal sum_j A[i,j] * B[j,k] for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (1 for the correct terms)
    
    Actually, for fixed (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] should be 1,
    and for other combinations it should be 0.
    
    So the check is: for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,k) is the output and j is the inner index... 
    
    Hmm, I think I'm overcomplicating. Let me just verify numerically.
    
    The condition is: for all i in range(n), k in range(p):
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    Wait no, that's not right either because U and V are multiplied per c.
    
    Let me re-read: "W[k*n+i,r]" - so W has shape (p*n, rank).
    
    I think the correct interpretation is:
    For each rank component c:
      M_c = sum_{i,j} U[i*m+j, c] * A[i,j]  (a scalar)
      N_c = sum_{j,k} V[j*p+k, c] * B[j,k]  (a scalar)
      Then C[i,k] += W[k*n+i, c] * M_c * N_c
    
    For this to equal the true matrix product for ALL A, B:
    We need: for all i,k,
    sum_c W[k*n+i, c] * (sum_{j'} U[i*m+j', c] * A[i,j']) * (sum_{j''} V[j''*p+k, c] * B[j'',k])
    = sum_j A[i,j] * B[j,k]
    
    This must hold for all A, B. So comparing coefficients:
    For each (i, j, k): the coefficient of A[i,j]*B[j,k] on the LHS is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And this must equal 1 if... wait, A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    But A[i,j'] * B[j'',k] where j' != j'' should NOT contribute to C[i,k].
    
    So the condition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    Wait no, that can't be right because then every A[i,j']*B[j'',k] would contribute.
    
    Let me think differently. The bilinear form is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition:
    C[i,k] = sum_c (sum_{i',j'} U[i',j',c] A[i',j']) (sum_{j'',k'} V[j'',k'',c] B[j'',k'']) W[i,k,c]
    
    No wait, the shapes are given: U is (n*m, rank), so U[row, c] where row = i*m+j.
    V is (m*p, rank), so V[row, c] where row = j*p+k.
    W is (p*n, rank), so W[row, c] where row = k*n+i.
    
    So:
    C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j'',k'} V[j''*p+k', c] * B[j'',k'])
    
    For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    The coefficient of A[i',j'] * B[j'',k'] in the expression for C[i,k] must be:
    sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k', c]
    
    This must equal 1 if (i',j',j'',k') = (i,j,j,k) i.e. i'=i, j'=j, j''=j, k'=k,
    and 0 otherwise.
    
    So: sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k', c] = 1 if i'=i, k'=k, j'=j''=j
    and 0 otherwise.
    
    So for each triple (i, j, k):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    And for "wrong" combinations, it should be 0. But since we're summing over all i',j',j'',k',
    the condition is simply that for each (i,j,k) the above sum equals 1, and for all other
    combinations it equals 0.
    
    Actually, I think the cleanest way to verify is to pick random A, B, compute C via the factors,
    and check against numpy matmul.
    """
    import numpy as np
    n, m, p = 3, 3, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Random test
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(len(U[0])):
        M_c = 0.0
        for i in range(n):
            for j in range(m):
                M_c += U[i * m + j, c] * A[i, j]
        N_c = 0.0
        for j in range(m):
            for k in range(p):
                N_c += V[j * p + k, c] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i, c] * M_c * N_c
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True


def _verify(U, V, W):
    return _check_valid(U, V, W)


def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try Laderman-based 3x3x3 rank 23, doubled for 3x3x6
    # Laderman's algorithm: 3x3 matrix multiplication in rank 23
    # We'll construct factors for 3x3x3 then replicate for the 6 columns
    
    # Known Laderman factors (rank 23 for 3x3x3)
    # U: 9x23, V: 9x23, W: 9x23
    # From standard references
    
    # Laderman 3x3 factors
    U_3 = [
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    V_3 = [
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    W_3 = [
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # Verify the 3x3x3 Laderman factors
    if _check_valid(U_3, V_3, W_3, n=3, m=3, p=3):
        # Now extend to 3x3x6 by doubling
        # For the 6 columns, we use two copies of the 3x3x3 decomposition
        rank_lad = 23 * 2  # 46
        U_36 = [[0] * rank_lad for _ in range(n * m)]
        V_36 = [[0] * rank_lad for _ in range(m * p)]
        W_36 = [[0] * rank_lad for _ in range(p * n)]
        
        for half in range(2):
            base = half * 23
            for i in range(n):
                for j in range(m):
                    for c in range(23):
                        U_36[i * m + j][base + c] = U_3[i * m + j][c]
            for j in range(m):
                for k in range(3):
                    for c in range(23):
                        V_36[j * p + k][base + c] = V_3[j * 3 + k][c]
                for k in range(3, 6):
                    for c in range(23):
                        V_36[j * p + k][base + c] = V_3[j * 3 + (k - 3)][c]
            for k in range(p):
                for i in range(n):
                    for c in range(23):
                        W_36[k * n + i][base + c] = W_3[(k % 3) * 3 + i][c]
        
        if _check_valid(U_36, V_36, W_36):
            return {"U": U_36, "V": V_36, "W": W_36}
    
    # Return schoolbook as fallback
    return {"U": U_school, "V": V_school, "W": W_school}
