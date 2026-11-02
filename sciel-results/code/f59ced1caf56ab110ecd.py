"""Matrix multiplication tensor rank minimization for 3x4 * 4x8."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 135.0  # safety margin for 150s kill


def _time_check():
    """Return True if we should stop searching."""
    return time.time() - START_TIME >= TIME_LIMIT


def _tensor_product_decomp(U1, V1, W1, U2, V2, W2):
    """
    Compute the tensor product of two decompositions.
    If T1 has rank r1 with factors (U1, V1, W1) and T2 has rank r2 with factors (U2, V2, W2),
    then T1 ⊗ T2 has rank r1*r2.
    
    For matrix multiplication tensors:
    M_{a,b,c} ⊗ M_{d,e,f} = M_{ad, be, cf}
    
    U_combined[i1*a2 + i2, c1*r2 + c2] = U1[i1, c1] * U2[i2, c2]
    V_combined[j1*b2 + j2, c1*r2 + c2] = V1[j1, c1] * V2[j2, c2]
    W_combined[k1*c2 + k2, c1*r2 + c2] = W1[k1, c1] * W2[k2, c2]
    
    Wait, the indexing needs to be consistent with the trace convention W[k*n+i, r].
    For M_{n,m,p}: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    n1, m1, p1 = U1.shape[0] // m1 if False else (None, None, None)
    
    # We need to know the dimensions. Let's pass them explicitly.
    # Actually, let's infer from shapes:
    # U1 is (n1*m1, r1), V1 is (m1*p1, r1), W1 is (p1*n1, r1)
    # U2 is (n2*m2, r2), V2 is (m2*p2, r2), W2 is (p2*n2, r2)
    
    # This function is too complex to infer. Let me write specific constructors.
    raise NotImplementedError("Use specific constructors instead")


def _decomp_1x1x2():
    """
    Decomposition for 1x1 * 1x2 -> 1x2.
    T[i,j,k] = 1 if (i,j,k) is a valid multiplication entry.
    For 1x1 * 1x2: i=0, j∈{0,1}, k∈{0,1}
    T[0,j,k] = 1 if j==k (diagonal), else 0.
    
    Rank 2 decomposition:
    U shape (1*1, 2) = (1, 2): [[1, 1]]
    V shape (1*2, 2) = (2, 2): [[1, 0], [0, 1]]
    W shape (2*1, 2) = (2, 2): [[1, 0], [0, 1]]
    
    Check: T[0,j,k] = sum_c U[0,c] * V[j,c] * W[k,c]
    c=0: 1 * V[j,0] * W[k,0]
    c=1: 1 * V[j,1] * W[k,1]
    
    V = [[1, 0], [0, 1]], W = [[1, 0], [0, 1]]
    T[0,0,0] = 1*1*1 + 1*0*0 = 1 ✓
    T[0,0,1] = 1*1*0 + 1*0*1 = 0 ✓
    T[0,1,0] = 1*0*1 + 1*1*0 = 0 ✓
    T[0,1,1] = 1*0*0 + 1*1*1 = 1 ✓
    """
    U = np.array([[1, 1]], dtype=object)
    V = np.array([[1, 0], [0, 1]], dtype=object)
    W = np.array([[1, 0], [0, 1]], dtype=object)
    return U, V, W, 2


def _decomp_2x2x2_strassen():
    """
    Strassen's rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    
    Strassen's 7 products:
    m1 = (a+b)(e+f)
    m2 = (c+d)(e+f)
    m3 = a(h-f)
    m4 = d(g-e)
    m5 = (a+e)(h+f)
    m6 = (c+a)(f-h)
    m7 = (b+d)(g-c)
    
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m5 - m1 + m2 + m6
    C[1,0] = m1 - m3 + m4 + m6
    C[1,1] = -m3 + m2 + m5 + m7
    
    For the tensor decomposition, we need:
    T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all valid (i,j,k)
    
    Wait, this is the bilinear complexity formulation. The tensor T for matrix mult
    has T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    Actually no. Let me reconsider. The matrix multiplication tensor M satisfies:
    C_{i,k} = sum_{j} M_{i,j,k} * A_{i,j} * B_{j,k}
    where M_{i,j,k} = 1 for all valid (i,j,k).
    
    So for 2x2*2x2, T[i,j,k] = 1 for all i,j,k ∈ {0,1}.
    
    Strassen gives us a rank-7 decomposition where:
    C[i,k] = sum_{c=1}^{7} (linear combo of A) * (linear combo of B) * (linear combo for C)
    
    In the tensor factor form:
    T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    For Strassen, the "multiplication" step c corresponds to:
    - U factors: how A[i,j] contributes to the c-th product
    - V factors: how B[j,k] contributes to the c-th product  
    - W factors: how the c-th product contributes to C[i,k]
    
    Let me work this out carefully.
    
    For Strassen, the 7 products are:
    m1 = (a+b)(e+f): A factors: a→1, b→1, c→0, d→0; B factors: e→1, f→1, g→0, h→0
    m2 = (c+d)(e+f): A factors: a→0, b→0, c→1, d→1; B factors: e→1, f→1, g→0, h→0
    m3 = a(h-f): A factors: a→1, b→0, c→0, d→0; B factors: e→0, f→-1, g→0, h→1
    m4 = d(g-e): A factors: a→0, b→0, c→0, d→1; B factors: e→-1, f→0, g→1, h→0
    m5 = (a+e)(h+f): A factors: a→1, b→0, c→0, d→0; B factors: e→1, f→1, g→0, h→1
    m6 = (c+a)(f-h): A factors: a→1, b→0, c→1, d→0; B factors: e→0, f→1, g→0, h→-1
    m7 = (b+d)(g-c): A factors: a→0, b→1, c→-1, d→1; B factors: e→0, f→0, g→1, h→0
    
    And the output:
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m5 - m1 + m2 + m6
    C[1,0] = m1 - m3 + m4 + m6
    C[1,1] = -m3 + m2 + m5 + m7
    
    So for the tensor T[i,j,k] = 1:
    U[i*2+j, c] = coefficient of A[i,j] in the c-th product
    V[j*2+k, c] = coefficient of B[j,k] in the c-th product
    W[k*2+i, c] = coefficient of the c-th product in C[i,k]
    
    Wait, I need to be careful with the W indexing. W[k*n+i, c] where n=2.
    So W[k*2+i, c] is the coefficient of product c in C[i,k].
    
    Let me build the matrices:
    
    U (shape 4x7): rows indexed by i*2+j
    Row 0 (i=0,j=0, a): [1, 0, 1, 0, 1, 1, 0]
    Row 1 (i=0,j=1, b): [1, 0, 0, 0, 0, 0, 1]
    Row 2 (i=1,j=0, c): [0, 1, 0, 0, 0, 1, -1]
    Row 3 (i=1,j=1, d): [0, 1, 0, 1, 0, 0, 1]
    
    V (shape 4x7): rows indexed by j*2+k
    Row 0 (j=0,k=0, e): [1, 1, 0, -1, 1, 0, 0]
    Row 1 (j=0,k=1, f): [1, 1, -1, 0, 1, 1, 0]
    Row 2 (j=1,k=0, g): [0, 0, 0, 1, 0, 0, 1]
    Row 3 (j=1,k=1, h): [0, 0, 1, 0, 1, -1, 0]
    
    W (shape 4x7): rows indexed by k*2+i
    Row 0 (k=0,i=0, C[0,0]): [1, 0, 0, 1, -1, 0, 1]
    Row 1 (k=1,i=0, C[0,1]): [-1, 1, 0, 0, 1, 1, 0]
    Row 2 (k=0,i=1, C[1,0]): [1, 0, -1, 1, 0, 1, 0]
    Row 3 (k=1,i=1, C[1,1]): [0, 1, -1, 0, 1, 0, 1]
    
    Let me verify T[0,0,0] = sum_c U[0,c]*V[0,c]*W[0,c]
    = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*1*(-1) + 1*0*0 + 0*0*1
    = 1 + 0 + 0 + 0 - 1 + 0 + 0 = 0
    
    That's wrong! T[0,0,0] should be 1.
    
    I think I'm confusing the formulation. Let me reconsider.
    
    The issue is that in the tensor decomposition T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c],
    the U, V, W are NOT the same as the linear combinations in Strassen.
    
    In Strassen, we have:
    C[i,k] = sum_c (sum_j A[i,j] * u_{i,j,c}) * (sum_j B[j,k] * v_{j,k,c}) * w_{i,k,c}
    
    But this is NOT the same as:
    C[i,k] = sum_j A[i,j] * B[j,k] * (sum_c u_{i,j,c} * v_{j,k,c} * w_{i,k,c})
    
    Because in Strassen, the sum over j is INSIDE the product for each c, not outside.
    
    So the Strassen decomposition is NOT directly a tensor decomposition in the form
    T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c].
    
    Strassen works by grouping the sums: it computes products of LINEAR COMBINATIONS of A and B,
    then combines those products linearly to get C. This is a different structure.
    
    For a direct tensor decomposition, we need T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    where each term is a product of individual A[i,j], B[j,k], and a scalar for C[i,k].
    
    This means Strassen's rank-7 does NOT directly give us a rank-7 tensor decomposition
    in the required form. The rank-7 is for the BILINEAR complexity, which allows
    linear combinations before multiplication.
    
    Hmm, this is a crucial distinction. Let me reconsider the problem.
    
    Actually, I think I was wrong. Let me re-examine.
    
    The bilinear complexity of matrix multiplication IS the tensor rank. The tensor
    T[i,j,k] represents the coefficient in C[i,k] = sum_{j,c} T[i,j,k]^{(c)} A[i,j] B[j,k].
    
    Wait no. The standard definition: the matrix multiplication tensor M ∈ A* ⊗ B* ⊗ C*
    where M = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k} (using appropriate basis).
    
    The rank of this tensor is the bilinear complexity.
    
    In the factor form: M = sum_c u_c ⊗ v_c ⊗ w_c where u_c ∈ A*, v_c ∈ B*, w_c ∈ C*.
    
    This means: M(a,b) = sum_c u_c(a) * v_c(b) * w_c
    where a ∈ A, b ∈ B, and the result is in C.
    
    For matrix mult: M(A,B)[i,k] = sum_j A[i,j] B[j,k]
    
    So: sum_j A[i,j] B[j,k] = sum_c u_c(A) * v_c(B) * w_{i,k,c}
    
    where u_c(A) = sum_{i,j} U[i,j,c] A[i,j]
    v_c(B) = sum_{j,k} V[j,k,c] B[j,k]
    w_{i,k,c} is the coefficient.
    
    This is NOT the same as T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    
    In the Strassen case, we have:
    C[i,k] = sum_c (sum_{i',j} U[i',j,c] A[i',j]) * (sum_{j',k'} V[j',k',c] B[j',k']) * W[i,k,c]
    
    This is a different structure from what the problem asks for!
    
    The problem asks for: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    which means: C[i,k] = sum_j A[i,j] B[j,k] (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])
    = sum_j A[i,j] B[j,k] * T[i,j,k]
    
    So T[i,j,k] must equal 1 for all valid (i,j,k).
    
    This is the DIRECT tensor decomposition, not the Strassen-style bilinear decomposition.
    
    The Strassen rank-7 is for the bilinear complexity, which allows the sums over j to be
    distributed differently. In the direct tensor form, we need T[i,j,k] = 1 for all (i,j,k),
    and the rank is the number of terms in the decomposition.
    
    For the schoolbook: T[i,j,k] = 1 for all (i,j,k), and we use rank = n*m*p with
    U[i*m+j, c] = 1 if c corresponds to (i,j,k), else 0.
    
    So the question is: can we find a lower-rank decomposition where
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k?
    
    This is a much harder problem than Strassen's bilinear complexity.
    
    Let me go back to the random search approach from inspiration 1, which directly
    solves this constraint.
    """
    # This function is not used in the final approach.
    # The Strassen approach doesn't directly apply to the tensor rank formulation.
    raise NotImplementedError


def _decomp_1x2x2():
    """
    Decomposition for 1x2 * 2x2 -> 1x2.
    A is 1x2: [a1, a2]
    B is 2x2: [[b11, b12], [b21, b22]]
    C is 1x2: [c1, c2] where c1 = a1*b11 + a2*b21, c2 = a1*b12 + a2*b22
    
    T[i,j,k] = 1 for all valid (i,j,k): i=0, j∈{0,1}, k∈{0,1}
    So T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
    
    Rank 4 (schoolbook):
    U shape (1*2, 4) = (2, 4)
    V shape (2*2, 4) = (4, 4)
    W shape (2*1, 4) = (2, 4)
    
    c=0: (i=0,j=0,k=0): U[0,0]=1, V[0,0]=1, W[0,0]=1
    c=1: (i=0,j=0,k=1): U[0,1]=1, V[1,1]=1, W[1,1]=1
    c=2: (i=0,j=1,k=0): U[1,2]=1, V[2,2]=1, W[0,2]=1
    c=3: (i=0,j=1,k=1): U[1,3]=1, V[3,3]=1, W[1,3]=1
    
    Can we do better? The tensor is all-ones on a 1x2x2 array.
    This is equivalent to: for all j,k, sum_c U[j,c]*V[j*2+k,c]*W[k,c] = 1
    
    Let's try rank 3 or 2.
    """
    # Schoolbook rank 4
    U = np.zeros((2, 4), dtype=object)
    V = np.zeros((4, 4), dtype=object)
    W = np.zeros((2, 4), dtype=object)
    U[0, 0] = 1
    U[0, 1] = 1
    U[1, 2] = 1
    U[1, 3] = 1
    V[0, 0] = 1
    V[1, 1] = 1
    V[2, 2] = 1
    V[3, 3] = 1
    W[0, 0] = 1
    W[1, 1] = 1
    W[0, 2] = 1
    W[1, 3] = 1
    return U, V, W, 4


def _decomp_3x4x2():
    """
    Decomposition for 3x4 * 4x2 -> 3x2.
    n=3, m=4, p=2.
    T[i,j,k] = 1 for all i∈{0,1,2}, j∈{0,1,2,3}, k∈{0,1}.
    
    Schoolbook rank = 3*4*2 = 24.
    
    Let me try to find a lower rank using the random search approach.
    """
    n, m, p = 3, 4, 2
    U = np.zeros((n*m, n*m*p), dtype=object)
    V = np.zeros((m*p, n*m*p), dtype=object)
    W = np.zeros((p*n, n*m*p), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, n*m*p


def _validate_decomp(U, V, W, n, m, p):
    """Validate that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = s + U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True


def _try_random_decomp(n, m, p, target_rank, max_attempts=50, time_limit=None):
    """
    Try to find a decomposition of rank target_rank using random U, V and solving for W.
    """
    start = time.time()
    
    for attempt in range(max_attempts):
        if time_limit is not None and time.time() - start > time_limit:
            break
        if _time_check():
            break
        
        # Random small integers for U and V
        U = np.random.randint(-2, 3, size=(n*m, target_rank))
        V = np.random.randint(-2, 3, size=(m*p, target_rank))
        
        # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        M = np.zeros((n, p, target_rank), dtype=float)
        for i in range(n):
            for k in range(p):
                for c in range(target_rank):
                    M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
        
        # For each (k,i), we need: sum_c M[i,k,c] * W[k*n+i, c] = 1
        # This is a linear system: M[i,k,:] dot W[k*n+i, :] = 1
        W = np.zeros((p*n, target_rank), dtype=float)
        valid = True
        for i in range(n):
            for k in range(p):
                M_vec = M[i, k, :]
                norm_sq = np.dot(M_vec, M_vec)
                if norm_sq < 1e-10:
                    valid = False
                    break
                # W[k*n+i, :] = M_vec / norm_sq
                W[k*n+i, :] = M_vec / norm_sq
        if not valid:
            continue
        
        # Round to half-integers
        U_rounded = np.round(U.astype(float) * 2) / 2
        V_rounded = np.round(V.astype(float) * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify with exact arithmetic
        valid_exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s = s + U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid_exact = False
                        break
                if not valid_exact:
                    break
            if not valid_exact:
                break
        
        if valid_exact:
            # Convert to object array with Fraction for exactness
            U_obj = np.zeros((n*m, target_rank), dtype=object)
            V_obj = np.zeros((m*p, target_rank), dtype=object)
            W_obj = np.zeros((p*n, target_rank), dtype=object)
            for idx in range(n*m):
                for c in range(target_rank):
                    U_obj[idx, c] = Fraction(int(U_rounded[idx, c] * 2), 2)
            for idx in range(m*p):
                for c in range(target_rank):
                    V_obj[idx, c] = Fraction(int(V_rounded[idx, c] * 2), 2)
            for idx in range(p*n):
                for c in range(target_rank):
                    W_obj[idx, c] = Fraction(int(W_rounded[idx, c] * 2), 2)
            
            return U_obj, V_obj, W_obj, target_rank
    
    return None


def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (trivial) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = Fraction(1, 1)
                V[j*p+k, col] = Fraction(1, 1)
                W[k*n+i, col] = Fraction(1, 1)
                col += 1
    return U, V, W, rank


def _try_structured_decomp(n, m, p, target_rank, max_attempts=100, time_limit=None):
    """
    Try structured decompositions with specific patterns.
    """
    start = time.time()
    
    for attempt in range(max_attempts):
        if time_limit is not None and time.time() - start > time_limit:
            break
        if _time_check():
            break
        
        # Try different random seeds and patterns
        np.random.seed(attempt * 7 + 42)
        
        # Random U and V with small integers
        U = np.random.randint(-2, 3, size=(n*m, target_rank))
        V = np.random.randint(-2, 3, size=(m*p, target_rank))
        
        # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        M = np.zeros((n, p, target_rank), dtype=float)
        for i in range(n):
            for k in range(p):
                for c in range(target_rank):
                    M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
        
        # Solve for W
        W = np.zeros((p*n, target_rank), dtype=float)
        valid = True
        for i in range(n):
            for k in range(p):
                M_vec = M[i, k, :]
                norm_sq = np.dot(M_vec, M_vec)
                if norm_sq < 1e-10:
                    valid = False
                    break
                W[k*n+i, :] = M_vec / norm_sq
        if not valid:
            continue
        
        # Round to half-integers
        U_rounded = np.round(U.astype(float) * 2) / 2
        V_rounded = np.round(V.astype(float) * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        valid_exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s = s + U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid_exact = False
                        break
                if not valid_exact:
                    break
            if not valid_exact:
                break
        
        if valid_exact:
            U_obj = np.zeros((n*m, target_rank), dtype=object)
            V_obj = np.zeros((m*p, target_rank), dtype=object)
            W_obj = np.zeros((p*n, target_rank), dtype=object)
            for idx in range(n*m):
                for c in range(target_rank):
                    U_obj[idx, c] = Fraction(int(U_rounded[idx, c] * 2), 2)
            for idx in range(m*p):
                for c in range(target_rank):
                    V_obj[idx, c] = Fraction(int(V_rounded[idx, c] * 2), 2)
            for idx in range(p*n):
                for c in range(target_rank):
                    W_obj[idx, c] = Fraction(int(W_rounded[idx, c] * 2), 2)
            
            return U_obj, V_obj, W_obj, target_rank
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix mult."""
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 96
    
    # Start with schoolbook
    U_best, V_best, W_best, rank_best = _schoolbook_decomp(n, m, p)
    
    # Try to find lower rank decompositions
    # Strategy: Try ranks from 95 downward, but focus on promising ranges
    
    # First, try the random search approach for various ranks
    # Given time constraints, we'll try a limited number of ranks
    
    ranks_to_try = [95, 94, 93, 92, 91, 90, 88, 85, 80, 75, 70, 64, 60, 56, 52, 48, 42, 40, 36, 32, 28, 24, 20, 16, 12, 10, 8]
    
    for target_rank in ranks_to_try:
        if _time_check():
            break
        
        # Try random search
        result = _try_random_decomp(n, m, p, target_rank, max_attempts=30, time_limit=5.0)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            break
        
        # Try structured search
        result = _try_structured_decomp(n, m, p, target_rank, max_attempts=20, time_limit=5.0)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            break
    
    # Convert to lists for the return value
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


# Test the schoolbook decomposition
if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}")
    print(f"V shape: {V.shape}")
    print(f"W shape: {W.shape}")
    
    # Verify
    n, m, p = 3, 4, 8
    rank = U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    print(f"Invalid at ({i},{j},{k}): {s}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
