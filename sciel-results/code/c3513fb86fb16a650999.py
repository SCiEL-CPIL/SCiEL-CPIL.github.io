"""Matrix multiplication 2x2, 2x7 exact tensor decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes C = A * B.
    C[k*n+i, j*p... wait, let me re-read the convention.
    
    W[k*n+i, r] - so W has p*n rows, indexed by k (0..p-1) and i (0..n-1).
    U has n*m rows, V has m*p rows.
    
    The decomposition should satisfy:
    For the matrix multiplication tensor T[i,j,k] (where i indexes A rows, j indexes A cols/B rows, k indexes B cols/C cols):
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    But wait, for C = A*B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor is T[i,j,k] = A[i,j]*B[j,k] = C[i,k]
    And we need: C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    Let me verify with schoolbook: each term has exactly one nonzero in each of U,V,W
    U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    So sum over c: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] is 1 when c corresponds to (i,j,k), else 0.
    So sum_r = 1 for the matching (i,j,k) triple.
    Then sum_j of that gives sum_j delta_{(i,j,k)} which is 1 for each (i,j,k) where the indices match... 
    Actually it should give: for fixed i,k, sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    = sum_j [the r where all three are nonzero simultaneously]
    In schoolbook, for each (i,j,k), there's exactly one r. So the sum over j gives sum over j of 1 (for each j, there's one r matching i,j,k) = m.
    That's not right - it should be 1 for the correct product.
    
    Hmm, I think I'm confusing myself. Let me just check: the decomposition must satisfy
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ... no.
    
    Actually the matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor decomposition says: T = sum_r U_r ⊗ V_r ⊗ W_r where
    (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And we need this to equal the multiplication tensor, which has:
    T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k].
    So the "value" of T at (i,j,k) should be such that summing over j gives the product.
    
    Actually, I think the convention is that the tensor T[i,j,k] = 1 for all valid (i,j,k) in the sense that:
    C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    So T[i,j,k] = 1 for all i,j,k.
    
    Wait no. The standard matrix multiplication tensor is:
    T(i,j,k) = 1 if the indices are compatible, representing the multiplication A[i,j]*B[j,k].
    And the rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r where
    T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    
    In our indexing: u_r[i*m+j], v_r[j*p+k], w_r[k*n+i]
    
    For schoolbook: each term r corresponds to (i0,j0,k0) with:
    u_r[i*m+j] = 1 if i=i0 and j=j0, else 0
    v_r[j*p+k] = 1 if j=j0 and k=k0, else 0  
    w_r[k*n+i] = 1 if k=k0 and i=i0, else 0
    
    Then sum_r u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i] = 1 if (i,j,k)=(i0,j0,k0) for exactly one r, else 0.
    So T[i,j,k] = 1 for all (i,j,k) in range. That's correct!
    
    So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # The tensor has n*m*p = 28 entries, each must equal 1
    # We need rank r such that sum_{c=0}^{r-1} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # For small improvements, try random search
    target_ranks = [27, 26, 25]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values in {-1, 0, 1} or small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W}
    
    # Return best (schoolbook)
    return best
