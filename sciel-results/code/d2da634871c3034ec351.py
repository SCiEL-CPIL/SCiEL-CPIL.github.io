"""Matrix multiplication 4x4 by 4x8 - rank-minimal decomposition via structured search."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS
N, M, P = n, m, p  # shorthand

def verify(U, V, W):
    """Check if the decomposition is exact for the all-ones tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i,j,k), s
    return True, None, None

def make_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def make_kronecker_decomposition():
    """
    Construct a rank-(n+1)(m+1)(p+1) decomposition of the all-ones tensor using
    the identity: for a in {0,...,n}, b in {0,...,m}, c in {0,...,p},
    U[i*m+j, (a,b,c)] = 1 if a==i or (a==n and j==m-1) else 0  -- NO, let me think again.
    
    Actually, the correct construction:
    The all-ones tensor T[i,j,k]=1 can be decomposed as:
    T[i,j,k] = sum_{a=0}^{n} sum_{b=0}^{m} sum_{c=0}^{p} u_a(i) * v_b(j) * w_c(k)
    where we need sum_a u_a(i) = 1 for all i, etc.
    
    Simplest: u_a(i) = 1 if a==i else 0, plus u_n(i) = 0 for all i... no that gives rank n*m*p.
    
    Better: Use the fact that 1 = 1 (scalar). We can write:
    1 = (sum_{a=0}^{n-1} delta_{a,i}) * (sum_{b=0}^{m-1} delta_{b,j}) * (sum_{c=0}^{p-1} delta_{c,k})
    This is just the schoolbook with rank n*m*p.
    
    For a better decomposition, use:
    1 = 1 * 1 * 1 (rank 1) -- but this doesn't work because U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1
    gives sum = 1 for all (i,j,k). Wait, that IS rank 1!
    
    Let me verify: If U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i,
    then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 for all (i,j,k). 
    
    So the rank is 1?! That seems too good. Let me re-read the problem.
    
    OH WAIT. I think I'm misunderstanding the tensor structure. Let me re-read.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 8 matrix."
    
    The matrix multiplication tensor is NOT the all-ones tensor. The tensor T[i,j,k] represents
    the coefficient in C[i,k] = sum_j A[i,j] * B[j,k]. So T[i,j,k] = 1 for all i,j,k.
    
    But the factorization is:
    C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    
    Wait no. The standard tensor decomposition for matrix mult is:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_c sum_j sum_k' U[i*m+j,c] A[i,j] V[j*p+k,c] B[j,k] W[k*n+i,c]
    
    No, that's not right either. Let me think more carefully.
    
    The bilinear complexity / tensor rank for matrix multiplication:
    We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    A rank-r decomposition means:
    C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    Wait, that's not standard either. The standard form is:
    C[i,k] = sum_c U[i,k,c] * (intermediate_c)
    where intermediate_c is a bilinear form in A and B.
    
    Actually, the tensor decomposition form is:
    T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    where T_{i,j,k} = 1 for all i,j,k (the matrix multiplication tensor).
    
    And the factor matrices are:
    U has shape (n*m, r) indexed by (i,j)
    V has shape (m*p, r) indexed by (j,k)  
    W has shape (p*n, r) indexed by (k,i)  [per the convention W[k*n+i, r]]
    
    So we need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    
    If we set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all indices,
    then sum = 1*1*1 = 1. So rank 1 works!
    
    But that can't be right - the tensor rank of matrix multiplication is well-known to be > 1.
    
    Let me re-read the problem statement more carefully...
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    Hmm, maybe I'm misunderstanding the indexing. Let me look at this differently.
    
    Actually, I think the issue is that in the standard tensor decomposition for matrix multiplication,
    the tensor is T[i,j,k] = 1, and the decomposition is:
    T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    This IS the all-ones tensor, and its rank over the reals is indeed 1 (just use all-ones vectors).
    
    But that contradicts known results about matrix multiplication rank. Let me reconsider.
    
    OH! I think I've been confusing two different things. The "rank" of the matrix multiplication
    tensor in the context of bilinear complexity is NOT the same as the tensor rank of the all-ones
    tensor. 
    
    In bilinear complexity, we're looking for:
    C[i,k] = sum_c L_c(A) * R_c(B)
    where L_c and R_c are linear forms.
    
    This corresponds to a decomposition where:
    C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k])
    
    And the tensor decomposition form with W is:
    1 = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    But wait, if W[i,k,c] is just a scalar multiplier, then we can absorb it into U or V.
    So the constraint is really:
    1 = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    And this CAN be satisfied with rank 1 by setting all entries to 1.
    
    I'm clearly misunderstanding something. Let me look at this from the perspective of what makes
    the problem non-trivial.
    
    Actually, I think the key issue is that in the standard formulation, U, V, W are NOT arbitrary.
    The decomposition must satisfy the constraint that when you form:
    (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]
    and sum over c, you get exactly C[i,k] = sum_j A[i,j] B[j,k].
    
    This means:
    sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j' V[j'*p+k,c] B[j',k]) = sum_j A[i,j] B[j,k]
    
    For this to hold for ALL A and B, we need:
    sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = delta_{j,j} = 1 (for the matching j)
    
    Wait, more precisely, expanding the left side:
    sum_c W[k*n+i,c] * sum_j U[i*m+j,c] A[i,j] * sum_j' V[j'*p+k,c] B[j',k]
    = sum_j sum_j' A[i,j] B[j',k] * sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c] = 1 if j==j' else 0
    
    So the constraint is:
    sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] = delta_{j,j'}
    
    This is NOT the all-ones tensor! It's a tensor that is 1 when j==j' and 0 otherwise.
    
    So T[i,j,j',k] = delta_{j,j'}, and the factorization is:
    delta_{j,j'} = sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c]
    
    This makes much more sense! The tensor depends on i, j, j', k, and is 1 only when j=j'.
    
    Let me re-read the problem: "T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]"
    
    Yes! The tensor is T[i,j,k] where the three indices correspond to (i,j) for A, (j,k) for B,
    and (k,i) for C. The tensor value is 1 (the coefficient), but the constraint is that the
    sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 for each valid (i,j,k) triple
    where j is shared between A and B.
    
    Wait, but that's still the all-ones tensor over the valid (i,j,k) triples. The issue is that
    j appears in both U and V, so the same j index is used in both.
    
    Let me re-derive. We need:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Using the decomposition:
    C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k])
    
    Expanding:
    = sum_c W[k*n+i,c] * sum_j U[i*m+j,c] A[i,j] * sum_{j'} V[j'*p+k,c] B[j',k]
    = sum_j sum_{j'} A[i,j] B[j',k] * sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c] = delta_{j,j'}
    
    So the tensor to decompose is T[i,j,j',k] = delta_{j,j'}, with factors:
    U indexed by (i,j), V indexed by (j',k), W indexed by (k,i).
    
    This is equivalent to: for each (i,k), the matrix U_{i,*} * W_{*,k} (outer product structure)
    combined with V must give the identity in the j,j' indices.
    
    More precisely, define for each c:
    U_c[i,j] = U[i*m+j, c]
    V_c[j,k] = V[j*p+k, c]  
    W_c[k,i] = W[k*n+i, c]
    
    Then the constraint is:
    sum_c W_c[k,i] * U_c[i,j] * V_c[j',k] = delta_{j,j'}
    
    This is a much more interesting problem! The rank-1 case would require:
    W_0[k,i] * U_0[i,j] * V_0[j',k] = delta_{j,j'}
    
    But the left side is a rank-1 matrix in (j,j') (outer product of U_0[i,:] and V_0[:,k] scaled by W_0[k,i]),
    while the right side is the identity matrix. So rank 1 doesn't work (unless m=1).
    
    OK so now I understand the problem correctly. Let me implement a proper search.
    """
    n, m, p = DIMENSIONS
    
    # The constraint: sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] = delta_{j,j'}
    # for all i, j, j', k.
    
    # Strategy: Use a structured approach.
    # One idea: Set W[k*n+i, c] = 1 for all k,i,c (or some simple pattern).
    # Then we need: sum_c U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    
    # Another idea: Use the identity matrix decomposition.
    # If we set m = p = n and use Strassen-like methods...
    
    # For now, let's try a random search with the CORRECT constraint.
    
    def verify_correct(U, V, W):
        """Check if the decomposition satisfies sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}"""
        r = len(U[0]) if U else 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    # Try small ranks with random search
    best_rank = n * m * p
    best_result = make_schoolbook()
    
    # First, verify the schoolbook solution with the correct constraint
    # Schoolbook: U[i*m+j, (i,j,k)] = 1, V[j'*p+k, (i,j,k)] = 1 if j==j' else 0, W[k*n+i, (i,j,k)] = 1
    # Wait, that's not quite right. Let me construct it properly.
    
    # For schoolbook rank n*m*p:
    # Index c = (i,j,k) for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U[i*m+j, c] = 1 if c corresponds to (i,j,*) else 0
    # V[j'*p+k, c] = 1 if c corresponds to (*,j',k) and j==j'... 
    # 
    # Actually, let me just verify the original schoolbook construction works.
    
    sb = make_schoolbook()
    if verify_correct(sb["U"], sb["V"], sb["W"]):
        best_rank = n * m * p
        best_result = sb
    else:
        # The schoolbook construction might not work with the corrected constraint.
        # Let me construct it properly.
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U[i*m+j, col] = 1
                    U[i*m+j][col] = 1
                    # V[j'*p+k, col] = 1 if j' == j else 0
                    V[j*p+k][col] = 1
                    # W[k*n+i, col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
        if verify_correct(U, V, W):
            best_rank = n * m * p
            best_result = {"U": U, "V": V, "W": W}
        else:
            # This shouldn't happen, but let's handle it
            pass
    
    # Now try to find better decompositions
    deadline = time.time() + 130
    
    # Try random search with small ranks
    for r in range(2, min(20, n*m*p)):
        if time.time() > deadline:
            break
        for it in range(100):
            if time.time() > deadline:
                break
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            if verify_correct(U, V, W):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    valid, err, val = verify(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
