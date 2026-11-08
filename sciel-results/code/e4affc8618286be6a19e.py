"""Find rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    deadline = time.time() + 120
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all (i,j,k) where the "wrong" combination, it should be 0.
    # Actually, the tensor T[i,j,k] for matrix mult is 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The constraint is: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor entries are: T[i,j,k,l] = delta_{appropriate}
    # In the trace convention W[k*n+i, r], the tensor is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this term contributes to C[i,k]
    # which is always the case for valid (i,j,k) - actually every (i,j,k) contributes.
    # Wait, let me reconsider. The matrix multiplication tensor has:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # and the decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # where W uses the trace convention W[k*n+i, r].
    
    # So we need: for all i in range(n), j in range(m), k in range(p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook rank 50
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # For small cases, we can try structured approaches
    
    # Approach: Try rank r and see if we can find a solution
    # The constraints are: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 50 equations in r^3 variables (with constraints)
    
    # Let's try a greedy/structured approach
    # For n=2, m=5, p=5: 
    # We have 10 output entries C[0,0..4] and C[1,0..4]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try rank 25: pair up the j dimension
    # For each (i,k), we need sum_j A[i,j]*B[j,k]
    # We can try to use the fact that B[j,k] for different j can be shared
    
    # Let's try a more systematic search for low rank
    # Use the fact that we can set up the problem as finding matrices
    
    # For now, let's try to verify if rank 50 is indeed minimal or if we can do better
    # by trying some structured lower-rank decompositions
    
    # Try rank 40, 35, 30, 25, etc. with a simple structured approach
    # For each candidate rank, try to construct a solution
    
    def try_rank(r):
        """Try to find a decomposition of rank r using structured approach."""
        if time.time() > deadline - 5:
            return None
        
        # For n=2, m=5, p=5, let's try a block structure
        # Idea: For each i, the computation C[i,:] = A[i,:] * B
        # is a (1x5) * (5x5) = (1x5) product
        # We can decompose this using rank-5 for each i, giving rank 10
        # But we can share factors between i=0 and i=1
        
        # Simple approach: use the fact that both rows share the same B
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # C[1,k] = sum_j A[1,j]*B[j,k]
        
        # Let's try: for each j and k, we have B[j,k] which is shared
        # U[0*m+j, r] and U[1*m+j, r] for the two rows
        # V[j*p+k, r] for the B entries
        # W[k*n+0, r] and W[k*n+1, r] for the two output rows
        
        # A natural decomposition: for each (j,k), use one rank-1 term
        # U[0*m+j, r] = A[0,j] (but A is variable, so this doesn't work)
        # The decomposition must be independent of the actual values
        
        # The correct formulation: we need constants such that
        # for any A and B: C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]... 
        # No wait, the tensor decomposition is:
        # C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * (A[i,j] * B[j,k])
        # But that's not right either. The bilinear map is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has components T[i,j,k] = 1 for all i,j,k
        # and we decompose as: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # So we need: for all i in [0,1], j in [0,4], k in [0,4]:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # This is a system of 50 equations. Let's try to solve it for low r.
        # For r=50, it's trivial. For lower r, we need to find solutions.
        
        # Let's try a structured solution for r=25:
        # Pair up consecutive j values
        # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + ... + A[0,4]*B[4,k]
        # We can group: (A[0,0]*B[0,k] + A[0,1]*B[1,k]) + (A[0,2]*B[2,k] + A[0,3]*B[3,k]) + A[0,4]*B[4,k]
        # Each pair needs 2 rank-1 terms (or 1 if we can factor)
        # A[0,0]*B[0,k] + A[0,1]*B[1,k] cannot be factored into 1 term generally
        
        # So the minimum for each pair is 2, giving 2*5 = 10 per row, 20 total
        # But we can share B[j,k] factors between rows!
        
        # Key insight: B[j,k] appears in both C[0,k] and C[1,k]
        # So we can have: for each (j,k), one term that contributes to both rows
        # Term r for (j,k): 
        #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
        #   V[j*p+k, r] = 1
        #   W[k*n+0, r] = 1, W[k*n+1, r] = 1
        # This gives: for C[0,k]: sum_j 1*1*1 = 5 (should be sum_j A[0,j]*B[j,k])
        # But this doesn't depend on A values, so it's wrong.
        
        # I'm confusing myself. Let me restart with the correct formulation.
        # The matrix multiplication as a bilinear map:
        # (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
        # The associated tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C
        # T[i,j | j,k | i,k] = 1 for all valid i,j,k
        # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # Constraint: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is correct. Now, for n=2,m=5,p=5, we have 50 constraints.
        # The schoolbook solution uses 50 terms. Can we do better?
        
        # Let's try to find a solution with r < 50 using linear algebra.
        # Fix W to be a basis (gauge fixing) and solve for U, V.
        
        # For a given r, we can try:
        # 1. Choose W arbitrarily (e.g., first r columns of identity if r <= p*n=10)
        # 2. For each column r, the constraint becomes:
        #    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        #    For fixed r, if W[k*n+i,r] != 0, this couples U and V.
        
        # This is still hard. Let's try a different approach: use known results.
        # For 2x5x5, the rank is known to be at most 50 (schoolbook).
        # Can we prove it's less? 
        
        # Let's just try to find any solution with r < 50 by search.
        # For small r, we can use the fact that the system is underdetermined.
        
        # Try r from 49 down to some minimum:
        for r in range(min(49, best_rank - 1), 10, -1):
            if time.time() > deadline - 10:
                break
            
            # Try to find a solution of rank r
            # Use a simple approach: start from schoolbook and try to merge terms
            
            # For now, let's just return the schoolbook solution if we can't find better
            # This is a placeholder for actual search
            
            pass
        
        return None
    
    # For this specific small case, let's try a direct construction
    # Known: for matrix multiplication (n,m,p), the rank is at most n*m*p
    # and at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    
    # So rank is between 25 and 50. Let's try to find rank 25 or close.
    
    # For rank 25: we need 25 terms to satisfy 50 equations.
    # This is possible if the equations are dependent.
    
    # Let's try a specific construction for rank 25:
    # Use the fact that for each k, the computation involves all j
    # C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1
    
    # Construction idea: for each k (5 values), use 5 rank-1 terms
    # Term (k, j): 
    #   U[0*m+j, r] = 1 if i=0, 0 otherwise... no, this doesn't work
    #   We need: sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Let me try: for each k, define 5 terms indexed by j:
    #   For term (k,j): r = k*5 + j
    #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #   V[j*p+k, r] = 1
    #   W[k*n+0, r] = 1, W[k*n+1, r] = 1
    #   W[k*n+i', r] = 0 for i' != 0,1 (but n=2, so only i=0,1)
    
    # Check: for fixed (i,j,k):
    #   sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    #   = U[i*m+j, k*5+j]*V[j*p+k, k*5+j]*W[k*n+i, k*5+j]
    #   = 1*1*1 = 1 (if i is 0 or 1, which it is)
    #   But we also need to check that other terms don't contribute
    #   For term (k',j') with k'!=k or j'!=j:
    #     If k'!=k: V[j*p+k, k'*5+j'] = 0 (since V[j*p+k, r] is only nonzero when r corresponds to (k,j))
    #     Wait, V[j*p+k, r] where r = k'*5+j', we need V[j*p+k, k'*5+j'] = 0 unless k'=k and j'=j
    #     But in my construction, V[j*p+k, k*5+j] = 1, and V[j*p+k, k'*5+j'] = 0 for (k',j') != (k,j)
    #     So yes, only the term (k,j) contributes.
    #   If j'!=j: U[i*m+j, k*5+j'] = 0 (since U[i*m+j, r] is only nonzero when r corresponds to j)
    #     In my construction, U[i*m+j, k*5+j] = 1, U[i*m+j, k*5+j'] = 0 for j'!=j
    #     So yes, only the term (k,j) contributes.
    
    # So this gives a valid decomposition of rank 25!
    
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for k in range(p):
        for j in range(m):
            r = k * m + j
            # U[i*m+j, r] = 1 for i=0,1
            U[0*m+j, r] = 1
            U[1*m+j, r] = 1
            # V[j*p+k, r] = 1
            V[j*p+k, r] = 1
            # W[k*n+i, r] = 1 for i=0,1
            W[k*n+0, r] = 1
            W[k*n+1, r] = 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if verify(U, V, W):
        best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
        best_rank = rank
    else:
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
        best_rank = rank_schoolbook
    
    return {"U": best_U, "V": best_V, "W": best_W}
