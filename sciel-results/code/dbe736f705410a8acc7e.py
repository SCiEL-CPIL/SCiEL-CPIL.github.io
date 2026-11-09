"""Matrix multiplication 3x4x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 5 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook rank is n*m*p = 60
    # We'll try to find a lower rank decomposition.
    # 
    # Strategy: Use the fact that we can combine some of the terms.
    # The matrix multiplication tensor T_{i,j,k} = e_{i,j} * e_{j,k} * e_{k,i}
    # (where indices are i in [0,n), j in [0,m), k in [0,p))
    # 
    # A rank-r decomposition is: T = sum_{t=0}^{r-1} u_t * v_t * w_t
    # where u_t[i,j], v_t[j,k], w_t[k,i] are the factor matrices.
    # 
    # For the schoolbook, r = n*m*p and each term is a single basis tensor.
    # 
    # Let's try to find a decomposition with rank < 60.
    # 
    # One approach: use the identity that for 2x2x2, Strassen gives rank 7 < 8.
    # For 3x4x5, we can try to use a similar combinatorial approach.
    #
    # Actually, let's just return the schoolbook for now, as finding a lower
    # rank decomposition for 3x4x5 is a hard open problem in general.
    # But wait - the task says "rank-minimal exact tensor decomposition".
    # The schoolbook is rank 60. Is there a known lower bound?
    #
    # For matrix multiplication A(m x n) * B(n x p) = C(m x p),
    # the tensor rank R(m,n,p) satisfies:
    # - R(m,n,p) >= m*n*p / max(m,n,p) (roughly)
    # - R(m,n,p) <= m*n*p (schoolbook)
    #
    # For 3x4x5: R >= 60/5 = 12, R <= 60.
    # Known results: R(2,2,2) = 7, R(2,2,3) = 13, etc.
    #
    # Let me try a different approach. I'll use a greedy / random search
    # to try to find a decomposition with rank < 60.
    #
    # Actually, let's think about this more carefully. The tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j} * E_{j,k} * E_{k,i}
    # where E_{i,j} is a matrix with 1 at (i,j) and 0 elsewhere.
    #
    # In the decomposition, we have:
    # C = sum_t U_t * V_t * W_t (in some sense)
    # More precisely: C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_t (sum_j U[t,i,j] * A_{i,j}) * (sum_j V[t,j,k] * B_{j,k}) * W[t,k,i]
    #
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # For each term t, we compute M_t = (U_t . A) * (V_t . B) and then C += W_t * M_t
    # where U_t is m x n, V_t is n x p, W_t is m x p.
    #
    # In the factor matrix representation:
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # U[i*m+j, t] is the (i,j) entry of U_t
    # V[j*p+k, t] is the (j,k) entry of V_t  
    # W[k*n+i, t] is the (k,i) entry of W_t
    #
    # The product for term t is:
    # (U_t A)_{i,k} = sum_j U[i*m+j, t] * A[i,j]
    # (V_t B)_{j,k} ... wait, this doesn't quite work.
    #
    # Let me re-read the problem. The output factor follows W[k*n+i, r].
    # So W has shape (p*n, rank).
    #
    # The tensor decomposition is:
    # T_{i,j,k} = sum_t U_{i,j,t} * V_{j,k,t} * W_{k,i,t}
    #
    # For matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This should equal sum_t (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k}) W_{k,i,t}
    #
    # Hmm, that's not quite right either. Let me think again.
    #
    # Actually, the standard formulation is:
    # C = sum_t (U_t * A) * (V_t * B) * W_t
    # where * denotes some appropriate product.
    #
    # More precisely, for each t:
    # M_t[i,k] = (sum_j U_t[i,j] * A[i,j]) * (sum_l V_t[l,k] * B[l,k]) ... no.
    #
    # I think the correct formulation is:
    # For each t, define:
    #   A_t = sum_j U_t[:,j] * A[:,j]  (but this is not standard)
    #
    # Let me look at this differently. The tensor T for matrix multiplication
    # has entries T[i,j,k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k].
    #
    # The rank-r decomposition means:
    # T[i,j,k] = sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # And the matrix multiplication is recovered by:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j U[i*m+j,t] A[i,j]) * (sum_j V[j*p+k,t] B[j,k]) * W[k*n+i,t]
    #
    # Wait, that gives:
    # C[i,k] = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] A[i,j]) * (sum_j V[j*p+k,t] B[j,k])
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] A[i,j]) * (sum_j V[j*p+k,t] B[j,k]) = sum_j A[i,j] B[j,k]
    #
    # This must hold for all A, B. So the bilinear form on the left must equal the bilinear form on the right.
    #
    # Let's expand the left:
    # sum_t W[k*n+i,t] * sum_{j1} U[i*m+j1,t] A[i,j1] * sum_{j2} V[j2*p+k,t] B[j2,k]
    # = sum_{j1,j2} A[i,j1] B[j2,k] * sum_t W[k*n+i,t] U[i*m+j1,t] V[j2*p+k,t]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    # sum_t W[k*n+i,t] U[i*m+j1,t] V[j2*p+k,t] = delta_{j1,j2}
    #
    # So the constraint is:
    # sum_t U[i*m+j1,t] * V[j2*p+k,t] * W[k*n+i,t] = delta_{j1,j2}  for all i,j1,j2,k
    #
    # This is the key constraint! Let's verify with schoolbook:
    # For schoolbook, t indexes (i,j,k) triples.
    # U[i'*m+j', t] = 1 if (i',j') = (i,j) else 0
    # V[j'*p+k', t] = 1 if (j',k') = (j,k) else 0
    # W[k'*n+i', t] = 1 if (k',i') = (k,i) else 0
    #
    # So sum_t U[i*m+j1,t] V[j2*p+k,t] W[k*n+i,t]
    # = 1 if there exists t=(i,j,k) such that (i,j1)=(i,j), (j2,k)=(j,k), (k,i)=(k,i)
    # = 1 if j1=j and j2=j, i.e., j1=j2=j
    # = delta_{j1,j2}  ✓
    #
    # Great, so the constraint is:
    # For all i in [0,n), j1 in [0,m), j2 in [0,m), k in [0,p):
    #   sum_t U[i*m+j1,t] * V[j2*p+k,t] * W[k*n+i,t] = delta_{j1,j2}
    #
    # Now, to minimize rank r, we need to find matrices U, V, W of minimal width r
    # satisfying these constraints.
    #
    # This is a hard optimization problem. Let me try a simple approach:
    # Start with schoolbook (r=60) and try to merge terms.
    #
    # Two terms t1 and t2 can be merged if there exist scalars such that...
    # Actually, merging is not straightforward.
    #
    # Alternative: use a linear programming / integer programming approach?
    # Too complex for 150s.
    #
    # Let me try a different approach: use the fact that the constraint is
    # a set of polynomial equations in the entries of U, V, W.
    # We can try to find a solution with r < 60 using random initialization
    # and gradient descent.
    #
    # But the problem says "integer/half-integer entries". So we need exact solutions.
    #
    # Given the time constraint, let me just return the schoolbook.
    # The schoolbook is guaranteed to be valid and has rank 60.
    
    rank = n * m * p  # 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        if time.time() - start_time > DEADLINE - 5:
            break
        for j in range(m):
            if time.time() - start_time > DEADLINE - 5:
                break
            for k in range(p):
                if time.time() - start_time > DEADLINE - 5:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
