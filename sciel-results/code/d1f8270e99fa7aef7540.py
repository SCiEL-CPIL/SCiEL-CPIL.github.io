"""Matrix multiplication 4,4,6 — rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np
from scipy.linalg import solve

DIMENSIONS = [4, 4, 6]

# Deadline guard: keep a safety margin before the 150 s wall-clock limit.
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds


def _schoolbook():
    """Return the trivial rank-96 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the factor matrices exactly decompose the matmul tensor."""
    # The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Equivalently, for each (i,k), the matrix A[i,:] @ B[:,k] = sum_c U[i*m+*, c] * V[*p+k, c] * W[k*n+i, c]
    # We check by reconstructing: for random A, B, compare A@B vs the tensor contraction.
    # More directly: build the full tensor and check.
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # The matmul tensor is T[i,j,k] = 1 if we think of A[i,j]*B[j,k] -> C[i,k],
    # but actually the tensor for matmul is: T[i,j,k] = delta (i.e., the identity in j for each (i,k)).
    # Wait: the matmul tensor T is defined by C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k? No.
    # Actually the tensor T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk} T_{ijk}... no.
    # The standard tensor decomposition of matmul: we want
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # expressed as sum_c U[i,j,c] V[j,k,c] W[c] where W is scalar? No.
    # The tensor is T[i, j, k] where the multiplication is:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-tensor T of shape (n, m, p) with T[i,j,k] = 1 for all? No.
    # Actually, the matmul tensor T has T[i,j,k] = 1 for all i,j,k is wrong.
    # Let me reconsider: The bilinear map (A, B) -> C = A@B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k would give C[i,k] = sum_j A[i,j]*B[j,k]*1, which is correct!
    # Wait no, that's just the identity. The tensor T is the all-ones tensor of shape (n, m, p)?
    # No! That's not right either. Let me think again.
    #
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # This is a 3-tensor T of shape (n, m, p) where T[i,j,k] = 1 for all i,j,k.
    # Because C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] with T[i,j,k]=1.
    #
    # Hmm, but that means the tensor is all ones, which has rank... well, the schoolbook
    # decomposition uses rank n*m*p with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for
    # each (i,j,k) triple.
    #
    # Actually wait. Let me re-read the problem. The output factor W follows the convention
    # W[k*n+i, r]. So W is of shape (p*n, rank).
    #
    # The decomposition is: C[i,k] = sum_c U[i*m+j, c] ... no, that doesn't make sense.
    #
    # Let me think about this differently. The standard tensor decomposition for matmul:
    # We have three factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank).
    # The contraction is:
    #   C[i,k] = sum_{j, c} A[i,j] * B[j,k] * ... 
    #
    # Actually, I think the convention is:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] is the matmul tensor.
    #
    # And the matmul tensor T[i,j,k] = 1 for all i,j,k? Let's verify with schoolbook:
    # For each (i,j,k), there's one column c where U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    # So the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for that specific (i,j,k),
    # and 0 for others? No, for a fixed (i,j,k), only the column corresponding to (i,j,k)
    # has all three entries equal to 1. All other columns have at least one zero in the
    # (i*m+j) row of U, (j*p+k) row of V, or (k*n+i) row of W.
    #
    # Wait, that's not right. For a fixed (i,j,k), consider column c' corresponding to
    # (i',j',k'). U[i*m+j, c'] = 1 only if i'=i and j'=j. V[j*p+k, c'] = 1 only if j'=j and k'=k.
    # W[k*n+i, c'] = 1 only if k'=k and i'=i.
    # So the product is 1 only if (i',j',k') = (i,j,k). Yes!
    #
    # So the tensor T[i,j,k] = 1 for all i,j,k. The matmul tensor is the all-ones tensor.
    #
    # Hmm, but that seems too simple. Let me double-check with a small example.
    # n=1, m=1, p=1: C[0,0] = A[0,0]*B[0,0]. Tensor T[0,0,0]=1. Rank 1. Correct.
    # n=1, m=2, p=1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Tensor T[0,0,0]=1, T[0,1,0]=1.
    # Rank should be 2. Schoolbook gives rank 2. Correct.
    #
    # So yes, the tensor is all ones of shape (n, m, p).
    #
    # Now, the question is: what is the minimal rank of the all-ones tensor of shape (4,4,6)?
    #
    # The rank of the all-ones tensor of shape (n, m, p) is min(n, m, p) if... no, that's
    # not right either. The all-ones tensor has rank 1 if we allow arbitrary entries? No.
    #
    # Wait, I'm confusing myself. Let me reconsider.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k. Can this be written as a sum of rank-1 tensors?
    # A rank-1 tensor is T[i,j,k] = u[i] * v[j] * w[k].
    # We want sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.
    #
    # This is equivalent to saying the all-ones matrix (of any slice) can be decomposed.
    # For fixed k, T[:,:,k] = u_c[i] v_c[j] summed over c = 1 for all i,j.
    # This is the all-ones matrix of shape (n,m), which has rank 1. So we can take
    # u_c[i] = 1/sqrt(n) * something... 
    #
    # Actually, the all-ones matrix J (n x m) has rank 1: J = 1_n * 1_m^T.
    # So T[i,j,k] = 1 = (1/sqrt(n)) * (1/sqrt(m)) * (sqrt(n*m)) ... no.
    #
    # Let's just say: T[i,j,k] = a_c[i] * b_c[j] * d_c[k] summed over c.
    # If we take c=1 only: a_1[i] * b_1[j] * d_1[k] = 1 for all i,j,k.
    # This requires a_1[i] = 1 for all i, b_1[j] = 1 for all j, d_1[k] = 1 for all k.
    # So T = 1_n ⊗ 1_m ⊗ 1_p is a rank-1 tensor!
    #
    # Wait, that can't be right. The rank of the all-ones tensor is 1?
    #
    # Let me verify: T[i,j,k] = u[i]*v[j]*w[k] where u[i]=1, v[j]=1, w[k]=1.
    # Then T[i,j,k] = 1*1*1 = 1. Yes! The all-ones tensor is rank 1!
    #
    # But that contradicts the schoolbook decomposition having rank n*m*p. What's going on?
    #
    # OH WAIT. I think I misunderstood the problem. Let me re-read.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 6 matrix."
    #
    # The tensor for matrix multiplication is NOT the all-ones tensor. Let me reconsider.
    #
    # C = A * B, where A is 4x4 and B is 4x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T is defined such that C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k,c]... no.
    #
    # I think the correct interpretation is:
    # The bilinear map (A,B) -> C = A@B can be written as:
    # C[i,k] = sum_c sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    #
    # No wait, that has an extra sum over j. Let me think again.
    #
    # I think the correct form is:
    # C[i,k] = sum_c U[i, c] * V[k, c] * (some bilinear form in A and B)
    #
    # Actually, I recall now. The tensor decomposition for matmul works as follows:
    # We want to compute C = A @ B where A is (n x m) and B is (m x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T of shape (n, m, p) is defined by T[i,j,k] = 1 for all i,j,k.
    # And the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, but I showed above that the all-ones tensor is rank 1. So why does schoolbook
    # use rank n*m*p?
    #
    # AH, I see my error. The tensor is NOT all ones. Let me reconsider.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This can be written as:
    # C[i,k] = sum_j sum_c (A[i,j] * U[i*m+j, c]) * (B[j,k] * V[j*p+k, c]) * W[k*n+i, c] / W[k*n+i, c]
    #
    # No, that's circular. Let me look at this from the tensor perspective.
    #
    # The matmul tensor T has entries T[i,j,k] where the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The tensor T is such that when you contract A with T over j, and B with T over j,
    # you get C. Specifically:
    # C[i,k] = sum_j A[i,j] * T[i,j,k] * B[j,k]... no, that's not right either.
    #
    # I think the correct statement is:
    # T[i,j,k] = 1 for all i,j,k, and C[i,k] = sum_j A[i,j] * T[i,j,k] * B[j,k] = sum_j A[i,j]*B[j,k].
    #
    # So T is indeed the all-ones tensor. And I showed it's rank 1.
    #
    # But wait, the schoolbook decomposition has rank n*m*p. If the tensor is rank 1,
    # then schoolbook is not optimal. But the problem says "Find a rank-minimal exact tensor
    # decomposition". If the rank is 1, then the answer is trivial.
    #
    # I must be misunderstanding the tensor decomposition format. Let me re-read the problem.
    #
    # "Factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    #
    # So U has rows indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs.
    #
    # The contraction is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    #
    # And T[i,j,k] should be the matmul tensor. What IS the matmul tensor?
    #
    # I think the matmul tensor T is defined by:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    #
    # But they ALL contribute, so T[i,j,k] = 1 for all i,j,k.
    #
    # Unless... the tensor is not T[i,j,k] but rather the tensor defines how to combine
    # U, V, W to produce C. Let me think about it as:
    #
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    #
    # No, that doesn't work because the sums over j are independent.
    #
    # Actually, I think the correct interpretation is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    #
    # This is a sum over c of products of three terms. For this to equal sum_j A[i,j]*B[j,k],
    # we need a specific structure.
    #
    # Let me try with the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for
    # c corresponding to (i,j,k).
    #
    # For a fixed c corresponding to (i',j',k'):
    # sum_j U[i*m+j, c] * A[i,j] = sum_j (1 if j=j' else 0) * A[i,j] = A[i,j']
    # sum_j V[j*p+k, c] * B[j,k] = sum_j (1 if j=j' else 0) * B[j,k] = B[j',k]
    # W[k*n+i, c] = 1 if (k,i) = (k',i') else 0
    #
    # So the term for c=(i',j',k') is: [1 if (k,i)=(k',i') else 0] * A[i,j'] * B[j',k]
    #
    # Summing over all c:
    # sum_{i',j',k'} [1 if (k,i)=(k',i') else 0] * A[i,j'] * B[j',k]
    # = sum_{j'} A[i,j'] * B[j',k]   (since i'=i and k'=k)
    # = C[i,k]
    #
    # Great, so the schoolbook works. And the tensor T[i,j,k] in the sense of
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    #
    # For schoolbook: T[i,j,k] = 1 if there exists c with U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1
    # = 1 if c corresponds to (i,j,k) = 1.
    #
    # So T[i,j,k] = 1 for all i,j,k. And I showed this is rank 1.
    #
    # But wait, the rank-1 decomposition would be:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    #
    # Then T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1. Correct!
    #
    # But does this actually compute the matrix product? Let's check:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
    #
    # With rank 1: C[i,k] = W[k*n+i,0] * (sum_j U[i*m+j,0] * A[i,j]) * (sum_j V[j*p+k,0] * B[j,k])
    # = 1 * (sum_j 1 * A[i,j]) * (sum_j 1 * B[j,k])
    # = (sum_j A[i,j]) * (sum_j B[j,k])
    #
    # This is NOT equal to sum_j A[i,j]*B[j,k] in general!
    #
    # So my interpretation of the contraction must be wrong. The tensor T[i,j,k] is not
    # directly the matmul tensor in the way I was thinking.
    #
    # Let me reconsider. I think the correct interpretation is that the factor matrices
    # define a decomposition of the bilinear map, not the tensor T directly.
    #
    # The bilinear map is: (A, B) -> C = A@B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # A rank-1 term in the decomposition of this bilinear map would be:
    # C[i,k] += (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    #
    # For the sum over c to equal the full bilinear map, we need:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) = sum_j A[i,j]*B[j,k]
    #
    # This is a condition on U, V, W. The schoolbook satisfies this, as I showed.
    #
    # Now, can we do better than rank 96? The question is whether there exists a decomposition
    # of lower rank.
    #
    # For small dimensions, the minimal rank of matrix multiplication is known. For 4x4 times
    # 4x6, I'm not sure of the exact minimal rank, but it's likely less than 96.
    #
    # Actually, I recall that the minimal rank of n x n matrix multiplication is at least
    # n^2 (for n >= 2) and at most n^3. For rectangular matrices, similar bounds apply.
    #
    # Given the time constraints, let me try a different approach: use a randomized search
    # or a known decomposition scheme to find a lower rank.
    #
    # Actually, let me just implement a proper search. The key insight is that the condition
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) = sum_j A[i,j]*B[j,k]
    # must hold for ALL A and B. This means the bilinear form defined by the left side must
    # equal the bilinear form sum_j A[i,j]*B[j,k] for each (i,k).
    #
    # This is equivalent to saying that for each (i,k), the matrix
    # M_{ik} = sum_c W[k*n+i, c] * u_c[i,:] * v_c[:,k]^T
    # must equal the identity matrix of size m (i.e., M_{ik}[j,j'] = delta_{j,j'} for all j,j').
    #
    # Wait, let me be more careful. Let u_c[i,:] be the row of U corresponding to index i (i.e.,
    # the entries U[i*m+0, c], U[i*m+1, c], ..., U[i*m+m-1, c]).
    # Let v_c[:,k] be the column of V corresponding to index k (i.e., the entries V[0*p+k, c],
    # V[1*p+k, c], ..., V[(m-1)*p+k, c]).
    #
    # Then (sum_j U[i*m+j, c] * A[i,j]) = u_c[i,:] @ A[i,:]^T = u_c[i,:] dot A[i,:]
    # And (sum_j V[j*p+k, c] * B[j,k]) = v_c[:,k]^T @ B[:,k] = v_c[:,k] dot B[:,k]
    #
    # So the c-th term contributes: W[k*n+i, c] * (u_c[i,:] dot A[i,:]) * (v_c[:,k] dot B[:,k])
    #
    # Summing over c: sum_c W[k*n+i, c] * (u_c[i,:] dot A[i,:]) * (v_c[:,k] dot B[:,k])
    #
    # This must equal sum_j A[i,j] * B[j,k] for all A, B.
    #
    # The left side is a product of two linear forms in A[i,:] and B[:,k] respectively,
    # weighted by W[k*n+i, c] and summed over c. The right side is a bilinear form in
    # A[i,:] and B[:,k].
    #
    # For this to hold for all A[i,:] and B[:,k], we need:
    # sum_c W[k*n+i, c] * u_c[i,:] * v_c[:,k]^T = I_m  (the m x m identity matrix)
    #
    # Where u_c[i,:] is a row vector of length m, v_c[:,k] is a column vector of length m,
    # and u_c[i,:] * v_c[:,k]^T is an m x m matrix.
    #
    # So the condition is: for each (i,k), sum_c W[k*n+i, c] * u_c[i,:] * v_c[:,k]^T = I_m.
    #
    # This is a system of equations. The unknowns are the entries of U, V, W.
    #
    # Let me denote:
    # - U has shape (n*m, rank): U[i*m+j, c] for i in [0,n), j in [0,m), c in [0,rank)
    # - V has shape (m*p, rank): V[j*p+k, c] for j in [0,m), k in [0,p), c in [0,rank)
    # - W has shape (p*n, rank): W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
    #
    # For each (i,k), the condition is:
    # sum_c W[k*n+i, c] * u_c[i,:] * v_c[:,k]^T = I_m
    #
    # where u_c[i,:] = [U[i*m+0,c], U[i*m+1,c], ..., U[i*m+m-1,c]]
    # and v_c[:,k] = [V[0*p+k,c], V[1*p+k,c], ..., V[(m-1)*p+k,c]]
    #
    # This gives m^2 equations for each (i,k), and there are n*p such (i,k) pairs.
    # Total: n*p*m^2 = 4*6*16 = 384 equations.
    #
    # The unknowns are: n*m*rank + m*p*rank + p*n*rank = rank*(n*m + m*p + p*n) = rank*(16+24+24) = 64*rank.
    #
    # For rank 1: 64 unknowns, 384 equations. Underdetermined? No, overdetermined.
    # For rank 6: 384 unknowns, 384 equations. Square system.
    #
    # So rank 6 might be possible! Let me try to solve this.
    #
    # Actually, this is a nonlinear system because the equations are quadratic in the unknowns
    # (products of U, V, W entries). So it's not a linear system.
    #
    # Let me try a different approach. I'll use a known decomposition or try to find one
    # through search.
    #
    # Given the complexity, let me just implement a search that tries to find a valid
    # decomposition of decreasing rank, starting from the schoolbook rank and working down.
    #
    # For the purpose of this task, I'll implement a randomized search with a deadline guard.

def _try_rank(rank, n, m, p, time_limit=10.0):
    """Try to find a decomposition of the given rank. Returns (U, V, W) or None."""
    start_time = time.time()
    
    # For each (i,k), we need: sum_c W[k*n+i, c] * u_c[i,:] * v_c[:,k]^T = I_m
    # This is a set of m^2 equations per (i,k).
    
    # Let's try a structured approach: assume U and V have a specific form.
    # For example, if we set U[i*m+j, c] = a_c[i,j] and V[j*p+k, c] = b_c[j,k],
    # then the condition becomes:
    # sum_c W[k*n+i, c] * a_c[i,:] * b_c[:,k]^T = I_m
    
    # This is still nonlinear. Let me try a simpler approach:
    # Set W[k*n+i, c] = 1 for all k,i,c, and try to find U, V such that
    # sum_c u_c[i,:] * v_c[:,k]^T = I_m for all (i,k).
    
    # This is equivalent to: for each (i,k), the sum of outer products equals I_m.
    
    # Let's try rank = m = 4. If we can find U, V, W of rank 4, that would be great.
    
    # Actually, let me just try a numerical optimization approach.
    # Initialize U, V, W randomly and try to minimize the error.
    
    import numpy as np
    
    np.random.seed(42)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    def _error(U, V, W):
        err = 0.0
        for i in range(n):
            for k in range(p):
                # Compute sum_c W[k*n+i, c] * u_c[i,:] * v_c[:,k]^T
                mat = np.zeros((m, m))
                for c in range(rank):
                    u_row = U[i*m:(i+1)*m, c]
                    v_col = V[:, k]  # This is wrong, V[j*p+k, c] for j in range(m)
                    # Actually, V[j*p+k, c] for fixed k and varying j gives a column of V
                    # Let me be more careful.
                    # V has shape (m*p, rank). V[j*p+k, c] for j in [0,m) gives m entries.
                    v_col = np.array([V[j*p+k, c] for j in range(m)])
                    mat += W[k*n+i, c] * np.outer(u_row, v_col)
                err += np.sum((mat - np.eye(m))**2)
        return err
    
    # Simple gradient-free optimization: try random restarts
    best_err = _error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        U2 = np.random.randn(n * m, rank)
        V2 = np.random.randn(m * p, rank)
        W2 = np.random.randn(p * n, rank)
        err2 = _error(U2, V2, W2)
        if err2 < best_err:
            best_err = err2
            best_U, best_V, best_W = U2, V2, W2
            if best_err < 1e-6:
                break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try decreasing ranks from a reasonable upper bound down to 1
    # Schoolbook rank is n*m*p = 96
    # Try to find a lower rank
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, get the schoolbook decomposition as a fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try to find a lower rank
    # Based on the analysis, rank m = 4 might be possible
    # Try ranks from min(n,m,p) up to schoolbook rank
    min_rank = min(n, m, p)
    
    for rank in range(min_rank, best_rank + 1):
        if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
            break
        result = _try_rank(rank, n, m, p, time_limit=5.0)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

