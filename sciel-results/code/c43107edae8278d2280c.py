"""Matrix multiplication tensor decomposition for 2x4 * 4x7.
Exploits the n=2 structure: treat A=[A_top, A_bot] (each 1x4) and B (4x7).
C_top = A_top @ B, C_bot = A_bot @ B.

Use the recurrence: C[j+1] = B[:, j+1] column? No, that doesn't work directly.
Better approach: use the known lower bound / known constructions for small cases.

Known result: matrix multiplication [2,4,7] can be done with rank at most 2*4*7 = 56 schoolbook.
But there exist sub-multiplicative algorithms. For [2,4,7], we can think of it as 
two separate [1,4,7] multiplications plus corrections.

Actually, let me think about this more carefully using the tensor decomposition framework.

The matrix multiplication tensor for n×m by m×p is a trilinear form:
C[i,k] = sum_j A[i,j] * B[j,k]

For n=2, m=4, p=7: we need 2*7 = 14 output entries, each being a sum of 4 products.

A rank-1 decomposition term looks like:
U[:,c] * V[:,c] * W[:,c] where:
- U has shape (n*m, rank) = (8, rank)
- V has shape (m*p, rank) = (28, rank)  
- W has shape (p*n, rank) = (14, rank)

And the product satisfies: C[i,k] = sum_c U[i*m + j? No...

Let me re-read the convention. W[k*n + i, c] is the output factor. So W has shape (p*n, rank) = (14, rank).

The decomposition is: the bilinear map (A, B) -> C where C[i,k] = sum_j A[i,j]B[j,k]

In index form: C[i,k] = sum_j (A_{i,j})(B_{j,k})

A rank-1 term contributes: u_{i,j} * v_{j,k} * w_{k,i} summed over c, giving C[i,k] = sum_c u_{i,j,c}... 

Wait, I need to be more careful. The standard tensor decomposition for matmul:

T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}... 

No. Let me use the standard formulation. The multiplication tensor is:
μ(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}

As a bilinear map, it can be written as:
μ(A,B) = sum_c (u_c^T A v_c)(w_c^T B x_c) ... no.

Standard rank-1 factorization of the matmul tensor:
C = sum_c (alpha_c^T A beta_c) (gamma_c^T B delta_c) doesn't work because we need elementwise output.

The correct formulation: The matmul tensor T has components T_{(i,j),(k),(l,m)} = delta_{j,l} * delta_{k,m}... 

Actually the simplest way: think of it as C[i,k] = sum_{j} A[i,j] B[j,k].

A rank-1 decomposition: C[i,k] = sum_c U[i*? ... 

The convention says:
- U has shape (n*m, rank): indexed by (i*m + j, c)
- V has shape (m*p, rank): indexed by (j*p + k, c) 
- W has shape (p*n, rank): indexed by (k*n + i, c)

And the relation is: C[i,k] = sum_c U[i*m + ? ... 

I think the convention is that the product is:
C[k*n + i] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed appropriately.

Actually for a valid decomposition:
C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

No, that's overcounting. Let me think again.

In a bilinear decomposition of matrix multiplication of rank r:
C = sum_{c=1}^{r} A * U_c ⊙ V_c * B * W_c  (schematically)

where ⊙ denotes appropriate outer product structure. The precise meaning:

For each c, we have vectors u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}.
Reshape u_c to (n,m), v_c to (m,p), w_c to (p,n).

Then: C[i,k] = sum_c sum_j (u_c[i,j]) (v_c[j,k]) (w_c[k,i])... no that's not bilinear.

The correct rank-1 term for a bilinear map C = sum_c (linear map on A)(linear map on B):
Each rank-1 term is: (a_c^T A) ⊗ (B b_c) reshaped... 

OK let me just use the standard: the matmul tensor M has M[i,j,k,l] = δ_{j,l} δ_{k,?}. 

Simplest: I'll just construct a valid decomposition by ensuring that for the given convention, the reconstruction works. Given the schoolbook in the seed:
U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for the c = i*m*p + j*p + k term.

Then C[k*n+i] should be: sum_c U[i*m+j_c, c] * V[j_c*p+k_c, c] * W[k_c*n+i_c, c]... 

I think the intended reconstruction is:
C[i,k] = sum_c sum_j U[i*m + j, c] * V[j*p + k, c] * (something with W)

But W[k*n+i, c] suggests W is already combined. Perhaps:
C[i,k] = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]

Let me verify with schoolbook: for term (i',j',k'), U[i*m+j,c]=1 only when i=i',j=j'. V[j'*p+k,c]=1 only when k=k'. W[k'*n+i',c]=1 only when k*k'=k, i'=i. So the product is 1 only when i=i', j=j', k=k'. And W[k*n+i,c]=W[k'*n+i',c]=1. So C[i,k] = sum over all (i',j',k') where i=i',k=k' of 1*1*1 = sum_{j'} 1 = 4... that's not right, should be sum_{j} A[i,j]B[j,k].

I think I'm overcomplicating. Let me just build a decomposition and test it numerically. The key insight for n=2: we can share work between the two rows of A.

Here's a concrete approach for [2,4,7]:
- C[0,k] = sum_j A[0,j] B[j,k]
- C[1,k] = sum_j A[1,j] B[j,k]

We can write A[1,j] = A[0,j] + (A[1,j] - A[0,j]). Let D[j] = A[1,j] - A[0,j].
Then C[1,k] = sum_j A[0,j] B[j,k] + sum_j D[j] B[j,k] = C[0,k] + D @ B.

So we need: C[0,k] = A_top @ B (a [1,4]×[4,7] multiply, rank ≤ 4*7=28)
and D @ B (also [1,4]×[4,7], rank ≤ 28), then C[1,k] = C[0,k] + (D@B)[k].

But this doesn't reduce the tensor rank of the full problem because we still need to compute D @ B as a bilinear map in (A,B).

The tensor for [1,4,7] has trivial lower bound 7 (since each output needs at least 7 independent inputs from B) and the Strassen-like or direct methods give rank 28 for schoolbook.

For the full [2,4,7] tensor, the schoolbook rank is 56. Can we do better?

Known results: For [2,m,p], there are constructions. Let me try a different approach: use the fact that we can combine the two outputs.

For each k (output column, 0..6):
C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]

That's 14 equations, each a sum of 4 products. Total 56 products schoolbook.

A rank-1 term in the tensor gives a contribution that is a product of a linear form in A, a linear form in B, contributing to one output entry... no, a rank-1 term of the trilinear form contributes to the tensor as an outer product, meaning it contributes (a^T A)(b^T B) to C in a specific way.

I'll just try to construct a valid decomposition with reduced rank using a different strategy. Given the difficulty of deriving a provably optimal rank here, and the need to produce a valid (not just better) program, let me ensure correctness first while attempting a modest improvement.

Let me try rank 50 as a target and construct it carefully, falling back to 56 if I can't get a valid smaller rank.
"""

import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, test_vectors=None):
    """Verify that the decomposition correctly computes C = A @ B."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    if test_vectors is None:
        # Use random test vectors
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
    else:
        A, B = test_vectors
    
    C_expected = A @ B  # shape (n, p)
    
    # Reconstruction: C[k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But we need to figure out the correct formula.
    # Standard interpretation: each rank-1 term c contributes
    # (u_c^T vec(A)) * (v_c^T vec(B)) to the output, masked by w_c
    # More precisely: C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c]
    # Wait, that doesn't work either. Let me think...
    
    # Actually, I believe the correct formula is:
    # For each term c, define:
    #   a_c = U[:,c] reshaped to (n,m)
    #   b_c = V[:,c] reshaped to (m,p)  
    #   w_c = W[:,c] reshaped to (p,n)
    # Then C[i,k] = sum_c (A * a_c)[i,:] dot (b_c * B)[:,k] * w_c[k,i]
    # No, that's not right either.
    
    # Let me try: C[i,k] = sum_c sum_j A[i,j] * U[i*m+j,c] * sum_j B[j,k] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook: U[i*m+j,c]=1 iff c=(i,j,k), V[j*p+k,c]=1 iff c=(i,j,k), W[k*n+i,c]=1 iff c=(i,j,k)
    # Then for fixed (i,k): C[i,k] = sum_c sum_j A[i,j]*delta_{c,(i,j,k)} * sum_j' B[j',k']*delta_{c,(i,j',k')} * delta_{c,(i,?,k)}
    # This only gives nonzero when c=(i,j,k) for some j, and k'=k
    # So C[i,k] = sum_j A[i,j] * 1 * B[j,k] * 1 = sum_j A[i,j]*B[j,k]. Correct!
    
    C_reconstructed = np.zeros((n, p))
    for c in range(rank):
        # Extract the linear functionals
        u_col = U[:, c]  # shape (nm,)
        v_col = V[:, c]  # shape (mp,)
        w_col = W[:, c]  # shape (pn,)
        
        # Reshape
        u_mat = u_col.reshape(n, m)  # (n, m)
        v_mat = v_col.reshape(m, p)  # (m, p)
        w_mat = w_col.reshape(p, n)  # (p, n)
        
        # Compute contribution
        # a_i = sum_j A[i,j] * u_mat[i,j] for each i
        # b_k = sum_j B[j,k] * v_mat[j,k] for each k
        # Then C[i,k] += a_i * b_k * w_mat[k,i]
        
        a_vec = np.array([sum(A[i, j] * u_mat[i, j] for j in range(m)) for i in range(n)])
        b_vec = np.array([sum(B[j, k] * v_mat[j, k] for j in range(m)) for k in range(p)])
        
        for i in range(n):
            for k in range(p):
                C_reconstructed[i, k] += a_vec[i] * b_vec[k] * w_mat[k, i]
    
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        raise ValueError(f"Decomposition failed! Max error: {np.max(np.abs(C_reconstructed - C_expected))}")
    
    return True

def _find_decomposition(n, m, p, target_rank):
    """Find a rank-target_rank decomposition using linear algebra."""
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor has components T[i*m+j, j*p+k, k*n+i] = 1 for the matmul tensor
    # We want to find U, V, W of rank r such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j, j'} * delta_{k, k'}
    # Wait, I need to be more careful about what the tensor actually is.
    
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: C[i,k] = sum_{j} sum_{j'} sum_{k'} A[i,j] B[j',k'] T[i,j,j',k',k]
    # where T[i,j,j',k',k] = delta_{j,j'} delta_{k',k}
    
    # A rank-1 decomposition term c gives:
    # C[i,k] += (sum_j A[i,j] U[i*m+j,c]) * (sum_{j',k'} B[j',k'] V[j'*p+k',c]) * W[k*n+i,c]
    # = (sum_j A[i,j] U[i*m+j,c]) * (sum_j' B[j',k] V[j'*p+k,c]) * W[k*n+i,c]
    
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] = delta_{j,j'} * delta_{k,?}
    # Actually: sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] should give
    # delta_{j,j'} when summed appropriately... 
    
    # Let me set up the linear system. For each (i, j, j', k), we need:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j, j'}
    
    # This is a system of equations in the variables U, V, W.
    # It's nonlinear (products of U*V*W), so we can't solve it directly.
    
    # Alternative: fix one matrix and solve for the others iteratively.
    # Or use the fact that we can parameterize differently.
    
    # Let me try a different approach: assume W is diagonal in some sense.
    # If we set W[k*n+i, c] = 1 for specific (k,i,c) pairs, we reduce to a bilinear problem.
    
    # Actually, let me just try to construct a valid decomposition with reduced rank
    # by exploiting the n=2 structure.
    
    # Strategy: 
    # C[0,k] = sum_j A[0,j] B[j,k]  -- needs rank 28 (schoolbook for [1,4,7])
    # C[1,k] = sum_j A[1,j] B[j,k] = sum_j (A[0,j] + (A[1,j]-A[0,j])) B[j,k]
    #        = C[0,k] + sum_j (A[1,j]-A[0,j]) B[j,k]
    
    # The term sum_j (A[1,j]-A[0,j]) B[j,k] is a [1,4,7] multiplication where the "A" vector
    # is D = A[1,:]-A[0,:]. Since D is linear in A, we can write:
    # D[j] = A[1,j] - A[0,j] = sum_{j'} alpha_{j,j'} A[j', j'] where alpha is a fixed matrix.
    
    # So D @ B is still a bilinear map in (A, B), and can be decomposed with rank 28.
    # Total rank: 28 + 28 = 56. No improvement.
    
    # To get improvement, we need to share terms between the two decompositions.
    # This requires a more sophisticated construction.
    
    # Let me try rank 49 and see if a valid decomposition exists.
    # I'll use a greedy/optimization approach.
    
    # For now, return schoolbook as a valid baseline.
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm, mp, pn = n*m, m*p, p*n
    
    # Try to find a reduced-rank decomposition
    # Start with schoolbook rank 56, try to reduce
    
    best_U, best_V, best_W = None, None, None
    
    for target_rank in range(42, 57):  # Try ranks from 42 to 56
        try:
            U, V, W = _find_decomposition(n, m, p, target_rank)
            if U is not None:
                _verify_decomposition(U, V, W, n, m, p)
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition with this rank
        except Exception:
            continue
    
    # If no reduced rank found, use schoolbook
    if best_U is None:
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        _verify_decomposition(U, V, W, n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
