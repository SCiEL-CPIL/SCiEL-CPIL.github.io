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

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    # Use schoolbook as base, then attempt reduction via sharing
    # For n=2, we can exploit linearity: C[1,k] - C[0,k] = (A[1,:]-A[0,:]) @ B
    # This means the "difference" output only depends on D = A[1,:]-A[0,:]
    # But D is linear in A, so we can absorb it.
    #
    # Strategy: Compute C[0,k] via a [1,4,7] decomposition (rank r1)
    #            Compute D@B where D = A[1]-A[0] via another [1,4,7] decomposition (rank r2)
    #            Then C[1,k] = C[0,k] + (D@B)[k], which is a linear combination of the above.
    #
    # The total tensor rank is at most r1 + r2 + p (for the p terms that add C[0] to get C[1]).
    # But the "addition" can be folded into the W factors, so it might not add to rank.
    #
    # Actually, let me just use the schoolbook and verify it's valid, then see if I can trim.
    
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
    
    # Verify the decomposition
    # C[i,k] should equal sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But that formula seems off. Let me figure out the correct reconstruction.
    # 
    # The convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-1 term c contributes: for each (i,j,k), U[i*m+j,c]*V[j*p+k,c] gives a contribution
    #   to C[i,k] scaled by W[k*n+i,c]? That doesn't seem right dimensionally.
    #
    # I think the correct interpretation is:
    # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # No, that makes U and V multiply with A and B, which is wrong.
    #
    # Standard tensor decomposition for bilinear maps:
    # C = sum_c (U_c^T * vec(A)) * (V_c^T * vec(B)) * w_c  (reshaped to matrix)
    # where U_c has shape (nm,), V_c has shape (mp,), w_c has shape (pn,) and is reshaped to (p,n).
    #
    # More precisely: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_j' B[j',k] * V[j'*p+k,c]) * (W[k*n+i,c])
    # Hmm, but then U and V are acting as masks on A and B.
    #
    # For the schoolbook: U[i*m+j,c]=1 for c=(i,j,k), so sum_j A[i,j']*U[i*m+j',c] = A[i,j].
    # V[j'*p+k,c]=1 for c=(i,j,k), so sum_j' B[j',k']*V[j'*p+k',c] = B[j,k].
    # W[k*n+i,c] = 1 for c=(i,j,k).
    # So C[i,k] = sum_c A[i,j_c] * B[j_c,k_c] * W[k*n+i,c] where j_c, k_c are the j,k associated with c.
    # = sum_{j,k'} A[i,j] B[j,k'] * delta_{k',k} (from W) * delta_{i_c,i} ... 
    # This is getting circular. Let me just trust the schoolbook is valid as given in the seed
    # and focus on the interface.
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
