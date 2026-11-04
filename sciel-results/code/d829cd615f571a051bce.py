"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Approach: Exploit the n=2 structure. Since both rows of A share the same B,
each rank-1 term can serve both output rows simultaneously.

For the matmul tensor C[i,k] = sum_j A[i,j] * B[j,k]:
- U has shape (n*m, rank) = (8, rank), indexed by (i*m + j, c)
- V has shape (m*p, rank) = (28, rank), indexed by (j*p + k, c)
- W has shape (p*n, rank) = (14, rank), indexed by (k*n + i, c)

The reconstruction formula is:
C[i,k] = sum_c U[i*m + j_c, c] * V[j_c*p + k_c, c] * W[k_c*n + i, c]
where for each term c, j_c and k_c are the specific indices where U and V are non-zero.

For rank = m*p = 28:
For each (j, k) pair (j in 0..3, k in 0..6), create one term c:
- U[0*m + j, c] = 1, U[1*m + j, c] = 1  (both rows of A at column j)
- V[j*p + k, c] = 1  (column k of B at row j)
- W[k*n + 0, c] = 1, W[k*n + 1, c] = 1  (both output rows at column k)

Verification:
C[i,k'] = sum_c U[i*m + j_c, c] * V[j_c*p + k_c, c] * W[k_c*n + i, c]
For a fixed (i, k'), the sum is over all c. Term c=(j,k) contributes only when:
- k_c = k' (from V being non-zero at k_c)
- W[k'*n + i, c] != 0, which is true since W[k*2+i, c]=1 for both i
- U[i*m + j, c] != 0, which is true since U[i*m+j, c]=1 for both i

So C[i,k'] = sum_{j=0}^{3} 1 * 1 * 1 = sum_j A[i,j]*B[j,k'] ... 

Wait, I need to be more careful. The reconstruction should be:
C[i,k] = sum_c (sum over the relevant indices)

Actually, the standard tensor decomposition for bilinear maps works as:
vec(C) = sum_c (U_c ⊙ V_c) ⊗ W_c  ... no.

Let me think of it as: the matmul tensor T has T[i,j,k,l] = δ_{j,l} δ_{k,m}... 

The correct formulation: we want C[i,k] = sum_j A[i,j] B[j,k].
This is a trilinear form in (A, B, C^T): sum_{i,j,k} C[i,k] A[i,j] B[j,k]... no.

As a bilinear map: (A, B) -> C where C = A @ B.
The tensor of this bilinear map has components T[(i,j),(j'),(k)] = δ_{j,j'} δ_{...}

Standard: T_{(i,j), (j,k), (i,k)} = δ_{j,j} (the middle index matches).

A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c where:
u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}

And the bilinear map is: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) (sum_j' V[j'*p+k, c] B[j',k]) W[k*n+i, c]

No wait, that's not right either. Let me use the standard definition.

For a bilinear map B: V × W -> Z, the tensor is in V* ⊗ W* ⊗ Z.
A rank-1 term is f ⊗ g ⊗ h where f ∈ V*, g ∈ W*, h ∈ Z.
The bilinear map is: B(v,w) = sum_c f_c(v) g_c(w) h_c

In our case:
- V = R^{n*m} (flattened A), W = R^{m*p} (flattened B), Z = R^{p*n} (flattened C)
- f_c ∈ (R^{n*m})*, g_c ∈ (R^{m*p})*, h_c ∈ R^{p*n}
- B(A,B)[k*n+i] = sum_c f_c(A) g_c(B) h_c[k*n+i]

Where f_c(A) = sum_{i,j} U[i*m+j, c] A[i,j]
      g_c(B) = sum_{j,k} V[j*p+k, c] B[j,k]

So C[k*n+i] = sum_c (sum_{i',j'} U[i'*m+j', c] A[i',j']) (sum_{j'',k''} V[j''*p+k'', c] B[j'',k'']) W[k*n+i, c]

For the schoolbook with rank 56: for each (i,j,k), one term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
Then C[k*n+i] = sum_c (A[i_c, j_c]) (B[j_c, k_c]) W[k*n+i, c]
= sum over all (i_c, j_c, k_c) where W[k*n+i, c_c]=1, which means k_c=k, i_c=i.
= sum_{j_c} A[i, j_c] B[j_c, k] * 1 = sum_j A[i,j] B[j,k]. Correct!

Now for rank 28: for each (j,k), one term with U[i*m+j,c]=1 for i=0,1; V[j*p+k,c]=1; W[k*n+i,c]=1 for i=0,1.
C[k*n+i] = sum_c (sum_{i',j'} U[i'*m+j', c] A[i',j']) (sum_{j'',k''} V[j''*p+k'', c] B[j'',k'']) W[k*n+i, c]

For term c=(j,k): 
f_c(A) = sum_{i',j'} U[i'*m+j', c] A[i',j'] = A[0,j] + A[1,j] (since U[i*m+j,c]=1 for i=0,1, and U[i*m+j',c]=0 for j'≠j)
g_c(B) = sum_{j'',k''} V[j''*p+k'', c] B[j'',k''] = B[j,k] (since V[j*p+k,c]=1, others 0)
W[k*n+i, c] = 1 for i=0,1.

So C[k*n+i] gets contribution: (A[0,j] + A[1,j]) * B[j,k] * 1 for each j.
Summing over all terms c=(j,k) for fixed k: sum_j (A[0,j]+A[1,j]) B[j,k] = (A[0,:]+A[1,:]) @ B[:,k]

But we need C[k*n+0] = A[0,:] @ B[:,k] and C[k*n+1] = A[1,:] @ B[:,k].
We're getting C[k*n+0] = (A[0,:]+A[1,:]) @ B[:,k] which is wrong.

So rank 28 with this naive approach doesn't work. The issue is that U can't distinguish between the two rows if we set both to 1.

Let me reconsider. We need C[k*n+0] = sum_j A[0,j] B[j,k] and C[k*n+1] = sum_j A[1,j] B[j,k].

If we use rank 28 with terms indexed by (j,k):
f_c(A) should give something that, when multiplied by g_c(B) and W, yields the right result.

If W[k*n+0, c] = 1 and W[k*n+1, c] = 1, then both outputs get the same contribution from each term. That's not what we want.

Alternative: use W[k*n+0, c] = 1, W[k*n+1, c] = 0 for some terms, and vice versa. But that would require more terms.

Actually, the minimum rank for [2,4,7] is at least max(2*7, 4*7) = 28? No, the lower bound is at least n*p = 14 and m*p = 28 and n*m = 8, so at least 28. But can we achieve 28?

For rank 28, we need exactly one term per (j,k). Let's see if we can assign U, V, W appropriately.

For term c=(j,k):
- V[j*p+k, c] = 1 (fixed, to pick out B[j,k])
- We need: C[k*n+0] += f_c(A) * 1 * W[k*n+0, c]
           C[k*n+1] += f_c(A) * 1 * W[k*n+1, c]

And we need sum_j f_{(j,k)}(A) W[k*n+0, c_{j,k}] = sum_j A[0,j] B[j,k] ... but B[j,k] is already in g_c.

Wait, I think I'm confusing myself. Let me restart.

C[i,k] = sum_j A[i,j] B[j,k]

For a rank-r decomposition: C[i,k] = sum_{c=1}^r u_c(i) v_c(k) w_c(i,k) ... no.

Standard: C = sum_c (A * U_c) (V_c * B) ... where * denotes appropriate contraction.

I'll just go with the schoolbook rank 56 to be safe, since my analysis shows the rank-28 attempt fails.

Actually, let me try a different rank-28 construction. What if:
- For term c=(j,k): U[0*m+j, c] = 1, U[1*m+j, c] = 0; V[j*p+k, c] = 1; W[k*n+0, c] = 1, W[k*n+1, c] = 0
  AND another term for the second row? That would be rank 56 again.

I think the minimum rank is actually 56 for this problem, or at least I can't easily find a better construction. Let me just return the schoolbook.

Wait, actually there might be a better approach. Let me think about it differently.

We have 14 output entries, each is a dot product of length 4. The total number of "independent" bilinear forms needed is at most 14*4 = 56, but there could be sharing.

For now, I'll return the schoolbook decomposition with rank 56.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56 schoolbook
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
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result
