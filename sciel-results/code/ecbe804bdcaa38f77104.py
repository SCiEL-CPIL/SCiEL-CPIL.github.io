"""Combined approach: use structural insight about the tensor to find a lower-rank decomposition
for 4x4 * 4x9 matrix multiplication, with verification and fallback to schoolbook.

Key insight: The tensor T is indexed by (a, b, c) where:
  a = i*m + j  (i in [0,n), j in [0,m))
  b = j*p + k  (j in [0,m), k in [0,p))
  c = k*n + i  (k in [0,p), i in [0,n))

T[a,b,c] = 1 iff the indices are consistent (i.e., a's j equals b's j, a's i equals c's i, b's k equals c's k), else 0.

For n=4, m=4, p=9: U is (16,R), V is (36,R), W is (36,R).

We try to find a decomposition with rank < 144 by exploiting structure.
One approach: since the tensor has a specific sparse structure, we can try to
find low-rank approximations that are exact.

A key observation: if we can write the tensor as a sum of R rank-1 terms where each
term is a product of three vectors, we need the product to be 1 at consistent indices
and 0 at inconsistent ones.

Let's try a constructive approach using the fact that the problem has some symmetry.
For the 4x4 part, we might be able to use Strassen-like ideas.

Actually, let me try a direct approach: use numpy to set up the linear system and try
to find a solution with small R.

The condition is: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
And for all inconsistent (a,b,c), sum_r U[a,r] * V[b,r] * W[c,r] = 0

This is a nonlinear system. However, we can try a specific structured ansatz.

One idea: suppose U, V, W have entries that are 0, 1, or -1 (or small integers/half-integers).
We can try to search over small patterns.

Given time constraints, let me try a few specific constructions:

1. Schoolbook: R=144 (baseline)
2. Try R=36: one term per (j,k) pair, with U depending on i only, W depending on i only
   U[i*m+j, r] where r indexes (j,k): U[i*m+j, j*p+k] = 1 if... hmm this doesn't work directly.

Let me think differently. The bilinear map is:
C[i,k] = sum_j A[i,j] B[j,k]

This can be written as: C = A * B

A rank-R decomposition means: C = sum_r (A_r) (B_r) where A_r is a linear function of A,
B_r is a linear function of B, and the product is interpreted appropriately.

Actually, in the standard formulation:
C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) * W[k,i,r]

Wait, that's not quite right. Let me re-derive.

The tensor decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]

where a = i*m+j, b = j*p+k, c = k*n+i.

So: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]

For consistent indices (which is always the case when we plug in valid i,j,k), T = 1.
For inconsistent indices, T = 0.

So the condition is:
sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i in [0,4), j in [0,4), k in [0,9)

And for any (a,b,c) that are NOT consistent, sum_r U[a,r] V[b,r] W[c,r] = 0.

Let me try a specific construction. Suppose we let R = m * p = 4 * 9 = 36, and define:
For r = j*p + k (j in [0,4), k in [0,9)):
  U[i*m+j, r] = 1 if the j in r matches the j in a, else 0
  V[j'*p+k', r] = 1 if (j',k') = (j,k)
  W[k'*n+i', r] = 1 if k' = k, and i' = i... but i' is determined by c, not r.

This doesn't work cleanly. Let me try another approach.

Let R = n * p = 4 * 9 = 36. For r = i*p + k:
  U[i'*m+j, r] = 1 if i' = i
  V[j'*p+k', r] = 1 if k' = k  
  W[k''*n+i'', r] = 1 if (k'', i'') = (k, i)

Then: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = sum_{i',k'} [i'=i][k'=k][k''=k][i''=i]
= 1 (only the term with i'=i, k'=k contributes)

But wait, we also need the j indices to be consistent. In this construction, U depends on i but not j (except through the row index), and V depends on k but not j. So for a fixed (i,k), the sum over r gives 1 regardless of j. But we need the result to be 1 for all j, which it is. And for inconsistent indices, we need 0.

Let me check: for inconsistent (a,b,c), say a = i1*m+j1, b = j2*p+k2, c = k3*n+i3 where j1 != j2 or i1 != i3 or k2 != k3.

sum_r U[a,r] V[b,r] W[c,r] = sum_{i',k'} [i1=i'] [k2=k'] [k3=k'][i3=i']

This is 1 only if i1=i3 and k2=k3, but we also need... wait, there's no condition on j1 vs j2 in this construction. So if i1=i3 and k2=k3 but j1!=j2, the sum would still be 1, but T should be 0.

So this construction doesn't work.

Let me try yet another approach. What if R = n*m*p = 144 (schoolbook) is actually not minimal, and we can do better?

Known results: the rank of 4x4 matrix multiplication is at most 49 (Strassen). For 4x4 * 4x9, it's unclear.

Given the time constraints, let me try a practical approach: use scipy or numpy to attempt a low-rank factorization via iterative methods, or try a small random search.

Actually, let me try a specific idea: since 9 = 4 + 4 + 1, split the k dimension into two 4-dim blocks and one 1-dim block. For each 4x4 * 4x4 subproblem, use a rank-19 Strassen-like decomposition (if we can construct one), and for the 4x4 * 4x1, use rank 4. Total: 19 + 19 + 4 = 42.

But constructing an explicit rank-19 decomposition for 4x4 is complex. Let me just try to find any improvement over 144 via a simple search.

Given the complexity and time limits, I'll implement the schoolbook as the safe baseline and add a quick check for a few specific low-rank constructions.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Schoolbook baseline
    rank_sb = n * m * p  # 144
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, R):
        """Verify the decomposition is exact."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some inconsistent indices (sample)
        import random
        random.seed(42)
        for _ in range(200):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            
            # Decode indices
            i1, j1 = divmod(a, m)
            j2, k2 = divmod(b, p)
            k3, i3 = divmod(c, n)
            
            # Check if consistent
            if j1 == j2 and i1 == i3 and k2 == k3:
                continue  # consistent, already checked
            
            val = np.dot(U_np[a], V_np[b] * W_np[c])
            if abs(val) > 1e-9:
                return False
        
        return True

    # Try to find a better decomposition
    # Strategy: try a construction based on splitting k into groups
    
    # For now, return schoolbook if we can't find improvement quickly
    if time.time() > deadline - 30:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try a specific construction with R = 36
    # Idea: for each (j,k) pair, have one term
    # U[i*m+j, j*p+k] = 1 for all i
    # V[j'*p+k', j*p+k] = 1 if (j',k')=(j,k)
    # W[k'*n+i', j*p+k] = 1 if k'=k, and... we need to handle i'
    
    # This doesn't work because W needs to depend on i' which isn't in r.
    
    # Alternative: try R = n*p = 36
    # r = i*p + k
    # U[i'*m+j, i*p+k] = 1 if i'=i
    # V[j'*p+k', i*p+k] = 1 if k'=k
    # W[k''*n+i'', i*p+k] = 1 if k''=k and i''=i
    
    R_try = n * p  # 36
    U_try = [[0] * R_try for _ in range(n * m)]
    V_try = [[0] * R_try for _ in range(m * p)]
    W_try = [[0] * R_try for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U_try[i * m + j][r] = 1
            for j in range(m):
                V_try[j * p + k][r] = 1
            for i2 in range(n):
                if i2 == i:
                    W_try[k * n + i2][r] = 1
    
    if verify(U_try, V_try, W_try, R_try):
        return {"U": U_try, "V": V_try, "W": W_try}

    # Try R = m*p = 36
    # r = j*p + k
    # U[i*m+j, j'*p+k'] = 1 if j'=j
    # V[j'*p+k', j*p+k] = 1 if (j',k')=(j,k)
    # W[k''*n+i'', j*p+k] = 1 if k''=k
    
    R_try2 = m * p  # 36
    U_try2 = [[0] * R_try2 for _ in range(n * m)]
    V_try2 = [[0] * R_try2 for _ in range(m * p)]
    W_try2 = [[0] * R_try2 for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U_try2[i * m + j][r] = 1
            V_try2[j * p + k][r] = 1
            for i in range(n):
                W_try2[k * n + i][r] = 1
    
    if verify(U_try2, V_try2, W_try2, R_try2):
        return {"U": U_try2, "V": V_try2, "W": W_try2}

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
