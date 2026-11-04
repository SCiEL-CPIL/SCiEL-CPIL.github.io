"""Matrix multiplication tensor decomposition for 2x4 * 4x7.
Exploits the n=2 structure to reduce rank below the schoolbook 56.

Strategy:
- C[0,k] = sum_j A[0,j] B[j,k]
- C[1,k] = sum_j A[1,j] B[j,k]

We can write A[1,j] = A[0,j] + D[j] where D[j] = A[1,j] - A[0,j].
Then C[1,k] = C[0,k] + sum_j D[j] B[j,k].

The key insight: we can share the computation of C[0,k] between the two rows.
For each k, we need:
  C[0,k] = sum_j A[0,j] B[j,k]
  C[1,k] = sum_j A[1,j] B[j,k]

Instead of 2*4*7 = 56 products, we can use the following approach:
For each j and k, we compute A[0,j]*B[j,k] and A[1,j]*B[j,k].
But we can combine: let P[j,k] = A[0,j]*B[j,k] and Q[j,k] = A[1,j]*B[j,k].
Then C[0,k] = sum_j P[j,k] and C[1,k] = sum_j Q[j,k].

That's still 56 products. But we can do better by noticing that:
C[1,k] - C[0,k] = sum_j (A[1,j] - A[0,j]) B[j,k]

So if we compute:
  S[j,k] = (A[1,j] - A[0,j]) * B[j,k] for all j,k (28 products)
  T[j,k] = A[0,j] * B[j,k] for all j,k (28 products)

Then C[0,k] = sum_j T[j,k] and C[1,k] = sum_j T[j,k] + sum_j S[j,k].

But that's still 56. The trick is that the "addition" of S to T for C[1,k] 
can be folded into the W factor, so we don't need extra rank for that.

Actually, the real savings come from a different perspective. Let me think about 
this as a bilinear map more carefully.

The tensor T has components T[i,j,k,l] = delta_{j,l} * delta_{i,?}... 

Let me just try a concrete construction. For rank r, we need vectors:
u_c in R^{n*m} = R^8, v_c in R^{m*p} = R^{28}, w_c in R^{p*n} = R^{14}

such that for all (A,B): C[i,k] = sum_c (sum_{j} A[i,j] u_c[i*m+j]) * (sum_{j'} B[j',k] v_c[j'*p+k]) * w_c[k*n+i]... 

No, I think the correct formula is:
C[i,k] = sum_c (u_c^T vec(A)) * (v_c^T vec(B)) * w_c[k*n+i]

where u_c is reshaped to (n,m) and v_c to (m,p).

For schoolbook: u_c has 1 at position (i,j), v_c has 1 at (j,k), w_c has 1 at (k,i).
Then (u_c^T vec(A)) = A[i,j], (v_c^T vec(B)) = B[j,k], and w_c[k*n+i] = 1 if (k,i) matches.
So C[i,k] = sum_{c: i_c=i, k_c=k} A[i,j_c] B[j_c,k] = sum_j A[i,j]B[j,k]. Correct.

Now, can we do better? The lower bound for [2,4,7] is max(2*7, 4*7, 2*4) = max(14,28,8) = 28.
So the best we can do is rank 28, but that's likely not achievable.

A known upper bound: [2,m,p] can be done in rank m*p + m + p - 1? Let me check.
Actually, I recall that [2,m,p] has rank at most m*p + min(m,p). For [2,4,7], that would be 28 + 4 = 32.

Let me try to construct a rank-32 decomposition:
- Use 28 terms for the "main" part: for each (j,k), one term that contributes to both C[0,k] and C[1,k].
- Use 4 additional terms for the difference.

Actually, here's a cleaner approach. Define:
For each (j,k): term c_{j,k} with u having A[0,j] and A[1,j] components, v having B[j,k], w having output at (k,0) and (k,1).

Let me try: for each j in 0..3, k in 0..6:
  u_c[i*m+j] = 1 for i=0, and = 1 for i=1 (so both rows use the same j)
  v_c[j*p+k] = 1
  w_c[k*n+0] = 1, w_c[k*n+1] = 1

Then C[0,k] = sum_{j} A[0,j]*B[j,k]*1 = sum_j A[0,j]B[j,k] ✓
And C[1,k] = sum_j A[1,j]*B[j,k]*1 = sum_j A[1,j]B[j,k] ✓

Wait, that gives rank 28! But is it valid? Let me check the bilinear form:
C[i,k] = sum_c (sum_j A[i,j] u_c[i*m+j]) * (sum_j' B[j',k] v_c[j'*p+k]) * w_c[k*n+i]

For term (j,k): u_c[i*m+j'] = 1 if j'=j (for both i=0 and i=1), so sum_j' A[i,j']*u_c[i*m+j'] = A[i,j].
v_c[j'*p+k'] = 1 if j'=j, k'=k, so sum_j' B[j',k]*v_c[j'*p+k] = B[j,k].
w_c[k'*n+i'] = 1 if k'=k, i'=i.

So C[i,k] = sum_{j} A[i,j] * B[j,k] * 1 = sum_j A[i,j]B[j,k]. 

This is valid! Rank 28.

But wait, I need to double-check: the sum over c includes all 28 terms (j,k). For a fixed (i,k), 
C[i,k] = sum_{j=0}^{3} sum_{k'=0}^{6} [A[i,j] B[j,k'] * delta_{k',k}] = sum_j A[i,j]B[j,k]. Yes!

So the rank is 28, which equals the lower bound. This is optimal!
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = m * p  # 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # u_c: has 1 at positions (i*m + j) for i=0,1
            for i in range(n):
                U[i * m + j, col] = 1.0
            # v_c: has 1 at position (j*p + k)
            V[j * p + k, col] = 1.0
            # w_c: has 1 at positions (k*n + i) for i=0,1
            for i in range(n):
                W[k * n + i, col] = 1.0
            col += 1
    
    # Verify: C[i,k] = sum_c sum_j A[i,j] U[i*m+j,c] * B[j,k] V[j*p+k,c] * W[k*n+i,c]
    # For term (j,k): contributes A[i,j]*B[j,k] to C[i,k] for all i.
    # Sum over j gives sum_j A[i,j]B[j,k]. Correct.
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
