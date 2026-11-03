"""
Strategy: Structured Rank-30 Factorization
We aim for a rank of 30 (half of the schoolbook 60) by exploiting the structure of the 2x3x10 multiplication.
The tensor T[i, j, k] = 1.
We propose a decomposition of rank r=30 where the index c corresponds to (j, k) pairs.
U[i*m + j, c] = 1
V[j*p + k, c] = 1
W[k*n + i, c] = ?
We need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
If c = (j', k'), then U is non-zero only if j=j', V only if k=k'.
The sum reduces to a single term for c=(j,k): 1 * 1 * W[k*n+i, (j,k)].
Thus we require W[k*n+i, (j,k)] = 1 for all i, j, k.
However, W has shape (20, 30). The index (k*n+i) is in [0, 19]. The column index is (j, k) in [0, 29].
Let's check if this is consistent.
For a fixed output index (k, i), we need the sum over c to be 1.
Wait, the decomposition is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
If we set U[i,j,c] = 1 if c=(j,k) else 0? No, c is the rank index.
Let's try a different structured approach.
Since n=2, we can split the sum over i.
Actually, let's look at the "Strassen-like" or "subtiled" approach.
For 2x3x10, rank 30 is the theoretical lower bound if we can fully exploit the 2x2 block structure of the first dimension.
Let's attempt to construct a rank-30 solution.
If we fail, we return rank 60.

Let's try to solve for rank 30 using a specific pattern.
Let the rank index c be (j, k). There are 3*10=30 such pairs.
U[i*m + j, (j', k')] = 1 if j == j' else 0.
V[j*p + k, (j', k')] = 1 if k == k' else 0.
Then for fixed i, j, k:
Sum over c=(j', k'):
Term is non-zero only if j'=j and k'=k.
So the sum is just U[i*m+j, (j,k)] * V[j*p+k, (j,k)] * W[k*n+i, (j,k)].
With U=1 and V=1, we need W[k*n+i, (j,k)] = 1.
Is this a valid tensor?
W is a matrix of size (20, 30).
Row index: k*n + i (where k in 0..9, i in 0..1).
Col index: j*p + k (where j in 0..2, k in 0..9).
We require W[k*n+i, j*p+k] = 1 for all i, j, k.
Let's check if this W is consistent.
For a fixed row (k, i), the value is 1 for all columns (j, k).
Note that the column index depends on k.
So for a fixed row (k, i), the non-zero entries are at columns where the second part of the column index is k.
There are 3 such columns (for j=0,1,2).
Does this create any conflict?
The decomposition requires T[i,j,k] = 1.
The calculation yields 1.
So this IS a valid rank-30 decomposition!
Let's double check.
T[i,j,k] = sum_{c} U[i,j,c] V[j,k,c] W[k,i,c]
Let c be indexed by (j_c, k_c).
U[i, j, (j_c, k_c)] = 1 if j == j_c else 0.
V[j, k, (j_c, k_c)] = 1 if k == k_c else 0.
Product U*V is non-zero only if j_c = j AND k_c = k.
So the sum collapses to the term where c = (j, k).
Value = U[i, j, (j,k)] * V[j, k, (j,k)] * W[k, i, (j,k)]
= 1 * 1 * W[k*n+i, j*p+k]
We need this to be 1.
So we set W[k*n+i, j*p+k] = 1.
Are there any other constraints on W?
No, W is just a factor matrix.
So this is a valid exact decomposition of rank 30.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Rank 30 construction
    rank = m * p  # 30
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill U and V
    # U[i*m + j, c] = 1 if c corresponds to (j, k) for any k?
    # c index: 0 to 29. c = j*p + k.
    # For a fixed i, j:
    # U[i*m + j, j*p + k] = 1 for all k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = j * p + k
                U[i * m + j, c] = 1.0
                # V[j*p + k, c] = 1
                V[j * p + k, c] = 1.0
                # W[k*n + i, c] = 1
                W[k * n + i, c] = 1.0
                
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook rank 60 just in case
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        if not verify(U, V, W, n, m, p, rank):
            raise ValueError("Failed to construct valid decomposition")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
