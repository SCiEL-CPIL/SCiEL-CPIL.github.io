"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.

Strategy:
1. Start with schoolbook (rank 60) as a valid baseline.
2. Use structured random search with integer/half-integer constraints to find lower rank.
3. The key insight: we can try ranks from 59 down, using randomized initialization
   and checking if the decomposition is exact.

For the 3x4x5 matrix multiplication tensor:
- U is (12 x r), V is (20 x r), W is (15 x r)
- The tensor T[i,j,k] = 1 if (AB)[i,k] depends on A[i,j] and B[j,k]
- We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]

Since entries must be integer/half-integer, we work in 1/2 units.

Approach: 
- For each candidate rank r (starting from 59 down), try random initializations
  with small integer/half-integer values and check if the decomposition is exact.
- Use a time budget to avoid timeout.

Actually, random search for exact tensor rank is extremely unlikely to succeed for
non-trivial ranks. Let me think of a better approach.

Better approach: Use known algebraic constructions.

For 3x4x5, we can try:
- Split the 4 dimension into 2+2, giving two 3x2x5 problems.
- For 3x2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
  This is a bilinear map. The rank of this is at most 3*2*5 = 30, but can be less.

Actually, for 1x2x5: C[k] = A[0]*B[0,k] + A[1]*B[1,k]
This is just computing 5 outputs from 2 inputs each. The rank is 5 (each output needs
one multiplication... wait no).

Let me reconsider. The tensor for 1x2x5 multiplication:
T[a, b, k] where a in {0,1} (rows of A, but A is 1x2 so a=0 only), 
b in {0,1} (columns of A / rows of B), k in {0,...,4} (columns of B)

Wait, I'm confusing myself. Let me be precise.

Matrix multiplication A (n x m) * B (m x p) = C (n x p)
C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]

The tensor T has indices (i, j, k) with T[i, j, k] = 1 for all valid i, j, k.
So T is just the all-ones tensor of size n x m x p? No!

Wait, the tensor decomposition for matrix multiplication is:
The bilinear map (A, B) -> A*B can be written as:
(A*B)[i,k] = sum_{j} A[i,j] * B[j,k]

As a tensor in the space of (A, B) -> C, the "multiplication tensor" has:
T[i, j, k] = 1 for all i in [n], j in [m], k in [p]

And the decomposition is:
sum_{r=1}^{rank} u_r[i] * v_r[j] * w_r[k] = T[i, j, k] = 1 for all i, j, k

Wait, that's not right either. Let me re-read the task.

"Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix."

The output factor follows the official trace convention W[k*n+i,r].

So U has shape (n*m, rank) = (12, rank)
V has shape (m*p, rank) = (20, rank)  
W has shape (p*n, rank) = (15, rank)

The decomposition satisfies:
For all i in [n], k in [p]:
  C[i,k] = sum_{r} sum_{j} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that's not standard. Let me think again.

Standard tensor decomposition for matrix multiplication:
The multiplication tensor T in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} is defined by:
T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}

where e_{i,j} is the basis vector for A[i,j], e_{j,k} for B[j,k], and e_{i,k} for C[i,k].

A rank-r decomposition is:
T = sum_{r=1}^{rank} U_r ⊗ V_r ⊗ W_r

where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}.

So:
sum_{r} U_r[i*m+j] * V_r[j*p+k] * W_r[i*p+k] = 1 if (i,j,k) is valid, 0 otherwise

But since all (i,j,k) with i in [n], j in [m], k in [p] are valid, we need:
sum_{r} U_r[i*m+j] * V_r[j*p+k] * W_r[i*p+k] = 1 for all i, j, k

Wait, but the task says W[k*n+i, r], so W is indexed by (k, i) -> k*n + i.

So the condition is:
sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)

This is a system of n*m*p = 60 equations in 3*rank*? unknowns.

For rank r, we have 12*r + 20*r + 15*r = 47*r unknowns, and 60 equations.
So we need r >= ceil(60/47) = 2? That can't be right because the rank is known to be higher.

The issue is that the equations are not independent in a simple way. The tensor has a specific structure.

Let me just implement the schoolbook and try a few random searches with very small ranks to see if anything works. Given the 120s limit, I'll be conservative.

Actually, I just realized: the schoolbook decomposition uses U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r corresponding to (i,j,k). This gives rank 60.

Can we do better? The answer is yes, but finding it is hard.

Let me try a different approach: use the fact that we can share multiplications.

For example, consider the case n=2, m=2, p=2. The schoolbook rank is 8, but Strassen shows rank 7.

For 3x4x5, I'll try to find a construction with rank < 60.

One idea: use the 2x2x2 Strassen algorithm as a building block.

But this is getting complex. Let me just implement the schoolbook and a simple random search for ranks 59, 58, ..., with a time limit.

Given the extreme difficulty of finding exact low-rank decompositions randomly, I'll focus on ensuring the schoolbook works and try a few simple structured improvements.

Actually, let me try one specific construction:
- For the 4 dimension, split into 2+2.
- Use Strassen for each 3x2x5 subproblem? No, Strassen is for square.

I'll just go with schoolbook and a minimal random search.
"""

import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to be safe within 120s
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank using random search
    # This is very unlikely to work, but let's try a few attempts
    for r in range(rank - 1, max(rank - 20, 1), -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations for this rank
        attempts = 5
        for _ in range(attempts):
            if time.time() > deadline - 30:
                break
            
            # Initialize with small random integer/half-integer values
            U_r = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n * m, r))
            V_r = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m * p, r))
            W_r = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p * n, r))
            
            # Check if the decomposition is exact
            # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        if best_rank == r:
            break
    
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
