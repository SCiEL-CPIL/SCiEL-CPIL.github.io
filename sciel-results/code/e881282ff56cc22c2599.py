"""Rewritten program: search for a lower-rank exact decomposition of the 3x4 * 4x5 matmul tensor.

We use the known structure of small matrix multiplication tensors. The schoolbook rank is 60.
For small dimensions, the true rank is often lower. We try a constructive approach using
the "Strassen-like" block decomposition and also a direct linear-algebraic search for a
low-rank decomposition over the rationals.

The key insight: the matmul tensor for (n,m,p) can sometimes be decomposed at rank less
than n*m*p. For 3x4*4x5, we can try to find a decomposition of rank r where r < 60.

Strategy:
1. First, try known small-rank decompositions via block structures.
2. Use a randomized / structured search: pick candidate ranks r from 50 down to some lower bound,
   and try to find U, V, W of shape (12,r), (20,r), (15,r) with entries in {0, 1/2, 1, -1/2, -1}
   such that the tensor identity holds.

The tensor identity: for all i in [0,3), j in [0,4), k in [0,5):
    sum_{c=0}^{r-1} U[i*4+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1
and for all (i,j,k) not corresponding to a valid matmul index combination... wait, actually
the matmul tensor T[i,j,k] = 1 if we think of it as the coefficient of x_i y_j z_k in the
product (sum_i x_i e_i)(sum_j e_j f_j)(sum_k f_k z_k)... 

Actually, the standard matmul tensor: T_{i,j,k} is the coefficient such that
    C_{i,k} = sum_j A_{i,j} B_{j,k}
can be written as:
    C_{i,k} = sum_c U_{i,j->c} V_{j,k->c} W_{k,i->c} ... 

Let me be precise. The matmul tensor T is a 3-index tensor T[i,j,k] where:
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}

In the bilinear/trilinear form, we write:
    (A*B)_{i,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  ... no, that's not right either.

The standard formulation: we want to compute C = A * B where A is n x m, B is m x p, C is n x p.
We write C_{i,k} = sum_j A_{i,j} B_{j,k}.

The tensor decomposition of rank r means:
    C_{i,k} = sum_{c=0}^{r-1} sum_j U_{i,j,c} A_{i,j} V_{j,k,c} B_{j,k} W_{k,i,c} ... 

Hmm, let me use the standard definition. The matmul tensor is a 3-tensor T in F^{n*m} x F^{m*p} x F^{n*p}
where T_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{j,k} delta_{k,k'}... 

Actually the cleanest way: the matmul tensor T is defined so that for matrices A (n x m), B (m x p):
    vec(C) = T * (vec(A) ⊗ vec(B))  ... no, that's not a tensor decomposition in the rank sense.

Let me use the standard definition from the task:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    and the decomposition satisfies: for all i in [0,n), j in [0,m), k in [0,p):
        sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    and for all other combinations of (i,j,k) that don't correspond to the matmul structure, the sum is 0.

Wait, actually the matmul tensor has non-zero entries only at specific indices. Let me think again.

The matmul tensor T is a 3-index tensor with indices (a, b, c) where:
    a = i*m + j  (i in [0,n), j in [0,m))  -> n*m indices
    b = j'*p + k  (j' in [0,m), k in [0,p))  -> m*p indices  
    c = k'*n + i'  (k' in [0,p), i' in [0,n))  -> p*n indices

And T[a, b, c] = 1 if and only if j = j' and k = k' and i = i', else 0.

So T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}

A rank-r decomposition means:
    T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]

So we need:
    sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t] = delta_{j,j'} delta_{k,k'} delta_{i,i'}

For the schoolbook, each term corresponds to a specific (i,j,k) triple, giving rank n*m*p.

Now, for 3x4*4x5, n=3, m=4, p=5, schoolbook rank = 60.

Let me try to find a lower rank. One approach: use the fact that the matmul tensor can sometimes be
decomposed using "Strassen-type" tricks. But for rectangular matrices, this is less well-studied.

Another approach: use the known lower bounds. The rank of the matmul tensor for (n,m,p) is at least
max(n*m, m*p, n*p) = max(12, 20, 15) = 20. And at most n*m*p = 60.

Let me try a direct search. For small r, we can set up a system of linear equations. But the entries
must be rational (or half-integer). 

Actually, let me try a different approach. I'll use the known decomposition for small cases.
For 2x2*2x2, Strassen gives rank 7 < 8.
For 3x3*3x3, the best known is 23 (Landsberg) or 22 (unconditional lower bound is 19).

For 3x4*4x5, I don't know the exact rank. Let me try to construct one.

Let me try a block-based approach. Write A as a 3x4 matrix, B as a 4x5 matrix.
Partition A into blocks and B into blocks, and use a Strassen-like recursion.

Actually, let me just try a computational search. I'll attempt to find a decomposition of rank r
for r from 55 down to 20, using a greedy / structured approach.

Given the 150s time limit, I need to be efficient. Let me try:
1. Start with the schoolbook decomposition (rank 60).
2. Try to merge terms: if two terms can be combined (i.e., their contributions are linearly dependent),
   reduce the rank.
3. Use a linear algebra approach: for a given rank r, set up the system and solve.

Actually, the most practical approach given the constraints: try a few known structured decompositions
and verify them. If none work, fall back to schoolbook.

Let me try a simple improvement: the "border rank" or "approximate" rank is lower, but we need exact.

I'll implement a search that tries to find a decomposition of rank r for decreasing r, using a
randomized approach with rational entries from a small set.

Given the complexity, let me just implement a careful search for r in a range, using numpy to
check validity, and return the best found.
"""

import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 12, 20, 15
    
    # Build the matmul tensor T as a 3D array T[a, b, c]
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # For a given rank r, we need U (nm x r), V (mp x r), W (pn x r) such that
    # sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c] for all a,b,c
    
    # We can fix U and V, then solve for W linearly, or use other approaches.
    # Let's try: fix U and V randomly, then for each c, solve for W[:,c] from the linear system.
    # Actually, it's easier to fix two of the matrices and solve for the third.
    
    # Approach: For a given r, fix U and V with random entries from a small set,
    # then the system is: for each (a,b), sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c]
    # This is a system in W. For each c, we have nm*mp equations in r unknowns W[:,c].
    # But W has pn rows and r columns, so for each c, we have a system of nm*mp equations in r unknowns.
    # This is overdetermined unless r is large enough.
    
    # Better approach: use the fact that the tensor is sparse. Most T[a,b,c] are 0.
    # Only n*m*p = 60 entries are 1, the rest are 0.
    
    # Let me try a different strategy: use the "rank-1" structure.
    # Each rank-1 term is u * v * w (outer product). We want to express T as a sum of r such terms.
    
    # Given the time constraint, let me try a few structured decompositions.
    
    # Try rank 50, 45, 40, ... down to 25, with a limited number of random trials per rank.
    candidate_ranks = [55, 50, 45, 40, 35, 30]
    
    for r in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        max_trials = 50
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random U, V with entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(nm, r)).astype(float)
            V = np.random.randint(-1, 2, size=(mp, r)).astype(float)
            
            # For each c in [0, pn), solve for W[:, c]
            # The equation is: sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c]
            # Let M[a,b,t] = U[a,t] V[b,t]. Then for each c:
            #   sum_t M[a,b,t] W[c,t] = T[a,b,c] for all a,b
            # This is a linear system: M_flat @ W[:,c] = T_flat[:,c]
            # where M_flat is (nm*mp, r) and T_flat[:,c] is (nm*mp,)
            
            M = np.einsum('at,bt->abt', U, V)  # (nm, mp, r)
            M_flat = M.reshape(nm * mp, r)
            
            W = np.zeros((pn, r))
            valid = True
            for c in range(pn):
                T_col = T[:, :, c].reshape(-1)
                # Solve M_flat @ w = T_col
                try:
                    w, residuals, rank_m, sv = np.linalg.lstsq(M_flat, T_col, rcond=None)
                    W[c, :] = w
                    # Check if the solution is good
                    residual = np.linalg.norm(M_flat @ w - T_col)
                    if residual > 1e-6:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Verify the full decomposition
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                if np.allclose(recon, T, atol=1e-6):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
