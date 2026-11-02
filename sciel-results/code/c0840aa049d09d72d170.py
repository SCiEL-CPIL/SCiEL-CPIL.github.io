"""
Matrix multiplication tensor decomposition for 2x5 * 5x5.
We seek a rank-minimal exact tensor decomposition.
The schoolbook rank is 2*5*5 = 50.
We try to find a lower rank using a structured approach.

For small matrices, the multiplication tensor T_{i,j,k} = delta_{i,j,k} in the sense that
C[i,k] = sum_j A[i,j] * B[j,k].
The tensor has entries T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j] and B[j,k], which is always 1 for valid i,j,k.
Wait, actually the multiplication tensor is a trilinear form:
C[i,k] = sum_{j} A[i,j] B[j,k]
So the tensor T has T[i,j,k] = 1 for all i,j,k (since every combination contributes to some C[i,k] via the sum over j? No, that's not right).

Actually, the tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p) is:
T[i, j, k] = 1 if we think of it as: the (i,k) entry is sum over j of A[i,j] * B[j,k].
But the tensor decomposition wants: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k} * (sum_j ...) -- no.

Let me reconsider. The matrix multiplication tensor is a 3-tensor with indices (i, j, k) where:
- i indexes the row of A and C (size n)
- j indexes the column of A and row of B (size m)
- k indexes the column of B and C (size p)

The tensor entry T[i,j,k] = 1 for all valid i,j,k, because the multiplication is:
C[i,k] = sum_j A[i,j] * B[j,k]

But in tensor decomposition, we want to write the multiplication as:
sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = sum_j A[i,j] B[j,k]

This is not quite right either. The standard tensor decomposition for matrix multiplication uses:
We want to find matrices U (size n*m x r), V (size m*p x r), W (size p*n x r) such that:
sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1 for all i, j, k

Wait, that would mean the tensor T[i,j,k] = 1 for all i,j,k. But that's not the multiplication tensor.

Let me re-read the problem. "exactly decompose the matrix multiplication tensor"

The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix has the property that:
The (i, k) entry of the result is sum_j A[i,j] * B[j,k].

In tensor form, this is a trilinear form. The standard way to express this as a tensor decomposition is:
We want: sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = sum_j A[i,j] B[j,k]

But actually, the tensor T is defined such that T[i,j,k] = 1 for all i,j,k in the "identity" sense? No.

I think the correct interpretation is:
The multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k, and we want to find a rank-r decomposition such that:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k

This is because the matrix multiplication can be written as a sum of rank-1 tensors where each term corresponds to one (i,j,k) triple.

Actually, I think the tensor is: T[i,j,k] = delta_{i,j,k} in the sense that each "term" in the sum corresponds to a unique (i,j,k). The schoolbook decomposition has rank n*m*p with each term being a single (i,j,k).

So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p).

This is a system of n*m*p equations. We want to minimize r (the number of terms).

For n=2, m=5, p=5, we have 50 equations. The schoolbook has r=50.

We can try to find a lower rank. One approach is to use the fact that for small matrices, there might be structured decompositions.

Let me try a different approach: use linear algebra. We have 50 linear equations in r*10 variables (since U has 10 rows, V has 25 rows, W has 10 rows, each with r columns). Wait, no: U is 10 x r, V is 25 x r, W is 10 x r. So total variables = 10r + 25r + 10r = 45r. We have 50 equations. So for r >= 2, we have more variables than equations (45*2 = 90 > 50), so there should be solutions. But we need integer/half-integer solutions.

Let me try r=1: 45 variables, 50 equations. Likely no solution.
r=2: 90 variables, 50 equations. Should be solvable.

But we need the entries to be integer or half-integer.

Let me try to construct a solution for small r. I'll use a randomized approach with rounding.

Actually, let me think about this differently. The problem is equivalent to finding a tensor of rank r that matches the multiplication tensor. For 2x5x5, the known lower bounds might be higher than 50/2.

Let me just try to find any valid decomposition with rank less than 50 using a numerical method and then round to nearest half-integer.
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 50
    
    # Start with schoolbook as fallback
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    start_time = time.time()
    deadline = 140  # leave 10s margin
    
    # Try to find lower rank decompositions
    # The equations are: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # We can fix one matrix and solve for others, or use iterative methods.
    
    # Let's try a simple approach: assume U and V are sparse/structured, solve for W
    # Or use a grid search for small rank values.
    
    # Try ranks from 20 up to 49
    best_rank = rank_schoolbook
    best_decomp = (best_U, best_V, best_W)
    
    for r in range(20, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Iteratively update each matrix
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U: for each row i*m+j, we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a system of p equations in r unknowns for each (i,j)
                for i in range(n):
                    for j in range(m):
                        # Build system: sum_c U[row,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for k=0..p-1
                        # Let B[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        # Then sum_c U[row,c] * B[k,c] = 1
                        B = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                B[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve B @ U[row] = ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row j*p+k, sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                B[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row k*n+i, sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        B = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                B[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check if we have a valid solution
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_rounded[i * m + j, c] * V_rounded[j * p + k, c] * W_rounded[k * n + i, c]
                        if abs(s - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to list of lists
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                # Round to exact half-integers
                U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                # Double check
                valid2 = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                            if abs(s - 1) > 1e-9:
                                valid2 = False
                                break
                        if not valid2:
                            break
                    if not valid2:
                        break
                if valid2:
                    best_rank = r
                    best_decomp = (U_list, V_list, W_list)
                    break  # found for this rank, try lower
            
            if best_rank < rank_schoolbook:
                break  # no need to try higher ranks
    
    if best_rank < rank_schoolbook:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
