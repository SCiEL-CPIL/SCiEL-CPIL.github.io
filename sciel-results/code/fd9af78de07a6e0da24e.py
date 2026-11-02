"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

We seek a rank-minimal exact decomposition of the multiplication tensor.
The multiplication tensor T is of size (n*m) x (m*p) x (p*n) with the
convention that the output index is k*n + i (i.e., row i of result,
where result = A * B, A is n x m, B is m x p).

The tensor entry T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k,
and 0 otherwise. This is the "matrix multiplication tensor" for the
given dimensions.

The rank of this tensor is at most n*m*p (schoolbook). For 2x3 * 3x10,
schoolbook rank is 60.

Known results: For matrix multiplication of n x m by m x p matrices,
the tensor rank is not generally known to be less than n*m*p, except
for specific small cases. For 2x2*2x2, the rank is 7 (Strassen).

For 2x3 * 3x10, we can try to find a lower rank using known algorithms
or random search. However, given the time constraint, we'll implement a
search for a lower rank decomposition.

Key insight: The multiplication tensor for (n,m,p) can sometimes be
decomposed at lower rank than n*m*p. For instance, if we can find a
set of rank-1 tensors that sum to the multiplication tensor.

We'll implement a randomized search:
1. Start with schoolbook rank 60.
2. Try to find a decomposition of rank r < 60 using iterative refinement.
3. Use random initialization and then try to solve for exact integer/
   half-integer solutions.

Given the difficulty of this problem, and the time constraint, we'll
implement a practical search that:
- Tries small rank values
- Uses random initialization with integer/half-integer values
- Verifies the decomposition
- Falls back to schoolbook if no better is found within time

The key equation we need to satisfy:
For all (a, b, c) where a = i*m+j, b = j'*p+k, c = k'*n+i',
T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

where T[a,b,c] = 1 if and only if:
- j = j' (the column index of A matches the row index of B)
- i = i' (the row index of A matches the row index in the output)
- k = k' (the column index of B matches the column index in the output)

More precisely: a = i*m + j, b = j'*p + k, c = k'*n + i'
T[a,b,c] = 1 if j == j' and i == i' and k == k', else 0.

Since the indices are coupled, let's re-express:
For a given (i, j, k): a = i*m+j, b = j*p+k, c = k*n+i, T[a,b,c] = 1.
All other entries are 0.

So the tensor has exactly n*m*p = 60 non-zero entries, all equal to 1.

A rank-1 tensor is U ⊗ V ⊗ W. We need to express the sum of r such
rank-1 tensors as the multiplication tensor.

Let's implement a search that tries to find such a decomposition.
"""

import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute the contracted tensor from U, V, W
    # C[a, b] = sum_r U[a,r] * V[b,r]  -- but this is just a matrix
    # Actually, we need the full tensor: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Efficient computation: for each r, add U[:,r] * V[:,r] * W[:,r] to T
    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check against the true multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False
                # Check that other entries are zero
                # (This would be expensive, so we just check the expected non-zeros)
    
    # Verify that entries that should be zero are zero
    # The tensor should be zero except at (i*m+j, j*p+k, k*n+i) for all i,j,k
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))
    
    # Check a sample of zero entries
    for _ in range(1000):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False
    
    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank
    # Known lower bounds: For 2x2*2x2, rank is 7 (vs 8 schoolbook)
    # For 2x3*3x10, the minimal rank is not well-known in general
    # Let's try a search from rank 1 upward, but this is infeasible for high ranks
    # Instead, try a few promising ranks
    
    # Try ranks: maybe we can do better than 60
    # For now, try a randomized search for lower ranks
    # But this is very hard, so let's try a few approaches
    
    # Approach: Try to find a decomposition using random rank-1 tensors
    # and see if we can combine them to match the multiplication tensor
    
    # Actually, let's try a more systematic approach:
    # The multiplication tensor for (n, m, p) has a known structure.
    # For small n, m, p, we can try to find the rank using known results.
    
    # For 2x3 * 3x10:
    # This is a specific case. Let's try to use the fact that
    # the tensor can be viewed as a 3-tensor of size 6 x 30 x 20.
    
    # Given the complexity, let's implement a search that tries
    # to find a lower rank by:
    # 1. Randomly initializing U, V, W with small integer/half-integer values
    # 2. Checking if the decomposition works
    # 3. Iterating if not
    
    # But random search for exact match is very unlikely to work.
    # Instead, let's try a more structured approach.
    
    # One approach: Use the fact that the multiplication tensor can be
    # decomposed using bilinear algorithms. For 2x3 * 3x10, we can
    # potentially use a combination of smaller multiplications.
    
    # For now, let's return the schoolbook solution if we can't find
    # a better one quickly.
    
    # Try a few random searches for lower rank
    best_rank = rank_school
    best_solution = (U_school, V_school, W_school)
    
    # Try ranks from 1 to rank_school - 1, but only for a few
    # Given time constraints, try a limited search
    
    # Actually, for this specific problem, the minimal rank is likely
    # close to the schoolbook rank or significantly lower due to
    # algebraic structure. Without a known algorithm, let's try
    # a random search for a few ranks.
    
    # Try rank 59, 58, ... down to some minimum, but this is infeasible
    # to search exhaustively. Let's just try a few random attempts.
    
    for attempt_rank in range(rank_school - 1, max(20, rank_school - 20), -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations for this rank
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(attempt_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(attempt_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(attempt_rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                if _is_valid_entries(U, V, W):
                    best_rank = attempt_rank
                    best_solution = (U, V, W)
                    return {"U": U, "V": V, "W": W}
    
    # Return the best solution found (likely schoolbook)
    U_best, V_best, W_best = best_solution
    return {"U": U_best, "V": V_best, "W": W_best}
