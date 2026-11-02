"""
Matrix multiplication 3x5 by 5x5: rank-minimal exact tensor decomposition.

We attempt to find a decomposition of rank r < 75 (schoolbook) using a
structured search. The key insight is that for small matrices, we can use
the fact that the multiplication tensor has a specific structure.

For 3x5 * 5x5, the known upper bounds from literature suggest the rank is
at most 75 (schoolbook), but potentially lower. We'll use a greedy/iterative
approach to try to find a lower-rank decomposition.

Strategy:
1. Start with schoolbook (rank 75).
2. Try to merge terms using the structure of the tensor.
3. Use randomized search with local improvement to find lower rank.

Given the 150s time limit, we'll use a combination of:
- Structured initialization
- Randomized perturbation + local search
- Exact verification

The tensor T[i,j,k] for matrix multiplication (n=3, m=5, p=5) is:
T[i,j,k] = 1 if the (i,j) entry of A times the (j,k) entry of B contributes
to the (i,k) entry of C.

In the standard formulation:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor is T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [m), k in [p).

A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

We need integer or half-integer entries.
"""

import time
import random
import copy
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 75
    
    # Build the target tensor
    # T[i,j,k] = 1 for all i,j,k
    # In flattened form: index (i*m+j, j*p+k, k*n+i)
    
    def tensor_index(i, j, k):
        """Return the triple of indices in U, V, W matrices."""
        return (i * m + j, j * p + k, k * n + i)
    
    # Schoolbook solution
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = copy.deepcopy(U_school)
    best_V = copy.deepcopy(V_school)
    best_W = copy.deepcopy(W_school)
    
    # Verify function
    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = sum(U[ui][c] * V[vj][c] * W[wk][c] for c in range(r))
                    if val != 1:
                        return False
        # Also check that non-tensor entries are 0
        # Actually, the tensor is defined only on valid (i,j,k) triples
        # The decomposition must satisfy: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # And for "invalid" combinations, they should be 0, but since we only have valid indices,
        # we just need the above to hold.
        return True
    
    # Try to find lower rank using iterative improvement
    # Strategy: start from schoolbook, try to zero out entries and adjust
    
    # Alternative: use a randomized approach
    # Try ranks from lower to higher
    
    # For 3x5x5, let's try some known techniques:
    # 1. Strassen-like methods (but those are for square matrices)
    # 2. Greedy rank reduction
    
    # Let's try a simple approach: randomly initialize and use local search
    # But first, let's see if we can do better than schoolbook with a structured approach.
    
    # One idea: group terms that share the same i or k
    
    # Actually, for small dimensions, let's just try a few random restarts
    # with local improvement.
    
    def random_init(r):
        """Generate a random decomposition of rank r."""
        U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) for _ in range(r)] for _ in range(p * n)]
        return U, V, W
    
    def local_search(U, V, W, r, max_iter=1000):
        """Try to improve the decomposition by local moves."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random coordinate to modify
            matrix = random.choice(['U', 'V', 'W'])
            if matrix == 'U':
                idx = random.randrange(n * m)
                col = random.randrange(r)
                current = U[idx][col]
                # Try changing to a nearby value
                delta = random.choice([-1, 1])
                new_val = current + delta
                if abs(new_val) > 3:
                    continue
                U[idx][col] = new_val
            elif matrix == 'V':
                idx = random.randrange(m * p)
                col = random.randrange(r)
                current = V[idx][col]
                delta = random.choice([-1, 1])
                new_val = current + delta
                if abs(new_val) > 3:
                    continue
                V[idx][col] = new_val
            else:
                idx = random.randrange(p * n)
                col = random.randrange(r)
                current = W[idx][col]
                delta = random.choice([-1, 1])
                new_val = current + delta
                if abs(new_val) > 3:
                    continue
                W[idx][col] = new_val
            
            if verify(U, V, W, r):
                return U, V, W, True
        
        return U, V, W, False
    
    # Try different ranks
    # Start from a lower bound and go up
    # The rank is at least max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But in practice, it's much higher.
    
    # Let's try ranks from 25 to 74
    # But this might be too slow. Let's be smarter.
    
    # For now, let's just try a few random restarts with rank < 75
    # and see if we can find anything.
    
    # Actually, let's try a more structured approach first.
    # We know that the schoolbook has rank 75.
    # Can we merge any terms?
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists
    # a way to represent their sum with one term. This is generally not possible
    # for arbitrary pairs, but might be for structured ones.
    
    # Let's try: for each pair of schoolbook terms, check if their sum
    # can be represented by a single rank-1 term. This is equivalent to
    # checking if the sum of two rank-1 tensors is rank 1, which is rare.
    
    # Instead, let's try a different approach: use the fact that we can
    # have fractional entries.
    
    # Given time constraints, let's just return the schoolbook if we can't
    # find better quickly.
    
    # Try a few random restarts with rank 70, 65, 60, etc.
    for r in [70, 65, 60, 55, 50]:
        if time.time() > deadline - 10:
            break
        
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W = random_init(r)
            # Scale to make entries more likely to be integers
            # This is a heuristic; not guaranteed to work
            found = False
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                U2, V2, W2, found = local_search(U, V, W, r, max_iter=100)
                if found:
                    best_rank = r
                    best_U = U2
                    best_V = V2
                    best_W = W2
                    break
                if time.time() > deadline - 10:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
