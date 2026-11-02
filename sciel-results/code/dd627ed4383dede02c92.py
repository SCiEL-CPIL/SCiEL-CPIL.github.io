"""Matrix multiplication 2,4,5 - rank-minimal tensor decomposition search."""
DIMENSIONS = [2, 4, 5]
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    # Check orthogonality: for non-matching (i,j,k), sum should be 0
    # Actually the tensor T[i,j,k] = 1 only when indices match in the product
    # The condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{correct}
    # For matrix multiplication, T[i,j,k] = 1 always (it's the multiplication tensor)
    # Wait - the multiplication tensor has T[i,j,k] = 1 for all i,j,k in the 2x4x5 case
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is all-ones? No.
    # The tensor is: T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k. That means we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for ALL i,j,k
    # And that's it - there are no "zero" entries to check.
    # Actually wait, that can't be right for a rank reduction. Let me reconsider.
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # So we need the decomposition to reproduce all-ones.
    # The schoolbook gives rank n*m*p with each term picking out one (i,j,k).
    # For lower rank, we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification: check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-10):
    """Fast float verification."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_solve_for_W(U, V, n, m, p, rank):
    """Given U and V, solve for W such that the decomposition holds.
    Returns W if found (as list of lists of Fractions), else None."""
    # For each (i,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
    # This is: for fixed (i,k), the vector W[k*n+i, :] must satisfy
    # sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) ... no wait.
    # Actually: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Group by (i,k): for each j, sum_c [U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c] = 1
    # So for fixed (i,k), define a_{j,c} = U[i*m+j,c] * V[j*p+k,c]
    # Then we need: for each j, sum_c a_{j,c} * W[k*n+i,c] = 1
    # This is a linear system: A_{ik} * w_{ik} = 1_vector, where A_{ik} is m x rank
    # We need to solve this for each (i,k) pair.
    
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            # Build m x rank matrix A where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
            A = [[Fraction(0)] * rank for _ in range(m)]
            for j in range(m):
                for c in range(rank):
                    A[j][c] = Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c])
            # Solve A * w = [1,1,...,1] (m equations, rank unknowns)
            # Use Gaussian elimination
            # Augmented matrix: m x (rank+1)
            aug = [[A[j][c] for c in range(rank)] + [Fraction(1)] for j in range(m)]
            # If m > rank, overdetermined - need least squares or check consistency
            # If m <= rank, underdetermined - pick a solution
            # Gaussian elimination on augmented
            n_eq = m
            n_var = rank
            # Make copy
            for col in range(min(n_eq, n_var)):
                # Find pivot
                pivot_row = -1
                for row in range(col, n_eq):
                    if aug[row][col] != 0:
                        pivot_row = row
                        break
                if pivot_row == -1:
                    # Check if consistent
                    for row in range(col, n_eq):
                        if aug[row][n_var] != 0:
                            return None  # Inconsistent
                    continue
                # Swap
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
                # Normalize
                pv = aug[col][col]
                for c2 in range(col, n_var + 1):
                    aug[col][c2] /= pv
                # Eliminate
                for row in range(n_eq):
                    if row != col and aug[row][col] != 0:
                        factor = aug[row][col]
                        for c2 in range(col, n_var + 1):
                            aug[row][c2] -= factor * aug[col][c2]
            # Check consistency for remaining rows
            for row in range(n_eq):
                all_zero = True
                for c2 in range(n_var):
                    if aug[row][c2] != 0:
                        all_zero = False
                        break
                if all_zero and aug[row][n_var] != 0:
                    return None
            # Extract solution (set free variables to 0)
            w = [Fraction(0)] * n_var
            for col in range(min(n_eq, n_var)):
                w[col] = aug[col][n_var]
            # Store in W
            for c in range(rank):
                W[k * n + i][c] = w[c]
    
    return W

def _generate_structured_UV(n, m, p, rank, seed):
    """Generate structured U and V matrices."""
    rng = random.Random(seed)
    
    # Strategy 1: Random sparse
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    
    for row in range(n * m):
        # Each row of U has ~rank/(n*m) non-zeros on average
        for c in range(rank):
            if rng.random() < 0.3:
                U[row][c] = Fraction(rng.choice([-1, 1]))
    
    for row in range(m * p):
        for c in range(rank):
            if rng.random() < 0.3:
                V[row][c] = Fraction(rng.choice([-1, 1]))
    
    return U, V

def _generate_dense_UV(n, m, p, rank, seed):
    """Generate dense random U and V with small integer entries."""
    rng = random.Random(seed)
    U = [[Fraction(rng.randint(-2, 2)) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(rng.randint(-2, 2)) for _ in range(rank)] for _ in range(m * p)]
    return U, V

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for col in range(best_rank):
        i = col // (m * p)
        j = (col // p) % m
        k = col % p
        best_U[i * m + j][col] = 1
        best_V[j * p + k][col] = 1
        best_W[k * n + i][col] = 1
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Also need rank >= max(n, m, p) for basic reasons
    min_rank = max(n * m, m * p, p * n)
    
    seed = 0
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        attempts = 0
        max_attempts = 50  # Limit attempts per rank
        
        while attempts < max_attempts and time.time() < deadline:
            seed += 1
            attempts += 1
            
            if seed % 2 == 0:
                U, V = _generate_structured_UV(n, m, p, rank, seed)
            else:
                U, V = _generate_dense_UV(n, m, p, rank, seed)
            
            # Solve for W
            W = _try_solve_for_W(U, V, n, m, p, rank)
            if W is not None:
                # Verify
                if _verify_full(U, V, W, n, m, p):
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    # Found a lower rank, restart search from here
                    break
        
        if best_rank < schoolbook_rank:
            # Found improvement, continue searching for even lower
            min_rank = best_rank - 5  # Look a bit below
            min_rank = max(min_rank, max(n*m, m*p, p*n))
    
    # Try to reduce further with more sophisticated methods
    if time.time() < deadline - 10 and best_rank > min_rank:
        # Try null-space projection on the best found solution
        for rank_try in range(max(min_rank, best_rank - 10), best_rank):
            if time.time() > deadline:
                break
            
            for attempt in range(20):
                if time.time() > deadline:
                    break
                seed += 1
                U, V = _generate_dense_UV(n, m, p, rank_try, seed)
                W = _try_solve_for_W(U, V, n, m, p, rank_try)
                if W is not None and _verify_full(U, V, W, n, m, p):
                    best_rank = rank_try
                    best_U = U
                    best_V = V
                    best_W = W
                    break
    
    # Convert to int where possible
    def to_int_or_frac(x):
        if x.denominator == 1:
            return x.numerator
        return x
    
    U_out = [[to_int_or_frac(x) for x in row] for row in best_U]
    V_out = [[to_int_or_frac(x) for x in row] for row in best_V]
    W_out = [[to_int_or_frac(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
