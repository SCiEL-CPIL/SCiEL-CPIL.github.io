"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

We seek a rank-minimal exact decomposition of the multiplication tensor.
The multiplication tensor T is of size (n*m) x (m*p) x (p*n) with the
convention that the output index is k*n + i (i.e., row i of result,
where result = A * B, A is n x m, B is m x p).

The tensor entry T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k,
and 0 otherwise.

The rank of this tensor is at most n*m*p (schoolbook). For 2x3 * 3x10,
schoolbook rank is 60.

Key insight: The multiplication tensor for (n,m,p) can sometimes be
decomposed at lower rank than n*m*p. For instance, if we can find a
set of rank-1 tensors that sum to the multiplication tensor.

We'll implement a search that tries to find such a decomposition.
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
    
    # Verify that entries that should be zero are zero
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


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _block_decomposition(n, m, p, rank):
    """
    Find a rank decomposition for the multiplication tensor of n x m * m x p.
    Uses random search with integer/half-integer entries.
    """
    # Random search
    values = [0, 0.5, 1, -0.5, -1]
    for trial in range(1000):
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p) and _is_valid_entries(U, V, W):
            return U, V, W
    
    return None


def _combine_decompositions(parts, n, m, p_total):
    """
    Combine multiple (n, m, p_i) decompositions into a single (n, m, sum(p_i)) decomposition.
    parts: list of (U_i, V_i, W_i, p_i)
    """
    n, m = n, m
    total_rank = sum(len(U[0]) for U, V, W, pi in parts)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p_total)]
    W = [[0] * total_rank for _ in range(p_total * n)]
    
    col_offset = 0
    k_offset = 0
    for U_i, V_i, W_i, p_i in parts:
        r_i = len(U_i[0])
        for a in range(n * m):
            for r in range(r_i):
                U[a][col_offset + r] = U_i[a][r]
        for b in range(m * p_total):
            # b corresponds to j*p_total + k
            j = b // p_total
            k = b % p_total
            if k < p_i:
                b_i = j * p_i + k
                for r in range(r_i):
                    V[b][col_offset + r] = V_i[b_i][r]
            # else: zero
        for c in range(p_total * n):
            # c corresponds to k_total * n + i
            k_total = c // n
            i = c % n
            if k_total < p_i:
                c_i = k_total * n + i
                for r in range(r_i):
                    W[c][col_offset + r] = W_i[c_i][r]
            # else: zero
        col_offset += r_i
        k_offset += p_i
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator
    
    # Try to find a lower rank decomposition
    # Strategy: decompose p=10 into smaller blocks
    # e.g., 10 = 5 + 5, or 10 = 3 + 3 + 4, etc.
    
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = len(U_school[0])
    best_solution = (U_school, V_school, W_school)
    
    # Try different block decompositions
    # For p=10, try splitting into blocks of size 2, 3, 4, 5
    block_sizes = [2, 3, 4, 5]
    
    # Try combinations of block sizes that sum to 10
    from itertools import combinations_with_replacement
    
    def find_combinations(target, max_size, min_size=1):
        """Find all combinations of block sizes that sum to target."""
        results = []
        def backtrack(remaining, current):
            if remaining == 0:
                if current:
                    results.append(tuple(sorted(current)))
                return
            for size in range(min_size, max_size + 1):
                if size <= remaining:
                    backtrack(remaining - size, current + [size])
        backtrack(target, [])
        # Remove duplicates
        seen = set()
        unique = []
        for combo in results:
            if combo not in seen:
                seen.add(combo)
                unique.append(combo)
        return unique
    
    # Try combinations of blocks that sum to 10
    all_combos = find_combinations(10, 5)
    
    # Sort by number of blocks (fewer blocks might be easier)
    all_combos.sort(key=lambda x: len(x))
    
    for combo in all_combos:
        if time.time() > deadline - 5:
            break
        
        # For each block size, find the best rank
        block_solutions = []
        feasible = True
        
        for p_i in combo:
            if time.time() > deadline - 5:
                feasible = False
                break
            
            # Try to find a decomposition for (n, m, p_i)
            # The schoolbook rank is n*m*p_i
            school_rank_i = n * m * p_i
            
            # Try ranks from some lower bound up to school_rank_i
            # For small p_i, the minimal rank might be known
            found = False
            for r in range(max(1, school_rank_i - 5), school_rank_i + 1):
                if time.time() > deadline - 5:
                    break
                
                result = _block_decomposition(n, m, p_i, r)
                if result is not None:
                    U_i, V_i, W_i = result
                    block_solutions.append((U_i, V_i, W_i, p_i))
                    found = True
                    break
            
            if not found:
                # Fall back to schoolbook for this block
                U_i, V_i, W_i = _build_schoolbook(n, m, p_i)
                block_solutions.append((U_i, V_i, W_i, p_i))
        
        if not feasible:
            continue
        
        # Combine the block solutions
        U_combined, V_combined, W_combined = _combine_decompositions(block_solutions, n, m, p)
        rank_combined = len(U_combined[0])
        
        # Verify the combined solution
        if _verify(U_combined, V_combined, W_combined, n, m, p) and _is_valid_entries(U_combined, V_combined, W_combined):
            if rank_combined < best_rank:
                best_rank = rank_combined
                best_solution = (U_combined, V_combined, W_combined)
    
    # Return the best solution found
    U_best, V_best, W_best = best_solution
    return {"U": U_best, "V": V_best, "W": W_best}
