"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for rank-minimal exact tensor decomposition
of 2x3 * 3x7 matrix multiplication. Uses structured families + exact verification with
a hard deadline guard. Falls back to schoolbook rank 42."""

import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = build_schoolbook()

    # Verification: given U, V, W as lists of lists (rows x rank), check exact decomposition.
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank.
    # Check that for all i,j,k: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    def verify(U, V, W):
        """U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank.
        Check that for all i,j,k: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    uidx = i * m + j
                    vidx = j * p + k
                    widx = k * n + i
                    Urow = U[uidx]
                    Vrow = V[vidx]
                    Wrow = W[widx]
                    for c in range(rank):
                        s += Urow[c] * Vrow[c] * Wrow[c]
                    if s != 1:
                        return False
        return True

    # Try to find a smaller rank using a structured family.
    # We'll use a systematic approach: for each candidate rank r from lower_bound to schoolbook_rank-1,
    # try to find a valid decomposition by solving a linear system.
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(NM, MP, PN)
    
    # For a given rank r, we need to find U (NM x r), V (MP x r), W (PN x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a bilinear problem. We can try to fix one matrix and solve for the others.
    # Strategy: fix U to be a random matrix with entries from {-1, 0, 1} or {-0.5, 0, 0.5, 1},
    # then for each (i,j,k), the equation becomes:
    # sum_c V[j*p+k, c] * W[k*n+i, c] = 1 / U[i*m+j, c] if U[i*m+j, c] != 0
    # This is still nonlinear.
    
    # Alternative: use the fact that the tensor is all-ones. Try structured decompositions.
    
    # Key insight: For matrix multiplication A (n x m) * B (m x p) = C (n x p),
    # the bilinear complexity is the rank of the tensor M[i,j,k] = 1.
    
    # Known results: For 2x3x7, the optimal rank is not known to be below 42 in general,
    # but let's try to find improvements using a systematic search.
    
    # Approach: Try ranks from 41 down to lower_bound. For each rank r, try several
    # structured initializations and use a local search to refine.
    
    def try_rank(r, max_attempts=50):
        """Try to find a decomposition of rank r."""
        if r < lower_bound:
            return None
        
        # Try several random initializations
        for attempt in range(max_attempts):
            # Initialize U, V, W with random values from a small set
            # Use {-1, 0, 1} or {-0.5, 0, 0.5, 1} to keep entries simple
            val_set = [-1, -0.5, 0, 0.5, 1]
            
            U = [[random.choice(val_set) for _ in range(r)] for _ in range(NM)]
            V = [[random.choice(val_set) for _ in range(r)] for _ in range(MP)]
            W = [[random.choice(val_set) for _ in range(r)] for _ in range(PN)]
            
            # Check if this random initialization works
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
            
            # If not, try to improve using a local search
            # For each (i,j,k), compute the current sum and try to adjust entries
            # to make it equal to 1.
            
            # This is complex. Instead, let's try a different strategy:
            # Use the structure of the problem. For each i, the slice T[i,:,:] is an
            # m x p matrix of all ones, which has rank 1. So we can try to decompose
            # using n terms, one for each i, where each term handles one i-slice.
            
            if attempt == 0:
                # Try the "i-slice" decomposition
                U2 = [[0] * r for _ in range(NM)]
                V2 = [[0] * r for _ in range(MP)]
                W2 = [[0] * r for _ in range(PN)]
                
                # For each i, assign r_i = r // n terms
                terms_per_i = r // n
                remaining = r % n
                
                col = 0
                for i in range(n):
                    terms_for_i = terms_per_i + (1 if i < remaining else 0)
                    for t in range(terms_for_i):
                        c = col
                        # Set U[i*m+j, c] = 1 for all j
                        for j in range(m):
                            U2[i * m + j][c] = 1
                        # Set V[j*p+k, c] = 1 for all j,k
                        for j in range(m):
                            for k in range(p):
                                V2[j * p + k][c] = 1
                        # Set W[k*n+i, c] = 1 for all k
                        for k in range(p):
                            W2[k * n + i][c] = 1
                        col += 1
                
                # This gives rank n * (terms_per_i) + remaining = r
                # But this only works if r >= n. For r = n, each i gets 1 term.
                # Let's verify this structure.
                if verify(U2, V2, W2):
                    return {"U": U2, "V": V2, "W": W2}
        
        return None

    # Try ranks from 41 down to lower_bound
    for r in range(schoolbook_rank - 1, lower_bound - 1, -1):
        if time.time() - t_start > DEADLINE - 5:
            break
        
        result = try_rank(r, max_attempts=20)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END
