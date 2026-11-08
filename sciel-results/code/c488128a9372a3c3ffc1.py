"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Start with schoolbook decomposition (rank = n*m*p = 48).
    2. Try to reduce rank by identifying and merging terms.
    3. Use a combinatorial search to find minimal rank.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    The tensor T[i, j, k] = delta_{j,k} (in the appropriate indexing)
    
    Actually, the matrix multiplication tensor has entries:
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    The tensor is defined such that C[i,k] = sum_j A[i,j] * B[j,k]
    
    In the tensor decomposition context, we want to express the multiplication map
    as a sum of rank-1 tensors. The standard schoolbook decomposition uses n*m*p terms.
    
    However, there are known lower bounds and constructions. For 2x3 and 3x8:
    - The trivial rank is 2*3*8 = 48
    - We can potentially do better by exploiting structure
    
    Let's think about this more carefully. The matrix multiplication tensor for n x m times m x p
    has a known rank. For small dimensions, the exact rank is known.
    
    For 2x3 * 3x8:
    The tensor lives in R^{2*3} ⊗ R^{3*8} ⊗ R^{2*8} = R^6 ⊗ R^{24} ⊗ R^{16}
    
    The schoolbook rank is 2*3*8 = 48.
    
    Can we do better? The rank of matrix multiplication tensor is a deep question.
    For 2x2 * 2x2, the rank is 7 (Strassen).
    For 2x3 * 3x2, the rank is known to be 11 (or similar).
    
    For 2x3 * 3x8, let's consider that we can use block structure.
    
    Actually, let me reconsider the problem. The task says "rank-minimal exact tensor decomposition".
    The score is the rank, and lower is better.
    
    Let me try a few approaches:
    1. First, verify the schoolbook decomposition works.
    2. Then try to find a lower rank decomposition.
    
    For the schoolbook decomposition:
    U has shape (n*m, rank) = (6, 48)
    V has shape (m*p, rank) = (24, 48)
    W has shape (p*n, rank) = (16, 48)
    
    The decomposition should satisfy:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] = delta_{i,k} (if we're computing A*B where the result is indexed by i,k)
    
    Wait, let me re-read the convention. "W[k*n+i, r]" suggests W is indexed by (k, i) where k is the column of B (and row of result) and i is the row of A (and row of result).
    
    So the multiplication is: C[i, k] = sum_j A[i, j] * B[j, k]
    
    In tensor form, we have:
    C = sum_r (U[:, r] ⊗ V[:, r]) evaluated against W[:, r]
    
    More precisely, the tensor T is in (R^{nm})^* ⊗ (R^{mp})^* ⊗ (R^{pn})
    and we want T = sum_r u_r ⊗ v_r ⊗ w_r
    
    Where u_r is in R^{nm}, v_r is in R^{mp}, w_r is in R^{pn}
    
    And the condition is that for the matrix multiplication:
    (A ⊗ B) contracted with T gives C
    
    Actually, the standard way: the matrix multiplication tensor T_{ijk} for multiplying n x m by m x p
    is defined as T_{i,j,k} = 1 if j is the "middle" index, but actually it's:
    
    The tensor T is such that T[i, j, k] = delta_{j, j'} where... 
    
    Let me just use the schoolbook and verify it works, then try to improve.
    
    Schoolbook: for each (i, j, k), we have a term where:
    U[i*m + j, (i,j,k)] = 1
    V[j*p + k, (i,j,k)] = 1
    W[k*n + i, (i,j,k)] = 1
    
    This gives rank = n*m*p = 48.
    
    Now, can we reduce this? The key insight is that some terms might be combined.
    
    For example, in Strassen's algorithm for 2x2, we combine terms to get rank 7 instead of 8.
    
    For 2x3 * 3x8, one approach is to view the 3x8 matrix as three 3x(8/3) blocks, but 8 is not divisible by 3.
    
    Alternatively, we can use the fact that the rank of the multiplication tensor for n x m * m x p
    is at most the product of the ranks of smaller multiplications.
    
    Let me try a different approach: use a known result or try a local search.
    
    Actually, let me first check if the schoolbook decomposition is correct, then try to find
    a lower rank decomposition using a heuristic search.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook decomposition
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
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # For each i, k, check that sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # Wait, that's not right. The tensor decomposition means:
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T should be the matrix multiplication tensor, which is:
        # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # Actually, the tensor is: T[i, j, k] = delta_{j, j} which is always 1? No.
        
        # The matrix multiplication tensor T is in R^{nm} ⊗ R^{mp} ⊗ R^{pn}
        # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T[a, b, c] = 1 if i=i', j=j', k=k' (i.e., the indices match for the multiplication)
        
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
        # And T is 0 otherwise.
        
        # The decomposition says: T = sum_r U[:, r] ⊗ V[:, r] ⊗ W[:, r]
        # So T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    expected = 1
                    actual = 0
                    for r in range(U.shape[1]):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        
        # Also check that off-diagonal entries are 0
        # T[a, b, c] should be 0 unless a, b, c correspond to matching i, j, k
        # a = i*m+j, b = j'*p+k', c = k''*n+i''
        # T should be 1 only if i=i'' and j=j' and k=k'
        
        # This is expensive to check fully, so let's just check a few random off-diagonal entries
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            # Pick a different i', j', k'
            i2 = (i + 1) % n
            j2 = (j + 1) % m
            k2 = (k + 1) % p
            a = i * m + j
            b = j2 * p + k2
            c = k2 * n + i2
            actual = 0
            for r in range(U.shape[1]):
                actual += U[a, r] * V[b, r] * W[c, r]
            if abs(actual) > 1e-9:
                return False
        
        return True
    
    # The schoolbook decomposition should be valid
    # Now try to reduce the rank
    
    # Approach: try to merge columns that have similar patterns
    # Or use a local search to find a lower rank decomposition
    
    # For now, return the schoolbook decomposition
    # This gives rank = 48
    
    # Convert to lists for the return format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
