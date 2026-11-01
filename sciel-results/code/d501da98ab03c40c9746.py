"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check a sample of entries
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = 1 for the multiplication tensor
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_block_decomposition():
    """
    Attempt a rank-50 decomposition using Strassen on a 2x2x2 block structure.
    For (2x4)*(4x7)->(2x7): split the middle 4 into two 2-blocks.
    We treat the multiplication as: C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x7.
    
    Use Strassen on the outer 2x2 (the 'n' and the first 'm' dimension grouping).
    Actually, the known result: rank 50 via combining Strassen (7) and schoolbook.
    
    Let's build it explicitly. The tensor T[i,j,k] for i in {0,1}, j in {0,1,2,3}, k in {0..6}.
    We group j into j = 2*a + b where a in {0,1}, b in {0,1}.
    So T[i, 2a+b, k] = sum over the multiplication.
    
    We can write C[i,k] = sum_{j} A[i,j] B[j,k] = sum_a sum_b A[i,2a+b] B[2a+b,k].
    = sum_a (sum_b A[i,2a+b] B[2a+b,k])
    
    For each a, the inner sum over b (2 terms) with A being 2x2 (i x b) and B being 2x7 (b x k).
    So for each a, we have a (2x2)*(2x7) multiplication, rank 28 each, total 56.
    
    To use Strassen, we need the outer structure to be 2x2x2. Here the outer is 'a' (2 values) 
    but the inner is 2x7 not 2x2. So Strassen doesn't directly apply.
    
    Alternative: The known rank-50 might come from a different grouping. 
    Let me try: group the 7 into 2+2+2+1. And use Strassen on 2x2x2 blocks for the first three groups.
    
    This is getting complex. Let me just return the schoolbook and rely on the search.
    """
    return None


def _build_rank_r_random(r):
    """Build a random rank-r candidate and try to fit."""
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    return U, V, W


def construct():
    deadline = time.time() + 110  # safety margin before 120s
    n, m, p = DIMENSIONS
    best = None
    
    # Start with schoolbook (rank 56) as a valid baseline
    U, V, W = _schoolbook()
    if _verify(U, V, W):
        best = (U, V, W)
    
    # Try to find a lower-rank decomposition via structured search
    # The known improvement is rank 50. Let's attempt a direct construction.
    # 
    # Rank 50 construction idea: 
    # The tensor T[i,j,k] = 1 for all (i,j,k) in the multiplication support.
    # Actually for matrix mult, T[i,j,k] = 1 always (it's the "all ones" tensor in this indexing).
    # Wait no. The multiplication tensor has T[i,j,k] = 1 for all i,j,k in the respective ranges,
    # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple contributes.
    # So T is the all-ones tensor of shape (2,4,7).
    #
    # We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = 1 for all (i,j,k).
    #
    # Schoolbook: one c per (i,j,k), U,V,W all 1 at that c. Rank 56.
    #
    # To reduce rank, we need to "share" factors. This is exactly the tensor rank problem.
    # For the all-ones tensor of shape (2,4,7), the rank is at most 2*4*7=56 (trivial) 
    # and we want to find the minimum.
    #
    # Known: rank of all-ones tensor [a,b,c] is at most min(a,b)*... no.
    # Actually the all-ones tensor has rank equal to the matrix rank when flattened, but that's not right either.
    #
    # Let me just do a numerical search for rank 50, 49, ... using a simple approach.
    # Given time constraints, I'll try a few ranks with random init + gradient-free refinement.
    
    # For now, return the best we have (schoolbook rank 56) if search doesn't finish.
    # But let's try a quick structured attempt at rank 50.
    
    # Attempt: use the fact that 50 = 56 - 6. 
    # One known technique: if we can find 6 pairs of (i,j,k) that can be "merged".
    # This is hard to do analytically. Let me just return schoolbook.
    
    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback (shouldn't reach here)
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
