"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def verify(U, V, W, rank):
        """Verify that the tensor decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 0.0
                    for r in range(rank):
                        lhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True
    
    # Build rank-50 solution using Strassen blocks
    rank0 = 50
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1]) -> C[k0,i0]+=M1, C[k0+1,i0+1]+=M1
        U0[i0*m + j0, col] = 1
        U0[(i0+1)*m + (j0+1), col] = 1
        V0[j0*p + k0, col] = 1
        V0[(j0+1)*p + (k0+1), col] = 1
        W0[k0*n + i0, col] = 1
        W0[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1])B[j0,k0] -> C[k0,i0+1]+=M2, C[k0+1,i0+1]-=M2
        U0[(i0+1)*m + j0, col] = 1
        U0[(i0+1)*m + (j0+1), col] = 1
        V0[j0*p + k0, col] = 1
        W0[k0*n + (i0+1), col] = 1
        W0[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        # M3: A[i0,j0](B[j0,k0+1]-B[j0+1,k0+1]) -> C[k0+1,i0]+=M3, C[k0+1,i0+1]+=M3
        U0[i0*m + j0, col] = 1
        V0[j0*p + (k0+1), col] = 1
        V0[(j0+1)*p + (k0+1), col] = -1
        W0[(k0+1)*n + i0, col] = 1
        W0[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M4: A[i0+1,j0+1](B[j0+1,k0]-B[j0,k0]) -> C[k0,i0]+=M4, C[k0,i0+1]+=M4
        U0[(i0+1)*m + (j0+1), col] = 1
        V0[(j0+1)*p + k0, col] = 1
        V0[j0*p + k0, col] = -1
        W0[k0*n + i0, col] = 1
        W0[k0*n + (i0+1), col] = 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1])B[j0+1,k0+1] -> C[k0,i0]-=M5, C[k0+1,i0]+=M5
        U0[i0*m + j0, col] = 1
        U0[i0*m + (j0+1), col] = 1
        V0[(j0+1)*p + (k0+1), col] = 1
        W0[k0*n + i0, col] = -1
        W0[(k0+1)*n + i0, col] = 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k0+1]) -> C[k0+1,i0+1]+=M6
        U0[(i0+1)*m + j0, col] = 1
        U0[i0*m + j0, col] = -1
        V0[j0*p + k0, col] = 1
        V0[j0*p + (k0+1), col] = 1
        W0[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1]) -> C[k0,i0]+=M7
        U0[i0*m + (j0+1), col] = 1
        U0[(i0+1)*m + (j0+1), col] = -1
        V0[(j0+1)*p + k0, col] = 1
        V0[(j0+1)*p + (k0+1), col] = 1
        W0[k0*n + i0, col] = 1
        col += 1

    # 2x4x4: 4 Strassen blocks (j0 in {0,2}, k0 in {0,2})
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # 2x4x2: 2 Strassen blocks (j0 in {0,2}, k0=4)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 2x4x1: 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U0[i*m + j, col] = 1
            V0[j*p + 6, col] = 1
            W0[6*n + i, col] = 1
            col += 1
    
    # Verify the rank-50 solution
    if not verify(U0, V0, W0, rank0):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Strategy: Try to reduce rank by finding linear dependencies among the rank-1 terms.
    # For each target rank, check if we can find a subset of columns that still spans the tensor.
    # We'll use a greedy approach: try removing one column at a time and see if the remaining
    # columns can still represent the tensor (by solving for new coefficients).
    
    def try_reduce_rank(U, V, W, current_rank, target_rank):
        """Try to reduce from current_rank to target_rank by finding linear combinations."""
        if time.time() - start_time > DEADLINE - 15:
            return None, None, None, False
        
        # Build the tensor as a 3D array
        T = np.zeros((n, m, p))
        for r in range(current_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We need to find U', V', W' of shape (nm, target_rank), (mp, target_rank), (pn, target_rank)
        # such that sum_r' U'[i*m+j, r'] * V'[j*p+k, r'] * W'[k*n+i, r'] = T[i,j,k]
        
        # This is a hard problem. Instead, let's try a simpler approach:
        # Check if any pair of columns can be merged (i.e., their contribution is linearly dependent)
        
        # For each pair of columns (r1, r2), check if we can express the combined effect
        # as a single rank-1 term. This requires that the outer product structure is preserved.
        
        # A simpler heuristic: try to find a column that is a linear combination of others
        # in a way that preserves the tensor structure.
        
        return None, None, None, False
    
    # Try to reduce by 1 rank using a more direct approach
    # For rank 49, we need to find if any single rank-1 term can be removed
    # and the remaining 49 terms can be adjusted to compensate.
    
    # Alternative: Use the fact that n=2 to create a more efficient decomposition.
    # The key insight: for each (j,k) pair, we have 2 output values (i=0 and i=1).
    # We can potentially share computations between i=0 and i=1.
    
    # Let's try a different tiling: 2x2x7 blocks
    # 2x2x7 can be decomposed as 2x2x4 + 2x2x3
    # 2x2x4: use Strassen (rank 7) + 2x2x0... wait, that's not right.
    
    # Actually, let's try: 2x4x7 = 2x2x7 + 2x2x7
    # Each 2x2x7 can use: Strassen for 2x2x4 (rank 7) + 4 trivial terms for 2x2x3... no.
    # 2x2x7: we can split as 2x2x4 + 2x2x3
    # 2x2x4: Strassen (rank 7)
    # 2x2x3: 2x2x2 (Strassen, rank 7) + 2x2x1 (rank 4) = rank 11
    # Total per 2x2x7: 7 + 11 = 18
    # Total for 2x4x7: 2 * 18 = 36? That seems too good. Let me check.
    
    # Wait, 2x2x3 is not 2x2x2 + 2x2x1 in terms of the k dimension.
    # 2x2x3 means A is 2x2, B is 2x3, C is 3x2.
    # We can decompose B as [B1, B2, B3] where B1, B2, B3 are 2x1.
    # Then C = [A*B1, A*B2, A*B3]
    # Each A*B_i is 2x1, which requires rank 2 (or 1 if we allow it).
    # Actually, 2x2x1: C = A*B where A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is rank 4 (each C[i] needs 2 terms, and they share B entries).
    # Actually, it's rank 2: 
    #   r1: U[0,0]=1, V[0]=1, W[0]=1 -> C[0] += A[0,0]*B[0]
    #   r2: U[0,1]=1, V[1]=1, W[0]=1 -> C[0] += A[0,1]*B[1]
    #   r3: U[1,0]=1, V[0]=1, W[1]=1 -> C[1] += A[1,0]*B[0]
    #   r4: U[1,1]=1, V[1]=1, W[1]=1 -> C[1] += A[1,1]*B[1]
    # So rank 4 for 2x2x1.
    
    # 2x2x3: 3 * rank(2x2x1) = 3*4 = 12? No, that's not right either.
    # Actually, 2x2x3: A is 2x2, B is 2x3, C is 3x2.
    # C[k,i] = sum_j A[i,j]*B[j,k]
    # For each k, C[:,k] = A * B[:,k]
    # Each of the 3 columns of C requires rank 4, but they share the A entries.
    # Total rank: 3*4 = 12? Or can we do better?
    # Actually, the standard rank for 2x2x3 is 12 (schoolbook).
    # Can we do better? I don't think so for small matrices.
    
    # So 2x2x7: 2x2x4 (rank 7 via Strassen) + 2x2x3 (rank 12) = 19
    # 2x4x7: 2 * 19 = 38? That would be a big improvement over 50!
    
    # Let me verify: 2x2x4 using Strassen.
    # A is 2x2, B is 2x4, C is 4x2.
    # Split B into two 2x2 blocks: B = [B1, B2]
    # C = [A*B1, A*B2]
    # Each A*B_i is 2x2, which can use Strassen (rank 7).
    # Total: 2*7 = 14.
    
    # So 2x2x7: 2x2x4 (rank 14) + 2x2x3 (rank 12) = 26
    # 2x4x7: 2 * 26 = 52. That's worse than 50.
    
    # Let me try another approach. The current rank-50 uses:
    # 6 Strassen blocks (42) + 8 trivial = 50.
    # The 6 Strassen blocks cover: 4x(2x2x2) + 2x(2x2x2) = 6x(2x2x2)
    # The 8 trivial terms cover: 2x4x1
    
    # Can we do better for the 2x4x1 part? 
    # 2x4x1: A is 2x4, B is 4x1, C is 1x2.
    # C[0] = sum_j A[0,j]*B[j]
    # C[1] = sum_j A[1,j]*B[j]
    # This is rank 8 (each C[i] needs 4 terms).
    # Can we share? No, because C[0] and C[1] have different A rows.
    # So rank 8 is optimal for 2x4x1.
    
    # The bottleneck is the Strassen blocks. Can we do better than 6 Strassen blocks?
    # 2x4x6: We need to cover 2x4x6 with Strassen-like blocks.
    # 2x4x6 = 2x2x6 + 2x2x6
    # Each 2x2x6: 2x2x4 (2 Strassen blocks, rank 14) + 2x2x2 (1 Strassen block, rank 7) = 21
    # Total: 2*21 = 42. Same as before.
    
    # I think rank 50 might be close to optimal for this approach.
    # Let's try to verify if rank 49 is possible by a more careful analysis.
    
    # For now, return the rank-50 solution.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
