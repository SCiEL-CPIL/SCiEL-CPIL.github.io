"""Matrix multiplication 3x4 x 4x8 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as baseline
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    
    # Try to find better decompositions using structured approaches
    # For rectangular matrix multiplication, we can try rank-reducing patterns
    
    # Approach: Try to combine terms where possible
    # For 3x4 x 4x8, we look for patterns that reduce rank below 96
    
    # Known: M_{2,2,2} = 7 (Strassen), M_{2,3,3} <= 19, etc.
    # For 3x4 x 4x8, we can try to use block structure
    
    # Try a simple rank reduction: combine some terms
    # If two terms share structure, we might merge them
    
    # Let's try to find if there's a decomposition with rank < 96
    # by looking for linear dependencies in the tensor
    
    # Build the multiplication tensor T[i,j,k] = e_{i,k} in A⊗B⊗C
    # where A is 3x4, B is 4x8, C is 3x8
    # T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    
    # The tensor is sum over j of (e_i ⊗ e_j ⊗ e_k) for each (i,k)
    # Actually: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * delta where it contributes
    
    # For matrix mult: T[i,j,k] = 1 if this (i,j,k) triple contributes to output
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
    
    # Rank of this tensor is what we're looking for
    
    # Try random structured decompositions
    np.random.seed(42)
    
    # Try ranks from a reasonable upper bound downward
    # Known bounds: M_{3,4,8} is between some values
    # Lower bound: max(n*p, m*p, n*m) = max(24, 32, 12) = 32
    # Upper bound: schoolbook = 96
    
    # Try to find decompositions with rank in [80, 96)
    for target_rank in range(95, 80, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a greedy approach: start with schoolbook and try to merge terms
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Initialize with random sparse patterns
        for c in range(target_rank):
            # Choose random indices for this term
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U[i * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + i][c] = 1
        
        # Check if this decomposes correctly
        # The decomposition is correct if:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
        # and 0 otherwise (but for matrix mult, all valid triples contribute)
        
        # Actually for matrix multiplication, every (i,j,k) triple contributes
        # So we need: for each output (i_out, k_out), 
        # sum_j sum_c U[i_out*m+j, c] * V[j*p+k_out, c] * W[k_out*n+i_out, c] = 1
        
        # Let's verify
        valid = True
        for i_out in range(n):
            for k_out in range(p):
                total = 0
                for j in range(m):
                    for c in range(target_rank):
                        total += U[i_out * m + j][c] * V[j * p + k_out][c] * W[k_out * n + i_out][c]
                if abs(total - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_rank = target_rank
            best_U = U
            best_V = V
            best_W = W
            break
    
    # Also try a more systematic approach: exploit the structure
    # For 3x4 x 4x8, we can think of it as 3 blocks of 1x4 x 4x8
    # Each 1x4 x 4x8 has rank 32 (schoolbook)
    # But we might be able to share structure between blocks
    
    if time.time() < deadline - 10:
        # Try a structured decomposition
        # Use the fact that 3 = 2 + 1, so M_{3,4,8} <= M_{2,4,8} + M_{1,4,8} + cross terms
        # M_{1,4,8} = 8 (just dot product)
        # M_{2,4,8} <= 2 * M_{1,4,8} = 16 (but can do better)
        # Actually M_{2,4,8}: C[2,8] = A[2,4] * B[4,8]
        # Schoolbook rank = 2*4*8 = 64
        # Can we do better?
        
        # For now, keep the schoolbook if no better found
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
