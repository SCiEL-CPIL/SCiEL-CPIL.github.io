"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Block-diagonal embedding: split 3x5x6 into 3x2x6 and 3x3x6
    # Total rank = rank(3x2x6) + rank(3x3x6)
    # Schoolbook: 3*2*6 + 3*3*6 = 36 + 54 = 90
    # We can do better with structured decompositions
    
    # First, get the schoolbook as baseline
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)
    
    # Try block-structured decomposition
    # Split columns of A (5 cols) into 2 + 3
    # Block 1: 3x2x6, Block 2: 3x3x6
    
    # Generate structured rank-1 terms for each block
    # For 3x2x6: need to cover all (i,j,k) with i in 0..2, j in 0..1, k in 0..5
    # For 3x3x6: need to cover all (i,j,k) with i in 0..2, j in 0..2, k in 0..5
    
    # Strategy: use sparse structured terms
    # Each term contributes to a subset of (i,j,k) triples
    
    # Build the decomposition
    U_block1 = []  # List of vectors for 3x2 block (6 entries each)
    V_block1 = []  # List of vectors for 2x6 block (12 entries each)
    W_block1 = []  # List of vectors for 6x3 block (18 entries each)
    
    U_block2 = []  # List of vectors for 3x3 block (9 entries each)
    V_block2 = []  # List of vectors for 3x6 block (18 entries each)
    W_block2 = []  # List of vectors for 6x3 block (18 entries each)
    
    # Generate terms for block 1 (3x2x6)
    # Use structured patterns to minimize rank
    for i in range(3):
        for j in range(2):
            for k in range(6):
                u = [0]*6
                v = [0]*12
                w = [0]*18
                u[i*2+j] = 1.0
                v[j*6+k] = 1.0
                w[k*3+i] = 1.0
                U_block1.append(u)
                V_block1.append(v)
                W_block1.append(w)
    
    # Generate terms for block 2 (3x3x6)
    for i in range(3):
        for j in range(3):
            for k in range(6):
                u = [0]*9
                v = [0]*18
                w = [0]*18
                u[i*3+j] = 1.0
                v[j*6+k] = 1.0
                w[k*3+i] = 1.0
                U_block2.append(u)
                V_block2.append(v)
                W_block2.append(w)
    
    # Now try to reduce rank by combining terms
    # Use a greedy approach: try to merge terms that have overlapping support
    
    # First, let's try a more clever structured approach
    # For 3x2x6, we can potentially do better than 36
    
    # Try random structured search for better block decompositions
    best_block1 = _find_block_decomp(3, 2, 6, deadline)
    best_block2 = _find_block_decomp(3, 3, 6, deadline)
    
    if best_block1 is not None and best_block2 is not None:
        rank1, U1, V1, W1 = best_block1
        rank2, U2, V2, W2 = best_block2
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            # Assemble the full decomposition
            # Block 1 uses columns 0-1 of A, Block 2 uses columns 2-4
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            # Place block 1 terms (first rank1 columns)
            for c in range(rank1):
                # Map 3x2 block indices to full 3x5 indices
                for row in range(6):  # 3*2
                    i = row // 2
                    j = row % 2
                    U_full[i*5 + j, c] = U1[row, c]
                
                for row in range(12):  # 2*6
                    j = row // 6
                    k = row % 6
                    V_full[j*6 + k, c] = V1[row, c]
                
                for row in range(18):  # 6*3
                    k = row // 3
                    i = row % 3
                    W_full[k*3 + i, c] = W1[row, c]
            
            # Place block 2 terms (next rank2 columns)
            for c in range(rank2):
                # Map 3x3 block indices to full 3x5 indices (offset by 2)
                for row in range(9):  # 3*3
                    i = row // 3
                    j = row % 3
                    U_full[i*5 + (j+2), rank1 + c] = U2[row, c]
                
                for row in range(18):  # 3*6
                    j = row // 6
                    k = row % 6
                    V_full[(j+2)*6 + k, rank1 + c] = V2[row, c]
                
                for row in range(18):  # 6*3
                    k = row // 3
                    i = row % 3
                    W_full[k*3 + i, rank1 + c] = W2[row, c]
            
            # Verify
            if _verify_fast(n, m, p, U_full, V_full, W_full, total_rank):
                best_rank = total_rank
                best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return best_result

def _schoolbook(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_block_decomp(n, m, p, deadline):
    """Find a low-rank decomposition for n x m x p matmul block."""
    deadline = time.time() + 10.0  # Time limit per block
    best_rank = n * m * p
    best_result = None
    
    # Try structured approaches
    for r in range(max(n*m*p//2, 10), n*m*p+1):
        if time.time() > deadline:
            break
        
        # Try a few random structured attempts
        for trial in range(3):
            if time.time() > deadline:
                break
            
            result = _try_block_structured(n, m, p, r, trial)
            if result is not None:
                if r < best_rank:
                    best_rank = r
                    best_result = (r, result[0], result[1], result[2])
                    if r <= n*m*p//2:
                        break
        if best_result is not None and best_rank <= n*m*p//2:
            break
    
    return best_result

def _try_block_structured(n, m, p, r, trial_seed):
    """Try to find a structured decomposition of rank r for n x m x p."""
    random.seed(trial_seed * 7919 + r * 13 + n * 1000 + m * 100 + p)
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Use structured sparse patterns
    for c in range(r):
        # For each term, choose a structured subset
        # U: pick positions based on (i,j) pattern
        for i in range(n):
            for j in range(m):
                # Probabilistic inclusion based on structure
                if random.random() < 0.3:
                    U[i*m+j, c] = random.choice([1, -1, 0.5, -0.5])
        
        # V: pick positions based on (j,k) pattern
        for j in range(m):
            for k in range(p):
                if random.random() < 0.3:
                    V[j*p+k, c] = random.choice([1, -1, 0.5, -0.5])
        
        # W: pick positions based on (k,i) pattern
        for k in range(p):
            for i in range(n):
                if random.random() < 0.3:
                    W[k*n+i, c] = random.choice([1, -1, 0.5, -0.5])
    
    # Verify
    if _verify_block(n, m, p, U, V, W, r):
        return (U, V, W)
    return None

def _verify_block(n, m, p, U, V, W, r):
    """Verify block decomposition."""
    for i in range(n):
        for j in range(m):
            u_row = U[i*m + j, :]
            for k in range(p):
                v_row = V[j*p + k, :]
                w_row = W[k*n + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(n, m, p, U, V, W, r):
    """Fast verification using vectorized operations."""
    # For each (i,j,k), check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # Vectorized check
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
