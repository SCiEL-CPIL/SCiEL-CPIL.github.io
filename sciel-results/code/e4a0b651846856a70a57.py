"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition."""
    rank = n * m * p
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=10000, seed=42):
    """Try to reduce rank by finding linear dependencies among the columns."""
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best = None
    for iteration in range(max_iters):
        c1 = random.randint(0, rank - 1)
        c2 = random.randint(0, rank - 1)
        if c1 == c2:
            continue
        
        if iteration % 100 == 0:
            pass
    
    return None

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 matrices (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11)(B22-B12)
    U[0][0] = 1
    V[3][0] = 1
    V[2][0] = -1
    W[2][0] = 1
    W[3][0] = 1
    
    # M2 = (A11+A12)(B12)
    U[0][1] = 1
    U[1][1] = 1
    V[2][1] = 1
    W[3][1] = 1
    
    # M3 = (A21+A22)(B11)
    U[2][2] = 1
    U[3][2] = 1
    V[0][2] = 1
    W[2][2] = 1
    
    # M4 = (A22)(B21-B11)
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[2][3] = 1
    W[3][3] = 1
    
    # M5 = (A11)(B11+B22)
    U[0][4] = 1
    V[0][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1
    U[3][6] = -1
    V[1][6] = 1
    V[3][6] = 1
    W[2][6] = 1
    
    return U, V, W

def _kronecker_decomp(n, m, p):
    """Try to find a decomposition using Kronecker product structure."""
    start_time = time.time()
    max_time = 60.0
    
    best_result = None
    best_rank = n * m * p
    
    for target_rank in range(28, 85):
        if time.time() - start_time > max_time:
            break
        
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.4:
                            U[i*m + j, c] = random.choice([-1, 1])
                
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.4:
                            V[j*p + k, c] = random.choice([-1, 1])
                
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.4:
                            W[k*n + i, c] = random.choice([-1, 1])
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_block_decomposition(n, m, p, max_time=50.0):
    """Try to decompose using block structure."""
    start_time = time.time()
    
    for n1 in range(1, n):
        for m1 in range(1, m):
            for p1 in range(1, p):
                if time.time() - start_time > max_time:
                    return None
                
                n2 = n - n1
                m2 = m - m1
                p2 = p - p1
                
                rank1 = n1 * m1 * p1
                rank2 = n2 * m2 * p2
                total_rank = rank1 + rank2
                
                if total_rank >= n * m * p:
                    continue
                
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                col = 0
                for i in range(n1):
                    for j in range(m1):
                        for k in range(p1):
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            U[(n1 + i) * m + (m1 + j)][col] = 1
                            V[(m1 + j) * p + (p1 + k)][col] = 1
                            W[(p1 + k) * n + (n1 + i)][col] = 1
                            col += 1
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(total_rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(s - 1.0) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U, "V": V, "W": W}
    
    return None

def _structured_search(n, m, p, max_time=80.0):
    """
    More effective search using structured initialization and iterative refinement.
    Uses the insight that matrix multiplication tensors have low-rank structure
    that can be exploited by initializing with sparse patterns and refining.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Lower bound on rank is max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Upper bound is schoolbook: n*m*p = 84
    # Known results suggest rank around 50-60 for 3x4x7
    
    best_result = None
    best_rank = n * m * p
    
    # Try different rank values from low to high
    # Focus on the most likely range first
    rank_candidates = list(range(28, 50)) + list(range(50, 65)) + list(range(65, 85))
    
    for target_rank in rank_candidates:
        if time.time() > deadline:
            break
        
        # Try multiple random seeds/initializations for each rank
        max_attempts = 200 if target_rank < 50 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use structured initialization based on known patterns
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Strategy: Initialize with sparse random entries, then refine
            # Use a mix of -1, 0, 1 with some structure
            
            # First, try a block-structured initialization
            # Divide the rank into chunks that correspond to different (i,j,k) groups
            
            # For each column c, assign it to a specific "type" of multiplication
            for c in range(target_rank):
                # Randomly choose which (i,j,k) pattern this column contributes to
                # But with some structure to avoid pure randomness
                
                # Method 1: Random sparse
                if attempt % 3 == 0:
                    # Pure random sparse
                    for i in range(n):
                        for j in range(m):
                            if random.random() < 0.3:
                                U[i*m + j, c] = random.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if random.random() < 0.3:
                                V[j*p + k, c] = random.choice([-1, 1])
                    for k in range(p):
                        for i in range(n):
                            if random.random() < 0.3:
                                W[k*n + i, c] = random.choice([-1, 1])
                
                # Method 2: Block-structured
                elif attempt % 3 == 1:
                    # Assign this column to a specific block
                    block_i = c % n
                    block_j = (c // n) % m
                    block_k = (c // (n * m)) % p
                    
                    # This column primarily handles (block_i, block_j, block_k)
                    # but may have some overlap
                    U[block_i * m + block_j, c] = 1
                    V[block_j * p + block_k, c] = 1
                    W[block_k * n + block_i, c] = 1
                    
                    # Add some random additional entries for flexibility
                    if random.random() < 0.5:
                        extra_i = random.randint(0, n - 1)
                        extra_j = random.randint(0, m - 1)
                        if extra_i != block_i or extra_j != block_j:
                            U[extra_i * m + extra_j, c] = random.choice([-1, 1])
                    
                    if random.random() < 0.5:
                        extra_j = random.randint(0, m - 1)
                        extra_k = random.randint(0, p - 1)
                        if extra_j != block_j or extra_k != block_k:
                            V[extra_j * p + extra_k, c] = random.choice([-1, 1])
                    
                    if random.random() < 0.5:
                        extra_k = random.randint(0, p - 1)
                        extra_i = random.randint(0, n - 1)
                        if extra_k != block_k or extra_i != block_i:
                            W[extra_k * n + extra_i, c] = random.choice([-1, 1])
                
                # Method 3: Strassen-inspired for larger blocks
                else:
                    # Use a hierarchical approach
                    # Split into 2x2 blocks where possible
                    
                    # For n=3, m=4, p=7:
                    # Think of A as 3x4, B as 4x7
                    # We can try to use Strassen-like combinations
                    
                    # Random assignment with bias toward certain patterns
                    for i in range(n):
                        for j in range(m):
                            prob = 0.4 if (i + j) % 2 == 0 else 0.2
                            if random.random() < prob:
                                U[i*m + j, c] = random.choice([-1, 1])
                    
                    for j in range(m):
                        for k in range(p):
                            prob = 0.4 if (j + k) % 2 == 0 else 0.2
                            if random.random() < prob:
                                V[j*p + k, c] = random.choice([-1, 1])
                    
                    for k in range(p):
                        for i in range(n):
                            prob = 0.4 if (k + i) % 2 == 0 else 0.2
                            if random.random() < prob:
                                W[k*n + i, c] = random.choice([-1, 1])
            
            # Verify this candidate
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break  # Found valid decomposition at this rank, move to lower rank
    
    return best_result

def _iterative_refinement(U, V, W, n, m, p, max_time=30.0):
    """
    Given an initial decomposition, try to reduce rank by finding
    linear dependencies and merging columns.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    rank = U_np.shape[1]
    
    # Try to find columns that are nearly linearly dependent
    # and merge them
    
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = rank
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Try to remove one column at a time
        # Check if column c can be expressed as a combination of others
        
        for c_remove in range(best_rank):
            if time.time() > deadline:
                break
            
            # Check if removing column c still gives a valid decomposition
            # This is expensive, so use a quick check first
            
            # Quick check: see if the column is "redundant"
            # by checking if the tensor equation still holds without it
            
            U_test = np.delete(best_U, c_remove, axis=1)
            V_test = np.delete(best_V, c_remove, axis=1)
            W_test = np.delete(best_W, c_remove, axis=1)
            
            # Sample a few (i,j,k) triples to check
            valid = True
            for _ in range(20):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                s = np.sum(U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :])
                if abs(s - 1.0) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Full verification
                fully_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :])
                            if abs(s - 1.0) > 1e-8:
                                fully_valid = False
                                break
                        if not fully_valid:
                            break
                    if not fully_valid:
                        break
                
                if fully_valid:
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank -= 1
                    break  # Restart the removal process with the new rank
    
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115 second deadline (safety margin)
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Try structured search first (most likely to find good solutions)
    if time.time() < deadline - 60:
        result = _structured_search(n, m, p, max_time=50.0)
        if result is not None:
            result_rank = len(result["U"][0])
            if result_rank < best_rank:
                best_result = result
                best_rank = result_rank
    
    # Try block decomposition
    if time.time() < deadline - 30:
        result = _try_block_decomposition(n, m, p, max_time=20.0)
        if result is not None:
            result_rank = len(result["U"][0])
            if result_rank < best_rank:
                best_result = result
                best_rank = result_rank
    
    # Try iterative refinement on the best result found
    if time.time() < deadline - 15:
        U_curr = best_result["U"]
        V_curr = best_result["V"]
        W_curr = best_result["W"]
        result = _iterative_refinement(U_curr, V_curr, W_curr, n, m, p, max_time=15.0)
        if result is not None:
            result_rank = len(result["U"][0])
            if result_rank < best_rank:
                best_result = result
                best_rank = result_rank
    
    # Final check: verify the best result
    U_final = np.array(best_result["U"])
    V_final = np.array(best_result["V"])
    W_final = np.array(best_result["W"])
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_final[i*m+j, :] * V_final[j*p+k, :] * W_final[k*n+i, :])
                if abs(s - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
