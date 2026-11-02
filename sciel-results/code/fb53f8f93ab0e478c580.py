"""Matrix multiplication 2,2,15 tensor decomposition with rank reduction."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from fractions import Fraction

def _strassen_block():
    """Return U, V, W for a single M_{2,2,2} block (rank 7).
    U: 4x7 (i*2+j), V: 4x7 (j*2+k), W: 4x7 (k*2+i)
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _build_initial_decomposition(n, m, p):
    """Build initial decomposition using 7 Strassen blocks + 1 schoolbook block.
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank) as numpy arrays.
    """
    U_s, V_s, W_s = _strassen_block()
    
    # 7 Strassen blocks for columns 0-13, 1 schoolbook for column 14
    rank = 7 * 7 + 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        for r in range(7):
            c = col_offset + r
            # U: i*m + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = U_s[i * m + j, r]
            
            # V: j*p + k
            for j in range(m):
                V[j * p + k0, c] = V_s[j * 2 + 0, r]
                V[j * p + k1, c] = V_s[j * 2 + 1, r]
            
            # W: k*n + i
            for k in range(2):
                kk = k0 + k
                for i in range(n):
                    W[kk * n + i, c] = W_s[k * 2 + i, r]
        
        col_offset += 7
    
    # Schoolbook block for column 14 (rank 4)
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(n):
            for j in range(m):
                U[i * m + j, c] = 1.0 if (i * m + j == r) else 0.0
        for j in range(m):
            V[j * p + k_last, c] = 1.0 if (j == r % 2 and r < 2) else (1.0 if (j == r % 2 and r >= 2) else 0.0)
        for i in range(n):
            W[k_last * n + i, c] = 1.0 if (i == r % 2 and r < 2) else (1.0 if (i == r % 2 and r >= 2) else 0.0)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if i,j,k form a valid product term.
    For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 for all valid (i,j,k), 0 otherwise.
    """
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_reduction=3, seed=42):
    """Try to reduce rank by finding linear dependencies.
    
    Strategy: For each pair of columns in the rank dimension, check if we can
    eliminate one by expressing it as a combination of others while preserving
    the tensor equation.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    current_rank = U.shape[1]
    rng = np.random.RandomState(seed)
    
    # Try to eliminate columns one at a time
    for attempt in range(max_reduction * 20):
        if U.shape[1] <= 40:  # Stop if we get very low
            break
        
        # Pick a random column to try to eliminate
        target = rng.randint(U.shape[1])
        
        # Try to express column target as combination of others
        # This is a constrained problem; use a heuristic approach
        # Try simple combinations: target = a * col_i + b * col_j
        found = False
        for i in range(U.shape[1]):
            if i == target:
                continue
            for j in range(U.shape[1]):
                if j == target or j == i:
                    continue
                # Try: new_col = a*U[:,i] + b*U[:,j]
                # We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # If we replace col_target with a*col_i + b*col_j, we need to
                # adjust other columns to maintain the sum
                
                # Simple approach: just try to see if removing target and
                # adding a combination works
                pass
        
        # Alternative: use null space approach
        # The tensor equation is: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        # If we can find a vector (u, v, w) such that adding u⊗v⊗w to the
        # decomposition still gives T, we can potentially reduce rank
        
        # For now, try a direct approach: check if any column is redundant
        # by seeing if the tensor can be represented with that column zeroed
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, target] = 0
        V_test[:, target] = 0
        W_test[:, target] = 0
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            # Column is redundant! Remove it
            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            found = True
            break
    
    return U, V, W

def _advanced_rank_reduction(U, V, W, n, m, p, time_limit=100):
    """More advanced rank reduction using structured search.
    
    Key insight: The 15 columns have structure. Adjacent column pairs can be
    processed with overlapping Strassen blocks, allowing shared computations.
    """
    start_time = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    rng = np.random.RandomState(123)
    
    # Build the full tensor T
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    # Try to find rank reduction by perturbing the decomposition
    # and checking if we can project to a lower-dimensional subspace
    
    current_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Attempt: try to merge adjacent Strassen blocks
    # If we process columns (0,1), (2,3), ..., (12,13) with Strassen,
    # and column 14 with schoolbook, maybe we can do better by
    # processing (0,1,2), (3,4,5), ... with a 3-column block
    
    # For now, try random linear combinations to find dependencies
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a linear combination of rank-1 terms that sums to zero
        # If sum_{r in S} c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0,
        # then we can eliminate one term
        
        # Simplified: try pairs
        for r1 in range(min(current_rank, 20)):
            for r2 in range(r1 + 1, min(current_rank, 20)):
                # Check if U[:,r1], V[:,r1], W[:,r1] is proportional to
                # U[:,r2], V[:,r2], W[:,r2]
                if np.linalg.norm(U[:, r1]) > 1e-10 and np.linalg.norm(U[:, r2]) > 1e-10:
                    ratio_U = np.dot(U[:, r1], U[:, r2]) / np.dot(U[:, r2], U[:, r2])
                    if np.linalg.norm(U[:, r1] - ratio_U * U[:, r2]) < 1e-8:
                        # Check V and W
                        if np.linalg.norm(V[:, r1]) > 1e-10 and np.linalg.norm(V[:, r2]) > 1e-10:
                            ratio_V = np.dot(V[:, r1], V[:, r2]) / np.dot(V[:, r2], V[:, r2])
                            if np.linalg.norm(V[:, r1] - ratio_V * V[:, r2]) < 1e-8:
                                if np.linalg.norm(W[:, r1]) > 1e-10 and np.linalg.norm(W[:, r2]) > 1e-10:
                                    ratio_W = np.dot(W[:, r1], W[:, r2]) / np.dot(W[:, r2], W[:, r2])
                                    if np.linalg.norm(W[:, r1] - ratio_W * W[:, r2]) < 1e-8:
                                        # They are proportional! Remove one
                                        U = np.delete(U, r2, axis=1)
                                        V = np.delete(V, r2, axis=1)
                                        W = np.delete(W, r2, axis=1)
                                        current_rank = U.shape[1]
                                        break
            if current_rank != best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = current_rank
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Step 1: Build initial valid decomposition
    U, V, W = _build_initial_decomposition(n, m, p)
    
    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return what we have (shouldn't happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    initial_rank = U.shape[1]
    
    # Step 2: Try to reduce rank
    if time.time() - start_time < 120:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_reduction=5)
    
    # Step 3: More advanced reduction
    if time.time() - start_time < 100:
        U, V, W = _advanced_rank_reduction(U, V, W, n, m, p, time_limit=80)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # If reduction broke it, revert to initial
        U, V, W = _build_initial_decomposition(n, m, p)
    
    # Convert to lists with clean values
    def clean_matrix(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-8:
                    clean_row.append(rounded)
                else:
                    clean_row.append(round(val, 10))
            result.append(clean_row)
        return result
    
    return {"U": clean_matrix(U), "V": clean_matrix(V), "W": clean_matrix(W)}
# EVOLVE-BLOCK-END
