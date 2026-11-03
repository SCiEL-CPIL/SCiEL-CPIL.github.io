"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i,j,k form a valid multiplication term, 0 otherwise
    # For matrix mult (A is n x m, B is m x p, C is n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k), 0 otherwise
    # But wait - in standard matrix multiplication, every (i,j,k) triple contributes
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """Try to reduce the rank of a decomposition to target_rank."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = len(U[0])
    
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    # Try to find a lower rank decomposition
    # This is a hard problem, so we use a heuristic approach
    # We'll try to find linear dependencies among the rank-1 components
    
    if time.time() > deadline:
        return None
    
    # Strategy: try to merge columns that are close to being linearly dependent
    # Or use a random projection approach
    
    # For now, try a simple approach: check if we can drop columns
    # that have very small contributions
    
    best_result = None
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create a random perturbation of the current decomposition
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Try to zero out some entries to create sparsity
        for c in range(current_rank):
            if random.random() < 0.3:
                # Zero out a random subset of entries in this column
                for idx in range(len(U_new)):
                    if random.random() < 0.5:
                        U_new[idx, c] = 0
                for idx in range(len(V_new)):
                    if random.random() < 0.5:
                        V_new[idx, c] = 0
                for idx in range(len(W_new)):
                    if random.random() < 0.5:
                        W_new[idx, c] = 0
        
        # Verify the perturbed decomposition
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            # Count non-zero columns
            nonzero_cols = 0
            for c in range(current_rank):
                if any(U_new[:, c] != 0) and any(V_new[:, c] != 0) and any(W_new[:, c] != 0):
                    nonzero_cols += 1
            
            if nonzero_cols < (best_result is None or len(best_result['U'][0])):
                # Extract only nonzero columns
                keep_cols = []
                for c in range(current_rank):
                    if any(U_new[:, c] != 0) and any(V_new[:, c] != 0) and any(W_new[:, c] != 0):
                        keep_cols.append(c)
                
                U_trim = U_new[:, keep_cols].tolist()
                V_trim = V_new[:, keep_cols].tolist()
                W_trim = W_new[:, keep_cols].tolist()
                
                best_result = {"U": U_trim, "V": V_trim, "W": W_trim}
        
        if time.time() > deadline:
            break
    
    return best_result

def _structured_search(n, m, p, deadline):
    """Structured search for low-rank decompositions."""
    # Try various structured patterns
    
    best = None
    best_rank = n * m * p
    
    # Pattern 1: Block structure
    # For 3x4x4, try to exploit the fact that n=3 is small
    
    # Try rank 30-45 range with random initialization
    for target_rank in range(35, 48):
        if time.time() > deadline - 10:
            break
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random sparse pattern
            for c in range(target_rank):
                # Choose random support sets
                u_support = set(random.sample(range(n*m), max(1, random.randint(1, n*m//2))))
                v_support = set(random.sample(range(m*p), max(1, random.randint(1, m*p//2))))
                w_support = set(random.sample(range(p*n), max(1, random.randint(1, p*n//2))))
                
                for idx in u_support:
                    U[idx, c] = random.choice([1, -1])
                for idx in v_support:
                    V[idx, c] = random.choice([1, -1])
                for idx in w_support:
                    W[idx, c] = random.choice([1, -1])
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search
    if time.time() < deadline - 20:
        found = _structured_search(n, m, p, deadline - 10)
        if found is not None:
            best_rank = len(found['U'][0])
            result = found
    
    # Try to reduce further
    if time.time() < deadline - 5:
        reduced = _try_reduce_rank(result['U'], result['V'], result['W'], n, m, p, best_rank - 1, deadline)
        if reduced is not None:
            new_rank = len(reduced['U'][0])
            if new_rank < best_rank:
                best_rank = new_rank
                result = reduced
    
    return result
# EVOLVE-BLOCK-END
