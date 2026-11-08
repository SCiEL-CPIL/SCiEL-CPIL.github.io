"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the target tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the tensor is: T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # More precisely: the matrix multiplication tensor T has entries T[i,j,k] = delta_{ik} for the "diagonal" 
    # Wait, let me reconsider. For matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where T[i,j,k] = 1 means "A[i,j] and B[j,k] contribute to C[i,k]"
    # So T[i,j,k] = 1 for all valid i,j,k (it's all ones in the appropriate shape)
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    
    # Actually, the standard formulation: the multiplication tensor has rank r if we can write
    # T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] (with appropriate index mapping)
    
    # For our indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, c] corresponds to the A[i,j] factor
    # V[j*p + k, c] corresponds to the B[j,k] factor  
    # W[k*n + i, c] corresponds to the C[i,k] factor (note: k*n + i, not i*n + k)
    
    # The tensor entry T[i,j,k] = 1 (all ones) because every (i,j,k) triple contributes
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Check a subset of entries for speed
    max_checks = 500
    checked = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if checked >= max_checks:
                    break
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
                checked += 1
            if checked >= max_checks:
                break
        if checked >= max_checks:
            break
    return True

def _verify_decomposition_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification of the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s deadline, leaving 20s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Strategy: Try to find a lower-rank decomposition using structured patterns
    # The key insight: we can sometimes combine terms using algebraic identities
    
    # Approach 1: Try rank = n*m*p - d for small d, using structured modifications
    # Approach 2: Use a bilinear form approach with random initialization and refinement
    
    # Let's try a structured approach: exploit that some entries can be shared
    # For matrix multiplication, there are known lower bounds and constructions
    
    # Try a simple structured decomposition that might reduce rank slightly
    # Use the fact that the tensor has a special structure
    
    # Method: Start with schoolbook, then try to merge pairs of terms
    U = best["U"]
    V = best["V"]  
    W = best["W"]
    rank = best_rank
    
    # Try to find linear dependencies that allow rank reduction
    # This is a heuristic approach - try random linear combinations
    
    np.random.seed(42)
    
    # Try several random structured decompositions
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
            
        # Try a different structured approach
        # Use the idea that we can sometimes use half-integer coefficients
        
        # Try rank = best_rank - 1
        target_rank = best_rank - 1
        if target_rank < n * m + m * p + p * n - 2:  # Strassen-like lower bound heuristic
            continue
            
        # Random initialization with structure
        U_try = np.zeros((n*m, target_rank))
        V_try = np.zeros((m*p, target_rank))
        W_try = np.zeros((p*n, target_rank))
        
        # Initialize with schoolbook pattern but with random perturbations
        for c in range(target_rank):
            # Pick a random (i,j,k) triple
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U_try[i*m+j, c] = 1.0
            V_try[j*p+k, c] = 1.0
            W_try[k*n+i, c] = 1.0
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(target_rank):
                        val += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists
            best = {
                "U": U_try.tolist(),
                "V": V_try.tolist(),
                "W": W_try.tolist()
            }
            best_rank = target_rank
            break
    
    # If we haven't improved, try a more systematic approach
    if best_rank == n * m * p:
        # Try a structured decomposition using block patterns
        # For 4x5 x 5x6, we can potentially use a Strassen-like approach
        
        # Try rank = 119, 118, etc. with structured patterns
        for target_rank in range(best_rank - 1, max(best_rank - 10, 100), -1):
            if time.time() > deadline - 20:
                break
                
            # Use a structured approach: try to find a valid decomposition
            # with this rank using a greedy algorithm
            
            # Initialize with a valid decomposition of higher rank, then try to reduce
            # This is complex, so use a simpler heuristic: random search with verification
            
            found = False
            for trial in range(100):
                if time.time() > deadline - 20:
                    break
                    
                U_try = np.zeros((n*m, target_rank))
                V_try = np.zeros((m*p, target_rank))
                W_try = np.zeros((p*n, target_rank))
                
                # Use a pattern-based initialization
                # Try to cover all (i,j,k) triples
                triples = list(product(range(n), range(m), range(p)))
                np.random.shuffle(triples)
                
                for c in range(target_rank):
                    if c < len(triples):
                        i, j, k = triples[c]
                        U_try[i*m+j, c] = 1.0
                        V_try[j*p+k, c] = 1.0
                        W_try[k*n+i, c] = 1.0
                    else:
                        # For extra terms, use random
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        k = np.random.randint(p)
                        U_try[i*m+j, c] = 1.0
                        V_try[j*p+k, c] = 1.0
                        W_try[k*n+i, c] = 1.0
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best = {
                        "U": U_try.tolist(),
                        "V": V_try.tolist(),
                        "W": W_try.tolist()
                    }
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    return best
# EVOLVE-BLOCK-END
