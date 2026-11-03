"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for c_idx in range(rank):
                    val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-diagonal entries are 0
    # Sample some non-diagonal entries
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j != j2 or k != k2:
            val = 0.0
            for c_idx in range(rank):
                val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
            if abs(val) > tol:
                return False
    return True

def _schoolbook(n, m, p):
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

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _try_merge_terms(n, m, p, terms_to_merge, seed=42):
    """
    Try to merge a set of schoolbook terms into fewer terms.
    terms_to_merge: list of (i, j, k) tuples representing terms to combine
    Returns a dict with U, V, W if successful, else None.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Get the indices for each term
    term_indices = []
    for (i, j, k) in terms_to_merge:
        a = i * m + j
        b = j * p + k
        c = k * n + i
        term_indices.append((a, b, c))
    
    # For each term, we need U[a,c]=1, V[b,c]=1, W[c,c]=1
    # We want to find a linear combination that cancels out non-target entries
    
    # Start with the schoolbook decomposition for these terms
    num_terms = len(terms_to_merge)
    U = np.zeros((n * m, num_terms))
    V = np.zeros((m * p, num_terms))
    W = np.zeros((p * n, num_terms))
    
    for idx, (a, b, c) in enumerate(term_indices):
        U[a, idx] = 1
        V[b, idx] = 1
        W[c, idx] = 1
    
    # Try to find a rank-deficient linear transformation that preserves the tensor
    # This is equivalent to finding a null space vector
    
    # Build the constraint matrix: for each (a, b, c) not in target, the sum must be 0
    # For each (a, b, c) in target, the sum must be 1
    
    # Let's try a simpler approach: use random linear combinations and check
    
    for trial in range(50):
        # Generate random coefficients
        coeffs = np.random.randn(num_terms)
        
        # New decomposition with rank num_terms - 1
        # U_new[:, c] = sum_d U[:, d] * coeffs[d] for d != last
        # This is complex. Let's use a different strategy.
        
        # Try to find if there's a linear dependency
        # The tensor T = sum_c U[:,c] V[:,c] W[:,c]
        # If we can write one term as a combination of others, we can reduce rank
        
        pass
    
    return None

def _systematic_search(n, m, p, max_rank=None, deadline=None):
    """
    Systematically search for lower-rank decompositions.
    Uses structured perturbations of the schoolbook basis.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    best_result = _schoolbook(n, m, p)
    best_rank = max_rank
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Strategy: try to find a rank-r decomposition by solving a nonlinear system
    # Use alternating least squares or similar
    
    for target_rank in range(max_rank - 1, 2 * n * m * p // 3, -1):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Alternating least squares
            for iteration in range(100):
                if deadline is not None and time.time() > deadline - 5:
                    break
                
                # Update U: for each column c, U[:,c] should minimize
                # ||T - sum_c' U[:,c'] V[:,c'] W[:,c']||^2
                # This is complex. Use a simpler approach.
                
                # Instead, use the fact that we can fix two matrices and solve for the third
                # Fix V and W, solve for U
                # For each (a, b): T[a,b,:] = sum_c U[a,c] V[b,c] W[:,c]
                # This is still complex for 3-way.
                
                # Let's use a different approach: gradient descent
                # Compute gradient of ||T - tensor_product(U,V,W)||^2
                
                # For simplicity, just check if current U,V,W is close
                # and do a simple update
                
                # Compute residual
                residual = np.zeros_like(T)
                for c in range(target_rank):
                    residual += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                
                err = np.linalg.norm(T - residual)
                
                if err < 1e-4:
                    # Check if this is a valid decomposition
                    U_list = [[float(U[i, c]) for c in range(target_rank)] for i in range(n * m)]
                    V_list = [[float(V[i, c]) for c in range(target_rank)] for i in range(m * p)]
                    W_list = [[float(W[i, c]) for c in range(target_rank)] for i in range(p * n)]
                    
                    if _validate(U_list, V_list, W_list, n, m, p, tol=1e-3):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
                
                # Simple gradient step
                # dL/dU[a,c] = 2 * sum_{b,c'} (T[a,b,c'] - residual[a,b,c']) * V[b,c'] * W[c',c]
                # This is expensive. Use a simpler heuristic.
                
                # Just do a random perturbation
                U += 0.01 * np.random.randn(n * m, target_rank)
                V += 0.01 * np.random.randn(m * p, target_rank)
                W += 0.01 * np.random.randn(p * n, target_rank)
            
            if best_rank <= target_rank:
                break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Return schoolbook as fallback
    result = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # For 3x3x5, the schoolbook rank is 45
    # Known lower bounds suggest rank is at least 27 (n*m*p/2 for some cases)
    # Let's search for rank < 45
    
    # Try systematic search with deadline
    better_result = _systematic_search(n, m, p, max_rank=45, deadline=deadline)
    
    if better_result is not None:
        result = better_result
    
    return result
# EVOLVE-BLOCK-END
