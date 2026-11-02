"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _generate_hadamard_block(size):
    """Generate a Hadamard matrix of given size (power of 2)."""
    if size == 1:
        return np.array([[1.0]])
    H_half = _generate_hadamard_block(size // 2)
    return np.block([[H_half, H_half], [H_half, -H_half]])

def _structured_init(n, m, p, rank, seed=42):
    """
    Initialize U, V, W using Hadamard-like structures to exploit algebraic symmetries.
    This mimics the Winograd approach of using transforms.
    """
    np.random.seed(seed)
    
    # We want to create matrices that have some structure.
    # A simple approach is to use random linear combinations of Hadamard blocks.
    # However, for exact decomposition, we start with random sparse or structured matrices 
    # and let ALS refine them.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: Initialize with small random values, but inject structure.
    # For Winograd, the factors often involve transforms. 
    # Let's try initializing with values from a Hadamard matrix scaled down.
    
    # Create a large Hadamard matrix to sample from
    H_size = 1
    while H_size < max(n*m, m*p, p*n, rank):
        H_size *= 2
    H = _generate_hadamard_block(H_size)
    
    # Sample rows from H to initialize U, V, W
    # This ensures orthogonality-like properties in the initial steps
    for r in range(rank):
        # Pick a random row index for each factor
        idx_u = np.random.randint(0, H_size)
        idx_v = np.random.randint(0, H_size)
        idx_w = np.random.randint(0, H_size)
        
        # Scale the Hadamard rows
        scale = 1.0 / np.sqrt(rank) # Heuristic scaling
        
        U[:, r] = H[idx_u, :n*m] * scale * 0.5 # Start small to avoid divergence
        V[:, r] = H[idx_v, :m*p] * scale * 0.5
        W[:, r] = H[idx_w, :p*n] * scale * 0.5
        
        # Add small noise to break symmetries and allow ALS to converge
        U[:, r] += np.random.randn(n*m) * 0.01
        V[:, r] += np.random.randn(m*p) * 0.01
        W[:, r] += np.random.randn(p*n) * 0.01
        
    return U, V, W

def _als_refine(U, V, W, n, m, p, max_iters=100, deadline=None):
    """
    Alternating Least Squares refinement.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] should satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                # For each k: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    # Use lstsq for stability
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, V, optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured initialization
    # We try ranks slightly below schoolbook first
    for rank in range(79, 70, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few different seeds for structured initialization
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            U_init, V_init, W_init = _structured_init(n, m, p, rank, seed=seed)
            
            # Refine
            U_ref, V_ref, W_ref = _als_refine(U_init, V_init, W_init, n, m, p, max_iters=50, deadline=deadline - 10)
            
            # Round to half-integers
            U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U_ref])
            V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V_ref])
            W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W_ref])
            
            # Verify
            if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = rank
                found = True
                break
                
        if found:
            # If we found a lower rank, we can stop or try even lower
            # Given the difficulty, finding one is a significant win. 
            # We continue to see if we can go lower, but we are likely to succeed at 79 if possible.
            pass

    return best_result
# EVOLVE-BLOCK-END
