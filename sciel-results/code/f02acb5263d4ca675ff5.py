"""Matrix multiplication 3x3x3 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def _build_matmul_tensor():
    """Build the 3x3x3 matrix multiplication tensor T[i,j,k] where
    T[i,j,k] = 1 if the (i,k) entry of the product involves A[i,j]*B[j,k].
    Using trace convention: output C[i,k] = sum_j A[i,j]*B[j,k].
    Tensor index: (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    The tensor T has T[i,j,k] = 1 for all valid (i,j,k) since each product term
    contributes to exactly one output entry C[i,k].
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _tensor_to_vec(T):
    """Flatten tensor to vector."""
    return T.flatten().copy()

def _vec_to_tensor(v, shape):
    """Reshape vector to tensor."""
    return v.reshape(shape)

def _check_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check if U, V, W exactly decompose the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each output (k, i) [trace convention: W[k*n+i, r]],
    the contribution is sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    This should equal 1 if (i,j,k) is a valid matmul term, else 0.
    Actually for 3x3x3, all (i,j,k) terms are present.
    """
    rank = U.shape[1]
    # Build the product tensor
    prod = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                prod[i, j, k] = s
    target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(prod, target, atol=tol, rtol=0)

def _verify_decomposition(U, V, W):
    """Verify using integer arithmetic for exactness."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                if s != 1:
                    return False
    return True

def _build_decomposition_from_factors(U, V, W):
    """Convert numpy arrays to lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    """Try to find a rank-minimal decomposition of 3x3x3 matrix multiplication.
    
    Known lower bound: rank >= 19 (Bürgisser)
    Known upper bound: 23 (Landsberg's rank 23 decomposition for 3x3)
    Actually the best known is 23, and it's unknown if 22 or lower works.
    
    Strategy: Try to find rank 23 first (known to exist), then try 22.
    Use the known Landsberg decomposition if we can recall it, otherwise
    use optimization to find one.
    
    For safety, if we can't verify a lower-rank decomposition in time,
    fall back to schoolbook rank 27.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s limit, keep margin
    best_rank = n * m * p  # schoolbook rank 27
    best_result = None
    
    # Schoolbook as fallback
    def schoolbook():
        U = np.zeros((n * m, n * m * p), dtype=np.int64)
        V = np.zeros((m * p, n * m * p), dtype=np.int64)
        W = np.zeros((p * n, n * m * p), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W
    
    # Try known rank 23 decomposition
    # The rank 23 decomposition of 3x3x3 matrix multiplication
    # This is a known result; let me try to construct it via search.
    
    # Approach: Use random initialization and iterative refinement
    # The matmul tensor for 3x3x3 has 27 nonzero entries (all 1s).
    # We need rank-r factors such that for all (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try rank 23, 22, 21, ...
    for target_rank in range(23, 18, -1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            
            # Random initialization with small integers/half-integers
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random values in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1.0, 0.0, 1.0], size=(n * m, target_rank))
            V = np.random.choice([-1.0, 0.0, 1.0], size=(m * p, target_rank))
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p * n, target_rank))
            
            # Iterative refinement: adjust one factor at a time
            # For a given (i,j,k), the constraint is:
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of 27 equations with 3*rank variables.
            
            # Use simple iterative method: for each equation, adjust one variable
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Check current error
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-6:
                    # Verify with exact arithmetic (convert to fractions if needed)
                    # For now, check if values are close to half-integers
                    converged = True
                    break
                
                # Pick the most violated equation
                worst = (0, 0, 0)
                worst_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = abs(s - 1.0)
                            if err > worst_err:
                                worst_err = err
                                worst = (i, j, k)
                
                if worst_err < 1e-6:
                    converged = True
                    break
                
                i, j, k = worst
                # Try adjusting each variable in this equation
                adjusted = False
                for r in range(target_rank):
                    u_val = U[i*m+j, r]
                    v_val = V[j*p+k, r]
                    w_val = W[k*n+i, r]
                    
                    # Current contribution
                    contrib = u_val * v_val * w_val
                    # Need to change contribution by (1 - s)
                    # s = sum of all contributions
                    s = 0.0
                    for r2 in range(target_rank):
                        s += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                    needed = 1.0 - s
                    
                    # Try small adjustments
                    for delta_u in [-1, 0.5, 1]:
                        for delta_v in [-1, 0.5, 1]:
                            for delta_w in [-1, 0.5, 1]:
                                new_u = u_val + delta_u
                                new_v = v_val + delta_v
                                new_w = w_val + delta_w
                                new_contrib = new_u * new_v * new_w
                                # Check if this helps
                                new_s = s - contrib + new_contrib
                                if abs(new_s - 1.0) < worst_err - 0.1:
                                    U[i*m+j, r] = new_u
                                    V[j*p+k, r] = new_v
                                    W[k*n+i, r] = new_w
                                    adjusted = True
                                    break
                            if adjusted:
                                break
                        if adjusted:
                            break
                    if adjusted:
                        break
                
                if not adjusted:
                    break
            
            if converged:
                # Verify exactness
                if _verify_decomposition(U.astype(np.int64) if np.allclose(U, np.round(U)) else U,
                                         V.astype(np.int64) if np.allclose(V, np.round(V)) else V,
                                         W.astype(np.int64) if np.allclose(W, np.round(W)) else W):
                    best_rank = target_rank
                    best_result = _build_decomposition_from_factors(
                        U.astype(int) if np.allclose(U, np.round(U)) else U,
                        V.astype(int) if np.allclose(V, np.round(V)) else V,
                        W.astype(int) if np.allclose(W, np.round(W)) else W
                    )
                    break
                # Also verify with float tolerance
                if _check_decomposition(U, V, W, tol=1e-6):
                    best_rank = target_rank
                    best_result = _build_decomposition_from_factors(
                        U.astype(int) if np.allclose(U, np.round(U)) else U,
                        V.astype(int) if np.allclose(V, np.round(V)) else V,
                        W.astype(int) if np.allclose(W, np.round(W)) else W
                    )
                    break
        
        if best_result is not None and best_rank == target_rank:
            break
    
    # If no better result found, return schoolbook
    if best_result is None:
        U, V, W = schoolbook()
        return _build_decomposition_from_factors(U, V, W)
    
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
