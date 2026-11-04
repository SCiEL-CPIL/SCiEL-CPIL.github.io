"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin before 150s limit

    # Helper to create schoolbook decomposition
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Helper to verify decomposition exactly using rational arithmetic
    def verify_exact(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition is exact."""
        # Check all 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check all 0-entries (full check for correctness)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (j == j2 and i == i2 and k == k2):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        
        return True

    # Helper to compute tensor error (Frobenius norm of difference)
    def tensor_error(U, V, W, n, m, p):
        """Compute the error in the tensor decomposition."""
        # Build the target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute the approximated tensor
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.linalg.norm(T_approx - T)

    # Helper to convert to fraction list
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    for denom in [1, 2, 4, 8, 16, 32, 64]:
                        numer = round(val * denom)
                        if abs(val - numer / denom) < 1e-6:
                            frac_row.append(numer / denom)
                            break
                    else:
                        frac_row.append(val)
            result.append(frac_row)
        return result

    # Initialize with schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = rank

    # Precompute target tensor for fast error computation
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    # STRUCTURAL APPROACH: Exploit n=2 structure
    # For 2x5x6, we can try to find decompositions that share computation
    # between the two rows.
    
    # Key insight: The tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # for some i,j,k. Since n=2, c = k*2+i, so i = c % 2, k = c // 2.
    # Similarly, a = i*m+j, so i = a // m, j = a % m.
    # b = j*p+k, so j = b // p, k = b % p.
    # So T[a,b,c] = 1 iff (a//m == c%2) and (a%m == b//p) and (b%p == c//2).
    
    # This structure can be exploited. Let's try a structured search.
    
    # Strategy: Try ranks from 40 down to 25, using structured patterns
    for target_rank in range(40, 24, -1):
        if time.time() > deadline - 20:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Use a structured initialization based on the tensor structure
        # For each rank position r, assign it to a specific (i,j,k) triple
        # but allow sharing between the two i values when possible
        
        # Pattern 1: Group by (j,k) and share between i=0 and i=1
        # For each (j,k), we need two entries: one for i=0, one for i=1
        # We can use 2 rank positions per (j,k), or try to share
        
        # Let's try a hierarchical approach:
        # Level 1: For each j, we have a "channel" of length p
        # Level 2: For each k within a channel, we need to compute for both i
        
        # Try different structured patterns
        patterns_tried = 0
        max_patterns = 5000
        
        while patterns_tried < max_patterns and time.time() < deadline - 10:
            patterns_tried += 1
            
            # Random seed based on rank and pattern number
            seed = target_rank * 1000 + patterns_tried
            np.random.seed(seed)
            
            # Generate structured random patterns with {-1, 0, 1}
            # but with some structure: make rows related
            U_r = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V_r = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W_r = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Add structure: for W, make W[2k, :] and W[2k+1, :] related
            # since they correspond to i=0 and i=1 for the same k
            for k in range(p):
                if np.random.random() < 0.5:
                    # Make them equal
                    W_r[2*k+1, :] = W_r[2*k, :]
                elif np.random.random() < 0.5:
                    # Make them negatives
                    W_r[2*k+1, :] = -W_r[2*k, :]
            
            # Compute error
            T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            err = np.linalg.norm(T_approx - T_target)
            
            if err < 1e-8:
                # Verify exactly
                if verify_exact(U_r, V_r, W_r, n, m, p, tol=1e-5):
                    best_rank = target_rank
                    best_result = (U_r.copy(), V_r.copy(), W_r.copy())
                    break
        
        if best_rank <= target_rank:
            break
    
    # If structured search didn't find anything, try gradient-based with better initialization
    if best_rank == n * m * p and time.time() < deadline - 30:
        # Try gradient-based with structured initialization
        for target_rank in range(35, 24, -1):
            if time.time() > deadline - 15:
                break
            
            if target_rank >= best_rank:
                continue
            
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(target_rank * 79 + restart * 13)
                
                # Initialize with structure
                U_r = np.random.randn(n * m, target_rank) * 0.2
                V_r = np.random.randn(m * p, target_rank) * 0.2
                W_r = np.random.randn(p * n, target_rank) * 0.2
                
                # Apply structure to W
                for k in range(p):
                    W_r[2*k+1, :] = W_r[2*k, :]
                
                # Gradient-based optimization
                lr = 0.02
                momentum = 0.9
                vel_U = np.zeros_like(U_r)
                vel_V = np.zeros_like(V_r)
                vel_W = np.zeros_like(W_r)
                
                best_err = float('inf')
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                
                for step in range(200):
                    if time.time() > deadline - 3:
                        break
                    
                    T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    error = T_approx - T_target
                    err = np.linalg.norm(error)
                    
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    
                    if err < 1e-8:
                        break
                    
                    grad_U = np.einsum('abc,br,cr->ar', error, V_r, W_r)
                    grad_V = np.einsum('abc,ar,cr->br', error, U_r, W_r)
                    grad_W = np.einsum('abc,ar,br->cr', error, U_r, V_r)
                    
                    vel_U = momentum * vel_U - lr * grad_U
                    vel_V = momentum * vel_V - lr * grad_V
                    vel_W = momentum * vel_W - lr * grad_W
                    
                    U_r += vel_U
                    V_r += vel_V
                    W_r += vel_W
                    
                    # Enforce structure on W
                    for k in range(p):
                        W_r[2*k+1, :] = W_r[2*k, :]
                    
                    if step % 40 == 0 and step > 0:
                        lr *= 0.7
                
                err = tensor_error(best_U, best_V, best_W, n, m, p)
                if err < 1e-6:
                    if verify_exact(best_U, best_V, best_W, n, m, p, tol=1e-5):
                        best_rank = target_rank
                        best_result = (best_U.copy(), best_V.copy(), best_W.copy())
                        break
            
            if best_rank <= target_rank:
                break
    
    # Final polish: try to clean up the solution
    if best_rank < n * m * p:
        U_final, V_final, W_final = best_result
        # Try to snap to rational values
        for mat in [U_final, V_final, W_final]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    for denom in [1, 2, 4, 8, 16]:
                        numer = round(mat[i, j] * denom)
                        if abs(mat[i, j] - numer / denom) < 1e-4:
                            mat[i, j] = numer / denom
                            break
    
    U_final, V_final, W_final = best_result
    return {"U": to_fraction_list(U_final), "V": to_fraction_list(V_final), "W": to_fraction_list(W_final)}

def tensor_error_target(n, m, p):
    """Create the target multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T
# EVOLVE-BLOCK-END
