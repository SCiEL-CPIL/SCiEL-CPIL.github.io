"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 145  # safety margin before 150s kill

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Strategy: Try to find a lower rank decomposition by exploiting structure.
    # For matrix multiplication, the tensor is T[i,j,k] = 1 for all valid (i,j,k).
    # We seek rank-r decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Key insight: For fixed U and V, the system is linear in W.
    # We can try various structured U, V and solve for W exactly.
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_np(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        r = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank_structured(r, max_tries=20, seed=42):
        """
        Try to find rank-r decomposition with structured U, V.
        
        Approach: 
        1. For each column c, the rank-1 term is u_c ⊗ v_c ⊗ w_c
        2. We need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all (i,j,k)
        
        Strategy: Fix U and V to have a specific structure, then solve for W.
        The system is linear in W: for each (i,j,k), sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        
        This is a linear system A * W_flat = b where:
        - A has shape (n*m*p, p*n*r)
        - b has shape (n*m*p,) with all 1s
        """
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_tries):
            if time.time() > deadline:
                return None
            
            # Initialize U and V with random values
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            
            # Build linear system for W
            # Equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            
            eq = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            w_idx = (k * n + i) * r + c
                            A[eq, w_idx] = U[i * m + j, c] * V[j * p + k, c]
                        eq += 1
            
            # Solve the linear system
            try:
                # Check if system is consistent
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                if len(residuals) > 0:
                    if residuals[0] < 1e-10:
                        # Solution exists and is exact
                        W = W_flat.reshape(p * n, r)
                        
                        # Verify
                        if verify_np(U.tolist(), V.tolist(), W.tolist()):
                            return {
                                "U": [[float(x) for x in U[i]] for i in range(n*m)],
                                "V": [[float(x) for x in V[i]] for i in range(m*p)],
                                "W": [[float(x) for x in W[i]] for i in range(p*n)]
                            }
            except Exception:
                pass
            
            # Try different random initialization
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
        
        return None

    # Try to find lower rank decompositions
    # Start from lower bounds and work up
    # Theoretical lower bound for matmul rank is at least max(n*m, m*p, p*n) / something
    # For 3x4x4, let's try ranks from 32 up to 47
    
    for r in range(32, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few different seeds for each rank
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = try_rank_structured(r, max_tries=10, seed=seed + r * 100)
            if result is not None:
                best_rank = r
                best = result
                print(f"Found rank {r} decomposition with seed {seed}")
                # Continue searching for even lower rank
                break
        
        # If we found a decomposition, continue to see if we can do better
        if best_rank < n * m * p:
            continue
    
    # Final verification
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook if verification fails
        best = schoolbook()
        best_rank = n * m * p
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
