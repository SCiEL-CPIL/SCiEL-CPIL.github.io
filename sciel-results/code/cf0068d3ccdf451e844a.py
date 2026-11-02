"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check dimensions
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    # For each (i, k) pair, check that sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    # Actually, the decomposition requires:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The standard form: for each output element C[i,k], we need:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form with factors U (n*m x rank), V (m*p x rank), W (p*n x rank):
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # And T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True

def _winograd_decomposition(n, m, p, deadline):
    """
    Attempt a Winograd-style decomposition using structured transforms.
    
    For matrix multiplication C = A * B where A is n×m and B is m×p,
    we can use Winograd's algorithm which reduces multiplications by
    exploiting algebraic structure.
    
    The key idea: instead of n*m*p scalar multiplications, use
    (n+m-1)*(m+p-1) or similar reduced counts with structured transforms.
    
    For 4×4 × 4×5:
    - Standard: 4*4*5 = 80 multiplications
    - Winograd 2x2 block: can reduce per-block multiplications
    """
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = best["U"][0].__len__() if best["U"] else n * m * p
    
    # Try to find a lower-rank decomposition using random initialization
    # and iterative refinement
    
    np.random.seed(42)
    
    # Try various rank values below schoolbook
    for target_rank in range(max(1, best_rank - 20), best_rank):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random small values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Iteratively refine using gradient-like updates
        # The objective is to match T[i,j,k] = 1 for all (i,j,k)
        
        lr = 0.01
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
                
            # Compute current tensor values and errors
            U_err = np.zeros_like(U)
            V_err = np.zeros_like(V)
            W_err = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                        # dT/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                        # dT/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                        
                        s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        error = s - 1.0
                        
                        U_err[i * m + j, :] -= lr * error * V[j * p + k, :] * W[k * n + i, :]
                        V_err[j * p + k, :] -= lr * error * U[i * m + j, :] * W[k * n + i, :]
                        W_err[k * n + i, :] -= lr * error * U[i * m + j, :] * V[j * p + k, :]
            
            U += U_err
            V += V_err
            W += W_err
        
        # Check if we achieved a valid decomposition
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-4):
            # Round to nearest half-integer if possible
            U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-3):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            else:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no improvement found, return schoolbook
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, ensure we have a valid baseline
    baseline = _schoolbook(n, m, p)
    
    # Try Winograd-style decomposition
    try:
        result = _winograd_decomposition(n, m, p, deadline)
        # Verify the result
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-3):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return baseline
# EVOLVE-BLOCK-END
