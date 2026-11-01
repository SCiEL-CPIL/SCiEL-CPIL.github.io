"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Decompose the 3x6 multiplication as a direct sum of two 3x3 multiplications.
    For each 2x3 * 3x3 block, use a rank-11 decomposition (if found).
    Total rank = 2 * 11 = 22.
    Falls back to schoolbook (rank 36) if no improvement found.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # First, try to find a rank-11 decomposition for 2x3 * 3x3
    # The tensor is all-ones: T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We need sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    def find_decomposition_2x3x3(target_rank=11):
        """Find a decomposition of the 2x3x3 all-ones tensor with given rank."""
        # Variables: U (2 x r), V (3 x r), W (3 x r)
        # Total variables: 2*r + 3*r + 3*r = 8*r
        # Constraints: 2*3*3 = 18 equations (T[i,j,k] = 1)
        
        # Use random initialization and least squares
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            r = target_rank
            # Initialize with random values
            np.random.seed(attempt)
            U = np.random.randn(2, r) * 0.5
            V = np.random.randn(3, r) * 0.5
            W = np.random.randn(3, r) * 0.5
            
            # Solve using iterative refinement
            # We need: sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
            # This is a system of 18 nonlinear equations in 8*r variables
            
            # Use scipy if available, otherwise simple gradient descent
            try:
                from scipy.optimize import least_squares
            except ImportError:
                # Fallback: simple approach
                return None
            
            def residuals(params):
                U_flat = params[:2*r].reshape(2, r)
                V_flat = params[2*r:5*r].reshape(3, r)
                W_flat = params[5*r:].reshape(3, r)
                
                res = []
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            val = sum(U_flat[i, rr] * V_flat[j, rr] * W_flat[k, rr] for rr in range(r))
                            res.append(val - 1.0)
                return np.array(res)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=1000)
                if result.success or np.max(np.abs(result.fun)) < 1e-10:
                    U_sol = result.x[:2*r].reshape(2, r)
                    V_sol = result.x[2*r:5*r].reshape(3, r)
                    W_sol = result.x[5*r:].reshape(3, r)
                    
                    # Verify
                    max_err = 0
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                val = sum(U_sol[i, rr] * V_sol[j, rr] * W_sol[k, rr] for rr in range(r))
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-10:
                        return (U_sol, V_sol, W_sol)
            except Exception:
                continue
        
        return None
    
    # Try to find rank-11 decomposition for 2x3x3
    decomp_2x3x3 = find_decomposition_2x3x3(11)
    
    if decomp_2x3x3 is not None:
        U1, V1, W1 = decomp_2x3x3
        r_block = 11
        
        # Construct the full 2x3x6 decomposition using two copies
        # First half (k=0,1,2): use first copy
        # Second half (k=3,4,5): use second copy
        r_total = 2 * r_block  # 22
        
        U_full = np.zeros((n*m, r_total))  # (6, 22)
        V_full = np.zeros((m*p, r_total))  # (18, 22)
        W_full = np.zeros((p*n, r_total))  # (12, 22)
        
        # First block: k in {0,1,2}
        for i in range(n):
            for j in range(m):
                U_full[i*m + j, 0:r_block] = U1[i, :]
        
        for j in range(m):
            for k in range(3):
                V_full[j*p + k, 0:r_block] = V1[j, :]
        
        for k in range(3):
            for i in range(n):
                W_full[k*n + i, 0:r_block] = W1[k, :]
        
        # Second block: k in {3,4,5}
        for i in range(n):
            for j in range(m):
                U_full[i*m + j, r_block:2*r_block] = U1[i, :]
        
        for j in range(m):
            for k in range(3):
                V_full[j*p + (k+3), r_block:2*r_block] = V1[j, :]
        
        for k in range(3):
            for i in range(n):
                W_full[(k+3)*n + i, r_block:2*r_block] = W1[k, :]
        
        # Verify the full decomposition
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_full[i*m+j, rr] * V_full[j*p+k, rr] * W_full[k*n+i, rr] for rr in range(r_total))
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
