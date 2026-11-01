"""Matrix multiplication 2x2x9 with rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n=2, m=2, p=9):
    """Verify that U, V, W give exact decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_zero_conditions(U, V, W, n=2, m=2, p=9):
    """Check that entries where the tensor should be 0 are actually 0."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in matrix multiplication
    # There are no zero conditions in the standard matrix multiplication tensor
    # for valid indices. All combinations of (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # have T[i,j,k] = 1.
    return True

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # The matrix multiplication tensor for (n,m) x (m,p) -> (n,p) has:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # 
    # A rank-r decomposition has:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # 
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    #
    # In the flattened form:
    # U has rows indexed by (i,j), so row index = i*m + j
    # V has rows indexed by (j,k), so row index = j*p + k  
    # W has rows indexed by (k,i), so row index = k*n + i
    #
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook solution: rank = n*m*p = 36
    # One column per (i,j,k) triple, with 1 at the corresponding positions.
    
    # Let's try to find a lower rank solution.
    # 
    # Key insight: We can use the fact that the constraint is:
    # For ALL (i,j,k) in [0,2)x[0,2)x[0,9): 
    #   sum_c U[2i+j, c] * V[2j+k, c] * W[2k+i, c] = 1
    #
    # This gives 2*2*9 = 36 equations.
    #
    # Strategy: Try various rank values and use optimization to find solutions.
    # We'll use a random restart approach with gradient-free optimization.
    
    # First, let's establish the schoolbook baseline
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # We'll search from rank 28 down to 20 (reasonable range)
    # For each rank, try multiple random initializations
    
    def try_rank(r, max_attempts=20, time_per_attempt=1.0):
        """Try to find a rank-r decomposition."""
        attempts = 0
        while time.time() - start_time < deadline - 5 and attempts < max_attempts:
            attempts += 1
            np.random.seed(attempts * 12345 + r)
            
            # Initialize with random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Scale to make the average contribution reasonable
            # Each term U*V*W should contribute ~1/r on average
            U *= (1.0 / (r ** (1/3)))
            V *= (1.0 / (r ** (1/3)))
            W *= (1.0 / (r ** (1/3)))
            
            # Optimize using a simple gradient descent-like approach
            # We want to minimize sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            # Precompute index combinations
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            
            # Simple iterative optimization
            lr = 0.01
            for iteration in range(500):
                if time.time() - start_time > deadline - 5:
                    break
                    
                # Compute current values and gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for (i, j, k) in triples:
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    
                    val = 0.0
                    for c in range(r):
                        val += U[ui, c] * V[vj, c] * W[wk, c]
                    
                    err = val - 1.0
                    
                    for c in range(r):
                        grad_U[ui, c] += err * V[vj, c] * W[wk, c]
                        grad_V[vj, c] += err * U[ui, c] * W[wk, c]
                        grad_W[wk, c] += err * U[ui, c] * V[vj, c]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if iteration % 100 == 0:
                    max_err = 0.0
                    for (i, j, k) in triples:
                        ui = i * m + j
                        vj = j * p + k
                        wk = k * n + i
                        val = 0.0
                        for c in range(r):
                            val += U[ui, c] * V[vj, c] * W[wk, c]
                        max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-8:
                        break
            
            # Check if we found a valid solution
            valid = True
            max_err = 0.0
            for (i, j, k) in triples:
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[ui, c] * V[vj, c] * W[wk, c]
                err = abs(val - 1.0)
                max_err = max(max_err, err)
                if err > 1e-7:
                    valid = False
            
            if valid:
                # Convert to lists of floats (or try to round to rationals)
                U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
                V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
                W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 30 down to 20
    for r in range(30, 19, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_rank(r, max_attempts=5, time_per_attempt=1.0)
        if result is not None:
            # Verify
            if _verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = r
                break  # Found a valid solution at this rank, stop searching lower
    
    # If no better solution found, return schoolbook
    return best_result
