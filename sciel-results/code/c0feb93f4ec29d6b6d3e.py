"""Program for finding a low-rank tensor decomposition for 2x3 x 3x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (12, rank)  
    W has shape (p*n, rank) = (8, rank)
    
    The decomposition satisfies: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try to find a low-rank decomposition using a constructive approach
    # based on algebraic structure of the tensor.
    
    # First, let's try rank 20 with a systematic construction
    # The key insight is that we can group terms cleverly.
    
    # Let me try a direct construction for rank 20.
    # We have 24 equations (one for each (i,j,k) triple) and need to satisfy them.
    
    # Actually, let me use a computational approach: try to find a valid decomposition
    # by solving a system of equations.
    
    # For a given rank r, we have:
    # U: 6 x r, V: 12 x r, W: 8 x r
    # Constraint: for all (i,j,k): sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    # This is a system of 24 equations in 6*r + 12*r + 8*r = 26*r unknowns.
    # For r=12: 312 unknowns, 24 equations - very underdetermined
    # For r=20: 520 unknowns, 24 equations - very underdetermined
    
    # The challenge is that the equations are cubic (products of three variables).
    # Let me try a different approach: use the schoolbook decomposition as a starting
    # point and try to merge columns.
    
    # Schoolbook decomposition has rank 24. Let me see if I can reduce it.
    
    # Actually, let me just try to construct a valid decomposition computationally.
    # I'll use a random initialization and gradient descent, or try specific patterns.
    
    # Let me try a structured approach:
    # For each j in 0..2, create a block of columns that handle all (i,k) pairs.
    
    # For fixed j, we need: for all i in 0..1, k in 0..3:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me try using 5 columns per j-value (total 15), plus 5 more for cross-terms.
    
    # Actually, let me just try to find ANY valid decomposition with rank < 24.
    # I'll use a simple iterative approach.
    
    def try_decomposition(rank, seed=42):
        """Try to find a valid decomposition with the given rank."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # We need: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let me define the residual for each (i,j,k):
        # res[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
        
        def compute_residuals(U, V, W):
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        res[i, j, k] = val - 1
            return res
        
        # This is a non-linear system. Let me try a different approach:
        # Fix U and V, solve for W (linear), then fix U and W, solve for V, etc.
        
        # For fixed U and V, the equation for W is:
        # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is linear in W. Let me set up the linear system.
        
        # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..rank-1
        # Total: p*n*rank = 8*rank variables
        
        # Equations: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Total: n*m*p = 24 equations
        
        # But the equations are not independent in W. Let me group by (k,i):
        # For fixed (k,i), the equations involving W[k*n+i, r] are for all j:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of m equations in rank unknowns (W[k*n+i, r] for r in 0..rank-1).
        
        # For each (k,i), we have a linear system: A_{k,i} * w_{k,i} = b
        # where A_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r] and b[j] = 1
        # and w_{k,i}[r] = W[k*n+i, r]
        
        # So we can solve for W given U and V!
        
        def solve_W(U, V):
            W = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # Build the linear system
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    b = np.ones(m)
                    # Solve A * w = b (least squares if rank > m)
                    if rank >= m:
                        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    else:
                        # Underdetermined, use minimum norm solution
                        w = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[k*n+i, :] = w
            return W
        
        # Similarly, for fixed U and W, solve for V:
        # For fixed (j,k), equations for all i:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is linear in V[j*p+k, r].
        
        def solve_V(U, W):
            V = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    b = np.ones(n)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = w
            return V
        
        # And for fixed V and W, solve for U:
        # For fixed (i,j), equations for all k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Linear in U[i*m+j, r].
        
        def solve_U(V, W):
            U = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    b = np.ones(p)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = w
            return U
        
        # Now use alternating minimization:
        # Initialize U, V, W randomly
        # Iterate: solve W given U,V; solve V given U,W; solve U given V,W
        
        best_U, best_V, best_W = None, None, None
        best_res = np.inf
        
        for iteration in range(200):
            W = solve_W(U, V)
            V = solve_V(U, W)
            U = solve_U(V, W)
            
            res = compute_residuals(U, V, W)
            res_norm = np.max(np.abs(res))
            
            if res_norm < best_res:
                best_res = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res_norm < 1e-8:
                break
        
        # Check if we got a valid solution
        res = compute_residuals(best_U, best_V, best_W)
        if np.max(np.abs(res)) < 1e-6:
            # Round to half-integers if possible
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            res_rounded = compute_residuals(U_rounded, V_rounded, W_rounded)
            if np.max(np.abs(res_rounded)) < 1e-6:
                return U_rounded, V_rounded, W_rounded
            else:
                return best_U, best_V, best_W
        
        return None
    
    # Try ranks from 12 to 23
    for rank in range(12, 24):
        for seed in range(10):
            result = try_decomposition(rank, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition (rank 24)
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
