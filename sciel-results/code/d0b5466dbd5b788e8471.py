"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = 140  # seconds, leaving 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # 
    # This is equivalent to: for each (i,k), the matrix M_{ik}[j1,j2] = sum_c V[j1*p+k,c]*W[k*n+i,c]*U[i*m+j1,c]... 
    # Actually let me reformulate.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]... 
    # No, that's not right.
    #
    # The correct formulation: We want
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c]) * W[k*n+i, c]
    # Wait, that's not right either. Let me be more careful.
    #
    # Standard tensor decomposition for matrix mult:
    # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c]) * W[k*n+i, c]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] summed over c = delta(j is the correct index)
    # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k] means the coefficient of A[i,j]*B[j,k] in C[i,k] must be 1.
    #
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a system of n*m*p = 40 equations in 3*rank*appropriate_size variables.
    
    # Let's try ranks from low to high
    # Lower bound: The tensor has slice ranks that give lower bounds.
    # For fixed i, the slice is an m x p matrix of ones, rank 1.
    # For fixed j, the slice is an n x p matrix of ones, rank 1.  
    # For fixed k, the slice is an n x m matrix of ones, rank 1.
    # But the overall tensor rank is not simply bounded by these.
    
    # Let's try to find a decomposition computationally.
    # We'll use a randomized approach with local search.
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
        # Equations: for each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a nonlinear system. Let's use a different approach.
        # Fix W and solve for U, V? Or use alternating minimization?
        
        # Alternative: parametrize differently.
        # Let's use the fact that this is a bilinear system.
        
        # Try a structured ansatz. For 2x4x5, maybe we can find a rank-20 or lower solution.
        
        # Let me try a direct linear algebra approach.
        # Flatten the constraint: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is trilinear. Hard to solve directly.
        
        # Let's try: assume W entries are from {0, 1, -1} or similar simple values.
        # Or try a grid search for small ranks.
        
        # Actually, let me try a different formulation.
        # Define for each c: u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
        # Constraint: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
        
        # Let's try rank 20 first (half of schoolbook).
        
        rng = np.random.default_rng(42)
        
        # Try alternating least squares or similar
        # Initialize
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        # Scale to make initial residual reasonable
        # Target: each equation sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Average term magnitude ~ 1/rank, so scale by sqrt(rank)
        scale = np.sqrt(rank)
        U *= scale
        V *= scale
        W *= scale
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residual += (val - 1.0)**2
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-12:
                break
            
            # Alternate: fix two, solve for third
            # Fix V, W, solve for U: for each (i,j), sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for all k
            # This is: for each (i,j), U[i*m+j, :] * diag(V[j*p+k,:]*W[k*n+i,:]) ... not quite.
            # For fixed (i,j), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j, :]:
            # Let M_{k,c} = V[j*p+k,c] * W[k*n+i,c]
            # Then sum_c M_{k,c} * U[i*m+j,c] = 1 for all k
            # So M @ U[i*m+j, :]^T = ones(p)
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve M @ x = ones(p)
                    try:
                        x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Fix U, W, solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n*m, rank))  # Wait, for fixed (j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
                    # So for fixed (j,k): sum_c V[j*p+k,c] * (U[i*m+j,c]*W[k*n+i,c]) = 1 for all i
                    # M_{i,c} = U[i*m+j,c] * W[k*n+i,c], size n x rank
                    M = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Fix U, V, solve for W
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i): sum_c W[k*n+i,c] * (U[i*m+j,c]*V[j*p+k,c]) = 1 for all j
                    # M_{j,c} = U[i*m+j,c] * V[j*p+k,c], size m x rank
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
        
        return best_residual, best_U, best_V, best_W
    
    # Try ranks from low to high
    # Lower bound considerations: 
    # The tensor T is the all-ones tensor of shape 2x4x5.
    # Its rank is at least max(slice ranks) but that's not tight.
    # For the all-ones tensor of shape a x b x c, the rank is min(a*b, a*c, b*c, ...) ? No.
    # Actually the all-ones tensor of shape a x b x c has rank min(a*b, a*c, b*c) ? Let me think.
    # No, that's not right either. The all-ones tensor J_{a,b,c} can be written as sum of rank-1 tensors.
    # J_{a,b,c} = sum_{i=1}^{min(a,b,c)} e_i^{(a)} \otimes e_i^{(b)} \otimes e_i^{(c)} ? No, that only works if a=b=c.
    # 
    # Actually J_{a,b,c}[i,j,k] = 1 for all i,j,k.
    # We can write it as: for each i, J[i,:,:] = ones(b,c) = 1_b \otimes 1_c
    # So J = sum_i e_i \otimes 1_b \otimes 1_c, which has rank a.
    # Similarly rank b or c. So rank <= min(a,b,c).
    # But can we do better? The rank is exactly min(a,b,c) for the all-ones tensor? 
    # Actually no. Let me reconsider.
    #
    # Wait, I think I'm confusing myself. Let me re-derive.
    #
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # This is NOT the all-ones tensor in the (i,j,k) space in a simple way because the indices are coupled.
    #
    # Let me try a small rank. Start with rank 10 (= p*n = 10, the output size).
    
    for rank in [10, 12, 15, 16, 18, 20, 24, 28, 30, 32, 36, 40]:
        if time.time() > deadline - 10:
            break
        
        residual, U, V, W = try_rank(rank)
        
        if residual < 1e-10:
            # Found a valid decomposition!
            # Round to nearest integers or half-integers if possible
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            res2 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                        res2 += (val - 1.0)**2
            
            if res2 < 1e-8:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            else:
                # Use the unrounded values
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: return schoolbook
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
# EVOLVE-BLOCK-END
