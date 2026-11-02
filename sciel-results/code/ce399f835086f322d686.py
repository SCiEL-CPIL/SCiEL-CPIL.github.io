"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try to find rank-minimal decomposition using linear programming
    # The matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a non-linear problem (trilinear), so we use a rank-1 search approach
    # Start with schoolbook and try to reduce rank
    
    # First, get the schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to verify if rank can be reduced
    # For 2x5x6, the lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But this is just a lower bound; actual rank may be higher
    
    # Use a greedy rank-1 approximation approach
    # Target: T[i,j,k] = 1 for all valid (i,j,k)
    
    def residual(U, V, W, rank):
        """Compute the maximum absolute residual of the tensor decomposition."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    max_res = max(max_res, abs(val - 1.0))
        return max_res
    
    # Try random initializations with decreasing rank
    # For small problems like 2x5x6, we can afford some search
    np.random.seed(42)
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 30
    # But let's try a few ranks near the known bounds
    
    # Known results: for 2x5 matrix multiplication, rank is at most 8
    # For 2x5x6, we can use Kronecker product: rank(2x5) * rank(1x6) or similar
    # Actually, we can use: rank(2x5x6) <= rank(2x5) * 6 or 5 * rank(1x6)
    
    # Let's try to use known low-rank results
    # For 2x3: rank 7 (Strassen-like)
    # For 2x5: rank 8 is known optimal
    
    # For 2x5x6, we can try:
    # 1. Use rank-8 2x5 decomposition combined with 6 ones: rank = 8*6 = 48
    # 2. Or use 5 rank-1 terms for first contraction * 6 rank-1 for second
    
    # Let's try to find if rank < 60 is achievable with random search
    # We'll do a quick random search for small rank improvements
    
    for target_rank in [48, 45, 42, 40, 36, 33, 30]:
        if time.time() > deadline:
            break
        
        # Random search for this rank
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # Initialize with small random values
            U_r = np.random.randn(n*m, target_rank) * 0.1
            V_r = np.random.randn(m*p, target_rank) * 0.1
            W_r = np.random.randn(p*n, target_rank) * 0.1
            
            # Local optimization: gradient descent
            lr = 0.01
            for it in range(200):
                # Compute gradient
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = 0.0
                            for r in range(target_rank):
                                diff += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            diff -= 1.0
                            for r in range(target_rank):
                                dU[i*m+j, r] += diff * V_r[j*p+k, r] * W_r[k*n+i, r]
                                dV[j*p+k, r] += U_r[i*m+j, r] * diff * W_r[k*n+i, r]
                                dW[k*n+i, r] += U_r[i*m+j, r] * V_r[j*p+k, r] * diff
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                # Check convergence
                if it % 50 == 0:
                    res = residual(U_r.tolist(), V_r.tolist(), W_r.tolist(), target_rank)
                    if res < 1e-6:
                        found = True
                        break
            
            if found:
                U_r = U_r.tolist()
                V_r = V_r.tolist()
                W_r = W_r.tolist()
                # Snap to nearest half-integer
                for r_idx in range(target_rank):
                    for i_idx in range(len(U_r)):
                        U_r[i_idx][r_idx] = round(U_r[i_idx][r_idx] * 2) / 2
                    for j_idx in range(len(V_r)):
                        V_r[j_idx][r_idx] = round(V_r[j_idx][r_idx] * 2) / 2
                    for k_idx in range(len(W_r)):
                        W_r[k_idx][r_idx] = round(W_r[k_idx][r_idx] * 2) / 2
                
                # Verify after snapping
                if residual(U_r, V_r, W_r, target_rank) < 1e-4:
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    break
    
    if time.time() < deadline:
        # If we found something better, use it
        if best_rank < schoolbook_rank:
            return best_result
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
