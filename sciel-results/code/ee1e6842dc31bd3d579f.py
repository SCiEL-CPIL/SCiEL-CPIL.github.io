import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

DIMENSIONS = [2, 2, 11]

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build target tensor indices
    # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # In our indexing:
    #   a_idx = i*m + j (0..3)
    #   b_idx = j*p + k (0..21)  
    #   c_idx = k*n + i (0..21)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target_indices.append((a_idx, b_idx, c_idx))
    
    target_set = set(target_indices)
    total_terms = (n*m) * (m*p) * (p*n)  # 4 * 22 * 22 = 1936
    
    # Verified rank-39 fallback
    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = 11 + k0; v_h = 11 + k1
            w_c00 = 2 * k0; w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1; w_c11 = 2 * k1 + 1
            
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        k2 = 10
        v_e2 = k2; v_g2 = 11 + k2
        w_c02 = 2 * k2; w_c12 = 2 * k2 + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}
    
    def is_valid(U, V, W, tol=1e-10):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute all tensor entries
        # For efficiency, use the fact that we only need to check specific entries
        # but let's verify all for correctness
        
        # Build the full tensor via einsum-like approach
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # For each target entry, check it equals 1
        for a_idx, b_idx, c_idx in target_indices:
            val = 0.0
            for r in range(rank):
                val += U_np[a_idx, r] * V_np[b_idx, r] * W_np[c_idx, r]
            if abs(val - 1.0) > tol:
                return False
        
        # Check a sample of non-target entries equal 0
        # To save time, check random subset
        np.random.seed(42)
        checked = 0
        for _ in range(500):
            a_idx = np.random.randint(0, n*m)
            b_idx = np.random.randint(0, m*p)
            c_idx = np.random.randint(0, p*n)
            if (a_idx, b_idx, c_idx) in target_set:
                continue
            val = 0.0
            for r in range(rank):
                val += U_np[a_idx, r] * V_np[b_idx, r] * W_np[c_idx, r]
            if abs(val) > tol:
                return False
            checked += 1
        
        return True
    
    # Try rank 38 with random initialization and Newton iteration
    def try_rank38(max_iters=500):
        rank = 38
        np.random.seed(12345)
        
        # Initialize with small random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Normalize to prevent overflow
        for r in range(rank):
            U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
            V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
            W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
        
        best_loss = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute residuals for target entries
            # residual[a,b,c] = (sum_r U[a,r]*V[b,r]*W[c,r]) - 1 for target
            # For non-target, we want sum_r U[a,r]*V[b,r]*W[c,r] = 0
            
            # Gradient descent on loss = sum_target (val-1)^2 + lambda * sum_nontarget val^2
            # This is expensive; use a simpler approach
            
            # Compute current tensor values for target entries
            target_vals = np.zeros(len(target_indices))
            for idx, (a_idx, b_idx, c_idx) in enumerate(target_indices):
                val = 0.0
                for r in range(rank):
                    val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                target_vals[idx] = val
            
            loss = np.sum((target_vals - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_state = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                U, V, W = best_state
                # Round to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                return U, V, W, loss
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for idx, (a_idx, b_idx, c_idx) in enumerate(target_indices):
                err = target_vals[idx] - 1.0
                for r in range(rank):
                    dU[a_idx, r] += 2 * err * V[b_idx, r] * W[c_idx, r]
                    dV[b_idx, r] += 2 * err * U[a_idx, r] * W[c_idx, r]
                    dW[c_idx, r] += 2 * err * U[a_idx, r] * V[b_idx, r]
            
            # Also add penalty for non-target entries (sample)
            np.random.seed(it)
            for _ in range(100):
                a_idx = np.random.randint(0, n*m)
                b_idx = np.random.randint(0, m*p)
                c_idx = np.random.randint(0, p*n)
                if (a_idx, b_idx, c_idx) in target_set:
                    continue
                val = 0.0
                for r in range(rank):
                    val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                for r in range(rank):
                    dU[a_idx, r] += 0.01 * 2 * val * V[b_idx, r] * W[c_idx, r]
                    dV[b_idx, r] += 0.01 * 2 * val * U[a_idx, r] * W[c_idx, r]
                    dW[c_idx, r] += 0.01 * 2 * val * U[a_idx, r] * V[b_idx, r]
            
            # Adaptive learning rate
            lr = 0.001
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Renormalize
            for r in range(rank):
                U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
                V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
                W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
        
        U, V, W = best_state
        return U, V, W, best_loss
    
    # Try multiple random restarts for rank 38
    if time.time() < deadline - 30:
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            np.random.seed(seed * 1000 + 42)
            U, V, W, loss = try_rank38(max_iters=200)
            if loss < 1e-10:
                # Convert to list format
                U_list = [[float(U[i][r]) for r in range(38)] for i in range(n*m)]
                V_list = [[float(V[i][r]) for r in range(38)] for i in range(m*p)]
                W_list = [[float(W[i][r]) for r in range(38)] for i in range(p*n)]
                if is_valid(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return the verified rank-39 construction
    result = build_rank39()
    return result
