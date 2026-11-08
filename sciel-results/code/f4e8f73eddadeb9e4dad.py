import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nU = n * m  # 6
    nV = m * p  # 9
    nW = p * n  # 6
    
    # Build target tensor
    T = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Schoolbook solution (rank 18)
    rank_sb = n * m * p
    U_sb = np.zeros((nU, rank_sb))
    V_sb = np.zeros((nV, rank_sb))
    W_sb = np.zeros((nW, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_and_round(U, V, W, T, atol=1e-6):
        """Check if decomposition is valid and round to half-integers."""
        # Check if entries are close to half-integers
        for mat in [U, V, W]:
            rounded = np.round(mat * 2) / 2.0
            if np.max(np.abs(mat - rounded)) > 1e-4:
                return False
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T, atol=atol)
    
    def try_rank(R, max_restarts=30):
        """Try to find a valid decomposition of rank R with momentum-based optimization."""
        if time.time() > deadline - 5:
            return False, None, None, None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            # Initialize with different strategies
            if restart == 0:
                # Sparse random initialization
                U = np.random.randn(nU, R) * 0.5
                V = np.random.randn(nV, R) * 0.5
                W = np.random.randn(nW, R) * 0.5
            elif restart == 1:
                # Binary-like initialization
                U = np.random.choice([0, 1], size=(nU, R)) * 0.5
                V = np.random.choice([0, 1], size=(nV, R)) * 0.5
                W = np.random.choice([0, 1], size=(nW, R)) * 0.5
            elif restart == 2:
                # Small random
                U = np.random.randn(nU, R) * 0.1
                V = np.random.randn(nV, R) * 0.1
                W = np.random.randn(nW, R) * 0.1
            else:
                scale = 0.1 + 0.1 * restart / max_restarts
                U = np.random.randn(nU, R) * scale
                V = np.random.randn(nV, R) * scale
                W = np.random.randn(nW, R) * scale
            
            # Momentum-based gradient descent
            lr = 0.03
            momentum = 0.9
            best_err = np.inf
            stuck_count = 0
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(3000):
                if time.time() > deadline - 2:
                    break
                
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err_tensor = T_recon - T
                err = np.sum(err_tensor**2)
                
                if err < 1e-8:
                    break
                
                if err > best_err * 1.5:
                    lr *= 0.5
                elif err < best_err * 0.8:
                    lr = min(lr * 1.1, 0.05)
                    best_err = err
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                if stuck_count > 300:
                    break
                
                # Gradients
                dU = np.einsum('ijk,kr,jr->ir', err_tensor, W, V)
                dV = np.einsum('ijk,ir,kr->jr', err_tensor, U, W)
                dW = np.einsum('ijk,ir,jr->kr', err_tensor, U, V)
                
                # Update with momentum
                vU = momentum * vU + (1 - momentum) * dU
                vV = momentum * vV + (1 - momentum) * dV
                vW = momentum * vW + (1 - momentum) * dW
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
            
            # Final check
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-5):
                if check_and_round(U, V, W, T):
                    return True, U, V, W
        
        return False, None, None, None
    
    # Try to find rank 17
    if time.time() < deadline - 20:
        success, U17, V17, W17 = try_rank(17, max_restarts=25)
        if success:
            best_rank = 17
            best_U = U17
            best_V = V17
            best_W = W17
            print(f"Found rank 17 solution!", file=sys.stderr)
    
    # Try rank 16 if time permits
    if best_rank == 18 and time.time() < deadline - 15:
        success, U16, V16, W16 = try_rank(16, max_restarts=20)
        if success:
            best_rank = 16
            best_U = U16
            best_V = V16
            best_W = W16
            print(f"Found rank 16 solution!", file=sys.stderr)
    
    # Convert to lists with half-integer rounding
    best_U = np.round(best_U * 2) / 2.0
    best_V = np.round(best_V * 2) / 2.0
    best_W = np.round(best_W * 2) / 2.0
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification
    R_final = len(U_list[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] for r in range(R_final))
                if abs(val - 1.0) > 1e-6:
                    # Fallback to schoolbook
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
