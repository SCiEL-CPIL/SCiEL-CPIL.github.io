# EVOLVE-BLOCK-START
import numpy as np
import time

def _is_half_integer(x, tol=1e-6):
    r = 2 * x
    return abs(r - round(r)) < tol

def _snap_to_half_int(x, tol=1e-6):
    r = 2 * x
    if abs(r - round(r)) < tol:
        return round(r) / 2.0
    return x

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for mat in [U_arr, V_arr, W_arr]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not _is_half_integer(mat[i,j], tol):
                    return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - target) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Fallback rank 24
    best_rank = 24
    best_U = [[0] * best_rank for _ in range(dim_U)]
    best_V = [[0] * best_rank for _ in range(dim_V)]
    best_W = [[0] * best_rank for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try rank 14 with Strassen-inspired initialization
    # Strassen 2x2x2 rank 7
    # U_str, V_str, W_str are 4x7
    
    # We need to map 2x3x4 to this structure.
    # If we assume it's two 2x2x2 blocks (conceptually), we can try to construct it.
    # But 3 and 4 don't fit two 2s perfectly.
    # However, the directive says "rank-14".
    
    # Let's try a dense random search for rank 14 with many restarts.
    target_rank = 14
    
    rng = np.random.RandomState(42)
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        U_init = rng.uniform(-1, 1, (dim_U, target_rank))
        V_init = rng.uniform(-1, 1, (dim_V, target_rank))
        W_init = rng.uniform(-1, 1, (dim_W, target_rank))
        
        # Snap to half-integers? No, optimize first.
        U_c = U_init.copy()
        V_c = V_init.copy()
        W_c = W_init.copy()
        
        lr = 0.05
        for step in range(1000):
            if time.time() > deadline - 5:
                break
                
            grad_U = np.zeros_like(U_c)
            grad_V = np.zeros_like(V_c)
            grad_W = np.zeros_like(W_c)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual = 1.0
                        for r in range(target_rank):
                            residual -= U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
                            grad_U[i*m+j, r] += residual * V_c[j*p+k, r] * W_c[k*n+i, r]
                            grad_V[j*p+k, r] += residual * U_c[i*m+j, r] * W_c[k*n+i, r]
                            grad_W[k*n+i, r] += residual * U_c[i*m+j, r] * V_c[j*p+k, r]
            
            U_c -= lr * grad_U
            V_c -= lr * grad_V
            W_c -= lr * grad_W
            
            if step % 100 == 0:
                U_proj = np.vectorize(_snap_to_half_int)(U_c)
                V_proj = np.vectorize(_snap_to_half_int)(V_c)
                W_proj = np.vectorize(_snap_to_half_int)(W_c)
                
                if _verify(U_proj.tolist(), V_proj.tolist(), W_proj.tolist(), n, m, p, tol=1e-4):
                    return {"U": U_proj.tolist(), "V": V_proj.tolist(), "W": W_proj.tolist()}
                
                U_c = U_proj
                V_c = V_proj
                W_c = W_proj
                
            if step % 500 == 0:
                lr *= 0.5
                
        U_final = np.vectorize(_snap_to_half_int)(U_c)
        V_final = np.vectorize(_snap_to_half_int)(V_c)
        W_final = np.vectorize(_snap_to_half_int)(W_c)
        
        if _verify(U_final.tolist(), V_final.tolist(), W_final.tolist(), n, m, p, tol=1e-4):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
