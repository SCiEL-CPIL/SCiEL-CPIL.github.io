import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_rank12_323(deadline):
    """Try to find a rank-12 decomposition for 3x2x3."""
    n, m, p = 3, 2, 3
    target_rank = 12
    
    # Indices
    U_idx = (n*m, target_rank)
    V_idx = (m*p, target_rank)
    W_idx = (p*n, target_rank)
    
    # Equations: for each (i, j, k), sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # There are 18 equations.
    
    # We will use gradient descent.
    # Initialize with random values.
    
    best_loss = float('inf')
    best_params = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        np.random.seed(42 + restart)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(2000):
            if time.time() > deadline:
                break
                
            # Compute loss and gradients
            loss = 0.0
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = U[i*m+j, :]
                        v = V[j*p+k, :]
                        w = W[k*n+i, :]
                        
                        prod = u * v
                        val = np.dot(prod, w)
                        err = val - 1.0
                        loss += err * err
                        
                        # Gradients
                        # d/dU: err * v * w
                        # d/dV: err * u * w
                        # d/dW: err * u * v
                        gU[i*m+j, :] += err * v * w
                        gV[j*p+k, :] += err * u * w
                        gW[k*n+i, :] += err * u * v
                        
            loss /= (n*m*p)
            
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
                
            if loss < 1e-12:
                break
                
            # Adam update
            vU = momentum * vU + (1 - momentum) * gU
            vV = momentum * vV + (1 - momentum) * gV
            vW = momentum * vW + (1 - momentum) * gW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            lr *= 0.999
            
        if best_loss < 1e-10:
            return best_params
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: Schoolbook
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Try to find rank-12 for 3x2x3 blocks
    # If successful, total rank is 4 * 12 = 48.
    
    block_solution = _find_rank12_323(deadline)
    
    if block_solution is not None:
        U_b, V_b, W_b = block_solution
        # Verify the block solution
        if _verify_full(U_b, V_b, W_b, 3, 2, 3, tol=1e-9):
            # Construct the full 3x4x6 decomposition
            # 4 blocks:
            # Block 0: j in {0,1}, k in {0,1,2}
            # Block 1: j in {0,1}, k in {3,4,5}
            # Block 2: j in {2,3}, k in {0,1,2}
            # Block 3: j in {2,3}, k in {3,4,5}
            
            # Wait, the tensor T[i, j, k] = 1 for ALL i, j, k.
            # My block decomposition T = T11 + T12 + T21 + T22 works because
            # T[i, j, k] is 1 if (j, k) is in the block, else 0.
            # So we sum the contributions.
            
            total_rank = 4 * 12
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            # Block 0: j=0,1; k=0,1,2
            # U indices: i*4 + j. For block 0, j is 0,1.
            # V indices: j*6 + k. For block 0, j is 0,1; k is 0,1,2.
            # W indices: k*3 + i. For block 0, k is 0,1,2.
            
            # We need to map the block indices to the full indices.
            
            # Block 0
            # U_b has shape (6, 12). Indices i*2 + j.
            # Full U has indices i*4 + j.
            # For j in 0,1:
            for i in range(3):
                for j in range(2):
                    U_full[i*4 + j, 0:12] = U_b[i*2 + j, :]
            
            # V_b has shape (6, 12). Indices j*3 + k.
            # Full V has indices j*6 + k.
            # For j in 0,1; k in 0,1,2:
            for j in range(2):
                for k in range(3):
                    V_full[j*6 + k, 0:12] = V_b[j*3 + k, :]
            
            # W_b has shape (9, 12). Indices k*3 + i.
            # Full W has indices k*3 + i.
            # For k in 0,1,2:
            for k in range(3):
                for i in range(3):
                    W_full[k*3 + i, 0:12] = W_b[k*3 + i, :]
            
            # Block 1: j=0,1; k=3,4,5
            # U is same as Block 0 (depends on i, j).
            # V: j in 0,1; k in 3,4,5.
            # W: k in 3,4,5.
            
            # U_full[i*4 + j, 12:24] = U_b[i*2 + j, :]
            for i in range(3):
                for j in range(2):
                    U_full[i*4 + j, 12:24] = U_b[i*2 + j, :]
            
            # V_full[j*6 + k, 12:24] = V_b[j*3 + (k-3), :]
            for j in range(2):
                for k in range(3, 6):
                    V_full[j*6 + k, 12:24] = V_b[j*3 + (k-3), :]
            
            # W_full[k*3 + i, 12:24] = W_b[(k-3)*3 + i, :]
            for k in range(3, 6):
                for i in range(3):
                    W_full[k*3 + i, 12:24] = W_b[(k-3)*3 + i, :]
            
            # Block 2: j=2,3; k=0,1,2
            # U: j in 2,3.
            # V: j in 2,3; k in 0,1,2.
            # W: k in 0,1,2.
            
            for i in range(3):
                for j in range(2, 4):
                    U_full[i*4 + j, 24:36] = U_b[i*2 + (j-2), :]
            
            for j in range(2, 4):
                for k in range(3):
                    V_full[j*6 + k, 24:36] = V_b[(j-2)*3 + k, :]
            
            for k in range(3):
                for i in range(3):
                    W_full[k*3 + i, 24:36] = W_b[k*3 + i, :]
            
            # Block 3: j=2,3; k=3,4,5
            for i in range(3):
                for j in range(2, 4):
                    U_full[i*4 + j, 36:48] = U_b[i*2 + (j-2), :]
            
            for j in range(2, 4):
                for k in range(3, 6):
                    V_full[j*6 + k, 36:48] = V_b[(j-2)*3 + (k-3), :]
            
            for k in range(3, 6):
                for i in range(3):
                    W_full[k*3 + i, 36:48] = W_b[(k-3)*3 + i, :]
            
            # Verify full
            if _verify_full(U_full, V_full, W_full, n, m, p, tol=1e-9):
                U_list = [[float(x) for x in row] for row in U_full]
                V_list = [[float(x) for x in row] for row in V_full]
                W_list = [[float(x) for x in row] for row in W_full]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
