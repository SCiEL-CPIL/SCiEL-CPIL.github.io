"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    def schoolbook():
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

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def verify_subtensor(U, V, W, n_sub, m_sub, p_sub):
        """Verify a sub-tensor decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = U_arr[i * m_sub + j, :]
                    v = V_arr[j * p_sub + k, :]
                    w = W_arr[k * n_sub + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_rank_kronecker():
        """
        Use Kronecker product: T_{2,3,9} = T_{2,3,3} ⊗ T_{1,1,3}
        T_{1,1,3} has rank 3 (schoolbook).
        If T_{2,3,3} has rank r, then T_{2,3,9} has rank 3r.
        
        We search for small-rank decompositions of T_{2,3,3} using
        structured random search with half-integer entries.
        """
        if time.time() > deadline - 5:
            return None

        n2, m2, p2 = 2, 3, 3
        # Schoolbook rank for T_{2,3,3} is 18
        # Try ranks from 10 up to 17
        
        for r in range(10, 18):
            if time.time() > deadline - 5:
                return None
            
            # Try multiple random initializations with small half-integer values
            for trial in range(30):
                if time.time() > deadline - 5:
                    return None
                
                np.random.seed(42 + r * 100 + trial)
                
                # Initialize with small half-integer values
                U2 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n2*m2, r))
                V2 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m2*p2, r))
                W2 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p2*n2, r))
                
                # Use gradient descent with L2 regularization to encourage sparsity
                lr = 0.02
                best_loss = float('inf')
                best_U2, best_V2, best_W2 = U2.copy(), V2.copy(), W2.copy()
                
                for it in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute loss and gradients
                    loss = 0.0
                    grad_U2 = np.zeros((n2*m2, r))
                    grad_V2 = np.zeros((m2*p2, r))
                    grad_W2 = np.zeros((p2*n2, r))
                    
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                u_idx = i * m2 + j
                                v_idx = j * p2 + k
                                w_idx = k * n2 + i
                                prod_UV = U2[u_idx, :] * V2[v_idx, :]
                                val = np.dot(prod_UV, W2[w_idx, :])
                                err = val - 1.0
                                loss += err * err
                                grad_U2[u_idx, :] += err * V2[v_idx, :] * W2[w_idx, :]
                                grad_V2[v_idx, :] += err * U2[u_idx, :] * W2[w_idx, :]
                                grad_W2[w_idx, :] += err * U2[u_idx, :] * V2[v_idx, :]
                    
                    # Update
                    U2 -= lr * grad_U2
                    V2 -= lr * grad_V2
                    W2 -= lr * grad_W2
                    
                    # Track best
                    if loss < best_loss:
                        best_loss = loss
                        best_U2, best_V2, best_W2 = U2.copy(), V2.copy(), W2.copy()
                    
                    # Decay learning rate
                    lr *= 0.995
                    
                    if loss < 1e-10:
                        break
                
                # Snap to half-integers and verify
                U2_snap = snap_to_half_int(best_U2)
                V2_snap = snap_to_half_int(best_V2)
                W2_snap = snap_to_half_int(best_W2)
                
                U2_list = U2_snap.tolist()
                V2_list = V2_snap.tolist()
                W2_list = W2_snap.tolist()
                
                if verify_subtensor(U2_list, V2_list, W2_list, n2, m2, p2):
                    # Found T_{2,3,3} with rank r, construct T_{2,3,9} with rank 3r
                    r_total = 3 * r
                    U = [[0] * r_total for _ in range(n*m)]
                    V = [[0] * r_total for _ in range(m*p)]
                    W = [[0] * r_total for _ in range(p*n)]
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                k1 = k % 3
                                k2 = k // 3
                                for c in range(r):
                                    col = c * 3 + k2
                                    U[i*m+j][col] = U2_list[i*m2+j][c]
                                    V[j*p+k][col] = V2_list[j*p2+k1][c]
                                    W[k*n+i][col] = W2_list[k1*n2+i][c]
                    
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
        
        return None

    def try_direct_search(rank):
        """Direct search for T_{2,3,9} with given rank."""
        if time.time() > deadline - 5:
            return None
        
        for trial in range(20):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(1000 + rank * 10 + trial)
            
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, rank))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, rank))
            
            lr = 0.01
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                loss = 0.0
                grad_U = np.zeros((n*m, rank))
                grad_V = np.zeros((m*p, rank))
                grad_W = np.zeros((p*n, rank))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            prod_UV = U[u_idx, :] * V[v_idx, :]
                            val = np.dot(prod_UV, W[w_idx, :])
                            err = val - 1.0
                            loss += err * err
                            grad_U[u_idx, :] += err * V[v_idx, :] * W[w_idx, :]
                            grad_V[v_idx, :] += err * U[u_idx, :] * W[w_idx, :]
                            grad_W[w_idx, :] += err * U[u_idx, :] * V[v_idx, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                lr *= 0.995
                
                if loss < 1e-10:
                    break
            
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try Kronecker product approach first (more likely to succeed)
    res = try_rank_kronecker()
    if res is not None:
        return res

    # Try direct search for ranks less than 54
    for rank in range(30, 54):
        if time.time() > deadline - 5:
            break
        res = try_direct_search(rank)
        if res is not None:
            return res

    return schoolbook()
# EVOLVE-BLOCK-END
