import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 110
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Step 1: Find a low-rank decomposition for 3x3x3
    # Tensor T333[i*3+j, j*3+k, k*3+i] = 1
    
    def get_333_tensor():
        T = np.zeros((9, 9, 9), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T[i*3 + j, j*3 + k, k*3 + i] = 1.0
        return T

    T333 = get_333_tensor()
    
    def find_333_decomposition(target_rank, max_restarts=50):
        """
        Attempts to find a rank target_rank decomposition for 3x3x3 tensor.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline - 5:
            return None
            
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = np.random.randint(-3, 4, size=(9, target_rank)) * 0.5
            V = np.random.randint(-3, 4, size=(9, target_rank)) * 0.5
            W = np.random.randint(-3, 4, size=(9, target_rank)) * 0.5
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed V, W: T[a, :, :] = sum_r U[a, r] (V[:, r] * W[:, r])
                # Design matrix D_U shape (81, R)
                D_U = np.zeros((81, target_rank))
                for r in range(target_rank):
                    D_U[:, r] = (V[:, r].reshape(9, 1) * W[:, r].reshape(1, 9)).reshape(-1)
                
                U_new = np.zeros((9, target_rank))
                for a in range(9):
                    t_a = T333[a, :, :].reshape(-1)
                    try:
                        U_new[a, :] = np.linalg.lstsq(D_U, t_a, rcond=None)[0]
                    except:
                        break
                U = U_new
                
                # Update V
                D_V = np.zeros((81, target_rank))
                for r in range(target_rank):
                    D_V[:, r] = (U[:, r].reshape(9, 1) * W[:, r].reshape(1, 9)).reshape(-1)
                
                V_new = np.zeros((9, target_rank))
                for b in range(9):
                    t_b = T333[:, b, :].reshape(-1)
                    try:
                        V_new[b, :] = np.linalg.lstsq(D_V, t_b, rcond=None)[0]
                    except:
                        break
                V = V_new
                
                # Update W
                D_W = np.zeros((81, target_rank))
                for r in range(target_rank):
                    D_W[:, r] = (U[:, r].reshape(9, 1) * V[:, r].reshape(1, 9)).reshape(-1)
                
                W_new = np.zeros((9, target_rank))
                for c in range(9):
                    t_c = T333[:, :, c].reshape(-1)
                    try:
                        W_new[c, :] = np.linalg.lstsq(D_W, t_c, rcond=None)[0]
                    except:
                        break
                W = W_new
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check exactness
            err = 0.0
            for r in range(target_rank):
                outer = U_r[:, r].reshape(9, 1, 1) * V_r[:, r].reshape(1, 9, 1) * W_r[:, r].reshape(1, 1, 9)
                err += np.max(np.abs(T333 - outer))
            
            if err < 1e-9:
                return U_r, V_r, W_r
                
        return None

    # Try to find rank 19, 20, 21, 22, 23
    U333 = None
    V333 = None
    W333 = None
    rank333 = 27 # Fallback schoolbook
    
    for r in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        res = find_333_decomposition(r, max_restarts=20)
        if res is not None:
            U333, V333, W333 = res
            rank333 = r
            break

    # If no better found, use schoolbook for 3x3x3 (Rank 27)
    if U333 is None:
        U333 = np.zeros((9, 27))
        V333 = np.zeros((9, 27))
        W333 = np.zeros((9, 27))
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U333[i*3+j, c] = 1.0
                    V333[j*3+k, c] = 1.0
                    W333[k*3+i, c] = 1.0
                    c += 1
        rank333 = 27

    # Step 2: Construct 3x3x6 decomposition from 3x3x3
    total_rank = 2 * rank333
    U = np.zeros((9, total_rank))
    V = np.zeros((18, total_rank))
    W = np.zeros((18, total_rank))
    
    # First half: C1 = A * B1
    U[:, :rank333] = U333
    
    # V mapping:
    # V333 index: j*3 + k_local
    # V_total index: j*6 + k_global
    # For C1, k_global in 0,1,2. k_local = k_global.
    for j in range(3):
        for k in range(3):
            V[j*6 + k, :rank333] = V333[j*3 + k, :]
    
    # W mapping:
    # W333 index: k_local*3 + i
    # W_total index: k_global*3 + i
    # For C1, k_global in 0,1,2.
    for k in range(3):
        for i in range(3):
            W[k*3 + i, :rank333] = W333[k*3 + i, :]

    # Second half: C2 = A * B2
    U[:, rank333:] = U333
    
    # V mapping:
    # For C2, k_global in 3,4,5. k_local = k_global - 3.
    for j in range(3):
        for k in range(3, 6):
            k_local = k - 3
            V[j*6 + k, rank333:] = V333[j*3 + k_local, :]
    
    # W mapping:
    # For C2, k_global in 3,4,5.
    for k in range(3, 6):
        k_local = k - 3
        for i in range(3):
            W[k*3 + i, rank333:] = W333[k_local*3 + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
