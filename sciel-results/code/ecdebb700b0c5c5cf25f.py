import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Initialize with schoolbook
    U_sb = [[0] * (n*m*p) for _ in range(n*m)]
    V_sb = [[0] * (n*m*p) for _ in range(m*p)]
    W_sb = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_rank = n*m*p
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find rank-23 decomposition for 3x3x3 using alternating minimization
    def find_333_rank23():
        n3 = 3
        target_rank = 23
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1
        
        best_loss = np.inf
        best_U, best_V, best_W = None, None, None
        
        for trial in range(15):
            if time.time() > deadline - 50:
                break
            np.random.seed(trial * 123456789 + 42)
            U = np.random.randn(9, target_rank) * 0.3
            V = np.random.randn(9, target_rank) * 0.3
            W = np.random.randn(9, target_rank) * 0.3
            
            for it in range(80):
                if time.time() > deadline - 50:
                    break
                
                # Alternating least squares
                for c in range(target_rank):
                    # Update U[:, c]
                    A = np.zeros((81, target_rank))
                    for c_idx in range(target_rank):
                        for b in range(9):
                            for d in range(9):
                                A[b*9+d, c_idx] = V[b, c_idx] * W[d, c_idx]
                    try:
                        U[:, c] = np.linalg.lstsq(A, T_target.flatten(), rcond=None)[0]
                    except:
                        pass
                    
                    # Update V[:, c]
                    A = np.zeros((81, target_rank))
                    for c_idx in range(target_rank):
                        for a in range(9):
                            for d in range(9):
                                A[a*9+d, c_idx] = U[a, c_idx] * W[d, c_idx]
                    try:
                        V[:, c] = np.linalg.lstsq(A, T_target.flatten(), rcond=None)[0]
                    except:
                        pass
                    
                    # Update W[:, c]
                    A = np.zeros((81, target_rank))
                    for c_idx in range(target_rank):
                        for a in range(9):
                            for b in range(9):
                                A[a*9+b, c_idx] = U[a, c_idx] * V[b, c_idx]
                    try:
                        W[:, c] = np.linalg.lstsq(A, T_target.flatten(), rcond=None)[0]
                    except:
                        pass
                
                # Compute loss
                T_cur = np.zeros((9, 9, 9))
                for c in range(target_rank):
                    T_cur += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                loss = np.sum((T_cur - T_target)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if loss < 1e-14:
                        # Verify exact decomposition
                        valid = True
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    if abs(T_cur[i*3+j, j*3+k, k*3+i] - 1) > 1e-12:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            for a in range(9):
                                for b in range(9):
                                    for d in range(9):
                                        i1, j1 = a//3, a%3
                                        j2, k1 = b//3, b%3
                                        k2, i2 = d//3, d%3
                                        if i1==i2 and j1==j2 and k1==k2:
                                            continue
                                        if abs(T_cur[a,b,d]) > 1e-12:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if valid:
                                return U, V, W, 23
        return None
    
    result = find_333_rank23()
    if result:
        U3, V3, W3, r3 = result
        total_rank = r3 + 12
        
        U_full = [[0]*total_rank for _ in range(9)]
        V_full = [[0]*total_rank for _ in range(21)]
        W_full = [[0]*total_rank for _ in range(21)]
        
        for i in range(9):
            for c in range(r3):
                U_full[i][c] = U3[i][c]
                V_full[i][c] = V3[i][c]
                W_full[i][c] = W3[i][c]
        
        col_off = r3
        for i in range(3):
            for j in range(3):
                for kg in range(3, 7):
                    U_full[i*3+j][col_off] = 1
                    V_full[j*7+kg][col_off] = 1
                    W_full[kg*3+i][col_off] = 1
                    col_off += 1
        
        # Verify full decomposition
        T = np.zeros((9, 21, 21))
        for c in range(total_rank):
            T += np.einsum('a,b,c->abc', np.array(U_full)[:,c], np.array(V_full)[:,c], np.array(W_full)[:,c])
        
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    if abs(T[i*3+j, j*7+k, k*3+i] - 1) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Sample zero entries
            for _ in range(50):
                a = random.randint(0, 8)
                b = random.randint(0, 20)
                d = random.randint(0, 20)
                i1, j1 = a//3, a%3
                j2, k1 = b//7, b%7
                k2, i2 = d//3, d%3
                if i1==i2 and j1==j2 and k1==k2:
                    continue
                if abs(T[a,b,d]) > 1e-10:
                    valid = False
                    break
        
        if valid:
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
    
    return {"U": best_U, "V": best_V, "W": best_W}
