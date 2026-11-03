import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # Schoolbook decomposition (Rank 36)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    def verify(U, V, W, tol=1e-9):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)

    def verify_half_integer(U, V, W):
        if not (np.allclose(U * 2, np.round(U * 2)) and
                np.allclose(V * 2, np.round(V * 2)) and
                np.allclose(W * 2, np.round(W * 2))):
            return False
        return verify(U, V, W)

    # Try to find rank 10 decomposition for 2x3x2
    # If found, construct rank 30 for 2x3x6
    def find_rank_10_232():
        # Dimensions for 2x3x2
        n2, m2, p2 = 2, 3, 2
        R = 10
        U2 = np.random.randn(n2 * m2, R)
        V2 = np.random.randn(m2 * p2, R)
        W2 = np.random.randn(p2 * n2, R)
        
        # Simple gradient descent
        lr = 0.01
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            U2_r = U2.reshape(n2, m2, R)
            V2_r = V2.reshape(m2, p2, R)
            W2_r = W2.reshape(p2, n2, R)
            
            # T2[i,j,k] = sum_r U2[i,j,r] V2[j,k,r] W2[k,i,r]
            T2 = np.einsum('ijr,jkr,kir->ijk', U2_r, V2_r, W2_r)
            err = T2 - 1.0
            
            # Gradients
            # dT/dU[i,j,r] = V2[j,k,r] W2[k,i,r] (sum over k)
            grad_U = np.einsum('jkr,kir,ijk->ijr', V2_r, W2_r, err)
            grad_V = np.einsum('ijr,kir,ijk->jkr', U2_r, W2_r, err)
            grad_W = np.einsum('ijr,jkr,ijk->kir', U2_r, V2_r, err)
            
            U2 -= lr * grad_U
            V2 -= lr * grad_V
            W2 -= lr * grad_W
            
            # Check convergence
            if np.max(np.abs(err)) < 1e-5:
                # Try to round to half-integers
                U2_h = np.round(U2 * 2) / 2
                V2_h = np.round(V2 * 2) / 2
                W2_h = np.round(W2 * 2) / 2
                if verify_232(U2_h, V2_h, W2_h, n2, m2, p2):
                    return U2_h, V2_h, W2_h
        return None

    def verify_232(U, V, W, n2, m2, p2, tol=1e-9):
        U_r = U.reshape(n2, m2, -1)
        V_r = V.reshape(m2, p2, -1)
        W_r = W.reshape(p2, n2, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)

    # Attempt search
    found_232 = find_rank_10_232()
    if found_232 is not None:
        U2, V2, W2 = found_232
        # Construct 2x3x6 from 2x3x2
        # p=6 is 3 blocks of p2=2
        n3, m3, p3 = 2, 3, 6
        R3 = 30
        U3 = np.zeros((n3 * m3, R3))
        V3 = np.zeros((m3 * p3, R3))
        W3 = np.zeros((p3 * n3, R3))
        
        for b in range(3):
            # Copy U
            U3[:, b*10:(b+1)*10] = U2
            # Copy V and W with index shift
            # V indices: j (0..2), k (0..5)
            # V2 indices: j (0..2), k2 (0..1)
            # For block b, k = b*2 + k2
            for j in range(m3):
                for k2 in range(p2):
                    k = b * p2 + k2
                    V3[j * p3 + k, b*10:(b+1)*10] = V2[j * p2 + k2, :]
            # W indices: k (0..5), i (0..1)
            # W2 indices: k2 (0..1), i (0..1)
            for k2 in range(p2):
                for i in range(n3):
                    k = b * p2 + k2
                    W3[k * n3 + i, b*10:(b+1)*10] = W2[k2 * n2 + i, :]
        
        if verify(U3, V3, W3):
            if verify_half_integer(U3, V3, W3):
                best_U, best_V, best_W = U3, V3, W3
                best_rank = 30

    # Return best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
