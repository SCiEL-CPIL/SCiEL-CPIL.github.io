import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    # The trivial (schoolbook) decomposition has rank n*m*p = 120
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1
    
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the multiplication tensor using vectorized einsum."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def compute_tensor(U, V, W):
        """Compute the tensor from U, V, W."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

    # Try to find lower rank decompositions using random initialization + gradient descent
    for target_rank in range(119, 114, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(2):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(target_rank * 100 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            target = np.ones((n, m, p))
            lr = 0.005
            
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                T_curr = compute_tensor(U, V, W)
                err = T_curr - target
                loss = np.sum(err ** 2)
                
                if loss < 1e-12:
                    if check_decomposition(U, V, W):
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        found = True
                        break
                
                U_3d = U.reshape(n, m, target_rank)
                V_3d = V.reshape(m, p, target_rank)
                W_3d = W.reshape(p, n, target_rank)
                
                dU = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                dV = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                dW = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
                
                lr_eff = lr / np.sqrt(it + 1)
                U -= lr_eff * dU.reshape(n * m, target_rank)
                V -= lr_eff * dV.reshape(m * p, target_rank)
                W -= lr_eff * dW.reshape(p * n, target_rank)
            
            if found:
                break
    
    if best_rank == trivial_rank:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
    else:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
