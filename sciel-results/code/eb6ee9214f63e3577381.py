import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    schoolbook_rank = n * m * p  # 48
    
    def make_schoolbook(rank):
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def residual(U, V, W):
        r = U.shape[1]
        R = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            R += np.einsum('ij,jk,ki->ijk', u, v, w)
        T = np.ones((n, m, p))
        return np.sum((R - T) ** 2)
    
    def gradient(U, V, W):
        r = U.shape[1]
        T = np.ones((n, m, p))
        R = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            R += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        E = R - T
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            
            dUc = np.einsum('ijk,jk,ki->ij', E, v, w)
            dU[:, c] = dUc.flatten()
            
            dVc = np.einsum('ijk,ij,ki->jk', E, u, w)
            dV[:, c] = dVc.flatten()
            
            dWc = np.einsum('ijk,ij,jk->ki', E, u, v)
            dW[:, c] = dWc.flatten()
        
        return dU, dV, dW, residual(U, V, W)
    
    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integer and verify."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        r = U_snap.shape[1]
        R = np.zeros((n, m, p))
        for c in range(r):
            u = U_snap[:, c].reshape(n, m)
            v = V_snap[:, c].reshape(m, p)
            w = W_snap[:, c].reshape(p, n)
            R += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        T = np.ones((n, m, p))
        err = np.max(np.abs(R - T))
        return U_snap, V_snap, W_snap, err
    
    def try_rank(r, num_restarts=20, max_iter=2000):
        """Try to find a rank-r decomposition."""
        if r >= schoolbook_rank:
            return None
        
        best_result = None
        best_err = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.default_rng(restart * 1000 + 42)
            
            # Initialize with random values
            U = rng.standard_normal((nm, r)) * 0.5
            V = rng.standard_normal((mp, r)) * 0.5
            W = rng.standard_normal((pn, r)) * 0.5
            
            # Try gradient descent
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                dU, dV, dW, res = gradient(U, V, W)
                
                # Adaptive learning rate
                if it % 100 == 0 and it > 0:
                    lr *= 0.9
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if res < 1e-10:
                    break
            
            # Snap to half-integers and verify
            U_snap, V_snap, W_snap, err = snap_to_half_integers(U, V, W)
            
            if err < 1e-6 and err < best_err:
                best_err = err
                best_result = (U_snap, V_snap, W_snap)
        
        if best_result is not None:
            return best_result
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)  # 24
    
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, num_restarts=10, max_iter=1000)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U, V, W = make_schoolbook(schoolbook_rank)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
