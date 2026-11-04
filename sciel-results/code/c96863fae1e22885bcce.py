"""Matrix multiplication 3x4x6 - model-and-optimize approach."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol


def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    def try_rank(r, max_iter=5000, lr_init=0.01, lr_decay=0.999, min_lr=0.0001,
                 n_restarts=5, tol=1e-8):
        """Try to find a rank-r decomposition using gradient descent with adaptive LR."""
        best_result = None
        best_loss = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 2:
                break
            
            # Different initializations for each restart
            np.random.seed(42 + restart * 1000)
            scale = 0.5 / np.sqrt(r)
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            Ut = U.reshape(n, m, r)
            Vt = V.reshape(m, p, r)
            Wt = W.reshape(p, n, r)
            
            lr = lr_init
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                # Compute T = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
                err = T - 1.0
                loss = np.sum(err ** 2)
                
                if loss < tol:
                    result = (Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r))
                    if loss < best_loss:
                        best_loss = loss
                        best_result = result
                    break
                
                # Adaptive learning rate: reduce if loss increases
                if loss > prev_loss:
                    lr = max(lr * 0.5, min_lr)
                else:
                    lr = min(lr * 1.05, lr_init)
                
                prev_loss = loss
                
                # Gradients
                dU = np.einsum('ijk,jkc,kic->ijc', err, Vt, Wt) * 2
                dV = np.einsum('ijk,ijc,kic->jkc', err, Ut, Wt) * 2
                dW = np.einsum('ijk,ijc,jkc->kic', err, Ut, Vt) * 2
                
                # Gradient normalization for stability
                gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if gnorm > 0:
                    dU /= gnorm
                    dV /= gnorm
                    dW /= gnorm
                
                Ut -= lr * dU
                Vt -= lr * dV
                Wt -= lr * dW
            
            # Check final result
            T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
            final_loss = np.sum((T - 1.0) ** 2)
            if final_loss < best_loss:
                best_loss = final_loss
                if final_loss < tol:
                    best_result = (Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r))
        
        if best_result is not None and best_loss < tol:
            U, V, W = best_result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
        return None
    
    def polish(U, V, W, r, max_iter=2000, lr=0.001, tol=1e-10):
        """High-precision polish pass."""
        Ut = U.reshape(n, m, r)
        Vt = V.reshape(m, p, r)
        Wt = W.reshape(p, n, r)
        
        for it in range(max_iter):
            if time.time() > deadline - 1:
                break
            
            T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
            err = T - 1.0
            loss = np.sum(err ** 2)
            
            if loss < tol:
                break
            
            dU = np.einsum('ijk,jkc,kic->ijc', err, Vt, Wt) * 2
            dV = np.einsum('ijk,ijc,kic->jkc', err, Ut, Wt) * 2
            dW = np.einsum('ijk,ijc,jkc->kic', err, Ut, Vt) * 2
            
            Ut -= lr * dU
            Vt -= lr * dV
            Wt -= lr * dW
        
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        if np.max(np.abs(T - 1.0)) < 1e-6:
            return Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r)
        return None
    
    # Try ranks from low to high
    for r in range(54, 72):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=4000, lr_init=0.02, n_restarts=3)
        if result is not None:
            U, V, W = result
            # Polish the solution
            polished = polish(U, V, W, r, max_iter=1000, lr=0.0005, tol=1e-12)
            if polished is not None:
                U, V, W = polished
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break
    
    if best_rank < n * m * p:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
