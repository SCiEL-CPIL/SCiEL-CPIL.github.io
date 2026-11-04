"""Matrix multiplication 3x4x6 - model-and-optimize approach with Adam and structured search."""
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
    deadline = 145.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    def try_rank(r, max_iter=10000, lr_init=0.01, n_restarts=10, tol=1e-8):
        """Try to find a rank-r decomposition using Adam optimizer."""
        best_result = None
        best_loss = float('inf')
        
        # Precompute indices for gradients if needed, but einsum is fast enough
        # We will use Adam optimizer
        
        for restart in range(n_restarts):
            if time.time() > deadline - 2:
                break
            
            # Initialization strategy:
            # 1. Random
            # 2. Trivial + Noise (if r == n*m*p, but here r < n*m*p so we can't use trivial directly as init shape mismatch)
            # We can initialize with random small values.
            
            np.random.seed(42 + restart * 1000 + r)
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            Ut = U.reshape(n, m, r)
            Vt = V.reshape(m, p, r)
            Wt = W.reshape(p, n, r)
            
            # Adam state
            m_U = np.zeros_like(Ut)
            v_U = np.zeros_like(Ut)
            m_V = np.zeros_like(Vt)
            v_V = np.zeros_like(Vt)
            m_W = np.zeros_like(Wt)
            v_W = np.zeros_like(Wt)
            
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            t = 0
            lr = lr_init
            
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                t += 1
                
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
                
                # Gradients
                dU = np.einsum('ijk,jkc,kic->ijc', err, Vt, Wt) * 2
                dV = np.einsum('ijk,ijc,kic->jkc', err, Ut, Wt) * 2
                dW = np.einsum('ijk,ijc,jkc->kic', err, Ut, Vt) * 2
                
                # Adam update
                m_U = b1 * m_U + (1 - b1) * dU
                v_U = b2 * v_U + (1 - b2) * (dU ** 2)
                m_hat_U = m_U / (1 - b1 ** t)
                v_hat_U = v_U / (1 - b2 ** t)
                
                m_V = b1 * m_V + (1 - b1) * dV
                v_V = b2 * v_V + (1 - b2) * (dV ** 2)
                m_hat_V = m_V / (1 - b1 ** t)
                v_hat_V = v_V / (1 - b2 ** t)
                
                m_W = b1 * m_W + (1 - b1) * dW
                v_W = b2 * v_W + (1 - b2) * (dW ** 2)
                m_hat_W = m_W / (1 - b1 ** t)
                v_hat_W = v_W / (1 - b2 ** t)
                
                Ut -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
                Vt -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
                Wt -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                
                # Decay LR slightly if loss stagnates? Adam handles this somewhat, but let's add a simple decay
                if it > 5000:
                    lr = lr_init * 0.5
            
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
    
    def polish(U, V, W, r, max_iter=5000, lr=0.001, tol=1e-10):
        """High-precision polish pass using simple GD with tiny steps."""
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
            
            # Decay LR
            if it % 1000 == 0 and it > 0:
                lr *= 0.5
        
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        if np.max(np.abs(T - 1.0)) < 1e-6:
            return Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r)
        return None
    
    # Try ranks from low to high, but prioritize 71 as it's the most likely improvement
    # The trivial rank is 72.
    # We will try 71, 70, 69...
    # If we find a solution for 71, we stop.
    
    ranks_to_try = [71, 70, 69, 68, 67, 66, 65, 64]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=5000, lr_init=0.02, n_restarts=5)
        if result is not None:
            U, V, W = result
            # Polish the solution
            polished = polish(U, V, W, r, max_iter=2000, lr=0.0005, tol=1e-12)
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
