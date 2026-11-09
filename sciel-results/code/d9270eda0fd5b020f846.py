"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the multiplication tensor T[i,j,k] = 1 if i<m, j<n, k<p
    # Actually for n×m * m×p -> n×p: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # The tensor has T[i,j,k] = 1 for all valid indices (it's the identity-like structure)
    # Wait, let me think more carefully.
    # C = A * B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook: rank = n*m*p = 48
    schoolbook_rank = n * m * p
    
    def make_schoolbook(rank):
        """Return schoolbook decomposition as numpy arrays."""
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
    
    def residual(U, V, W):
        """Compute ||T - sum_c U_c * V_c * W_c||_F^2"""
        # T[i,j,k] = 1 for all i,j,k
        # Reconstructed: R[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want to minimize ||R - T||_F^2
        # Efficient: for each c, outer product of U_c reshaped, V_c reshaped, W_c reshaped
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        
        # R is a 3D tensor of shape (n, m, p)
        # R[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        R = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # R[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is: for each (i,k), sum_j u[i,j]*v[j,k]*w[k,i]
            # = diag(w) * (u @ v) ... no, w[k,i] couples i and k
            # Let's just do it element-wise via broadcasting
            # u[i,j] * v[j,k] * w[k,i]
            # Shape: (n,m) * (m,p) * (p,n) -> need to be careful
            # u: (n, m), v: (m, p), w: (p, n)
            # R[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # = sum over nothing, just elementwise
            # Use einsum: 'ij,jk,ki->ijk'
            R += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        T = np.ones((n, m, p))
        return np.sum((R - T) ** 2)
    
    def gradient(U, V, W):
        """Compute gradients of residual w.r.t. U, V, W."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        
        T = np.ones((n, m, p))
        R = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            R += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        E = R - T  # error tensor (n, m, p)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            
            # d/dU[i*m+j, c] of sum_{i',j',k'} E[i',j',k'] * (R - T)[i',j',k']
            # R[i',j',k'] = sum_c' u'[i',j'] * v'[j',k'] * w'[k',i']
            # dR/dU[i,j,c] = v[j,k] * w[k,i] summed over k
            # dRes/dU[i,j,c] = sum_{i',j',k'} E[i',j',k'] * dR[i',j',k']/dU[i,j,c]
            # Only terms where i'=i, j'=j contribute
            # = sum_k E[i,j,k] * v[j,k] * w[k,i]
            
            # dU[i*m+j, c] = sum_k E[i,j,k] * v[j,k] * w[k,i]
            # Using einsum: E[i,j,k] * v[j,k] * w[k,i] -> 'ijk,jk,ki->ij'
            dUc = np.einsum('ijk,jk,ki->ij', E, v, w)
            dU[:, c] = dUc.flatten()
            
            # dV[j*p+k, c] = sum_i E[i,j,k] * u[i,j] * w[k,i]
            # 'ijk,ij,ki->jk'
            dVc = np.einsum('ijk,ij,ki->jk', E, u, w)
            dV[:, c] = dVc.flatten()
            
            # dW[k*n+i, c] = sum_j E[i,j,k] * u[i,j] * v[j,k]
            # 'ijk,ij,jk->ki'
            dWc = np.einsum('ijk,ij,jk->ki', E, u, v)
            dW[:, c] = dWc.flatten()
        
        return dU, dV, dW, residual(U, V, W)
    
    def try_rank(r, max_iter=2000, lr_init=0.01):
        """Try to find a rank-r decomposition."""
        if r >= schoolbook_rank:
            return None  # no point
        
        # Initialize with random small perturbations
        rng = np.random.default_rng(42)
        U = rng.standard_normal((n * m, r)) * 0.1
        V = rng.standard_normal((m * p, r)) * 0.1
        W = rng.standard_normal((p * n, r)) * 0.1
        
        # Add a scaled schoolbook-like structure for better init
        # Use first min(r, schoolbook_rank) columns of schoolbook
        Sb, Svb, Swb = make_schoolbook(schoolbook_rank)
        if r <= schoolbook_rank:
            # Blend: start from schoolbook columns
            U[:, :r] = Sb[:, :r]
            V[:, :r] = Svb[:, :r]
            W[:, :r] = Swb[:, :r]
            # Add small noise to break symmetry
            U += rng.standard_normal((n * m, r)) * 0.01
            V += rng.standard_normal((m * p, r)) * 0.01
            W += rng.standard_normal((p * n, r)) * 0.01
        
        best_res = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = lr_init
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            dU, dV, dW, res = gradient(U, V, W)
            
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate
            if it % 50 == 0:
                lr *= 0.95
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if res < 1e-12:
                break
        
        if best_res < 1e-10:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also, for matrix multiplication, Strassen showed rank < n*m*p for large n
    # For 2x3*3x8, let's try ranks from 24 upward
    
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=500, lr_init=0.05)
        if result is not None:
            U, V, W = result
            # Convert to lists with rounding to half-integers if possible
            U_list = [[round(float(x) * 2) / 2 for x in row] for row in U]
            V_list = [[round(float(x) * 2) / 2 for x in row] for row in V]
            W_list = [[round(float(x) * 2) / 2 for x in row] for row in W]
            
            # Verify
            def verify(Ul, Vl, Wl):
                nm = n * m
                mp = m * p
                pn = p * n
                r = len(Ul[0]) if Ul else 0
                T = np.ones((n, m, p))
                R = np.zeros((n, m, p))
                for c in range(r):
                    u = np.array(Ul)[:, c].reshape(n, m)
                    v = np.array(Vl)[:, c].reshape(m, p)
                    w = np.array(Wl)[:, c].reshape(p, n)
                    R += np.einsum('ij,jk,ki->ijk', u, v, w)
                return np.max(np.abs(R - T))
            
            err = verify(U_list, V_list, W_list)
            if err < 1e-6:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U, V, W = make_schoolbook(schoolbook_rank)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
