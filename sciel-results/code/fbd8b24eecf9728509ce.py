<<<<<<< SEARCH
def _compute_error_and_grad(U, V, W, n, m, p, rank):
    """Compute error tensor and gradients efficiently."""
    err = np.zeros((n, m, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        err[i, j, jp, k] += uc[i*m+j] * vc[jp*p+k] * wc[k*n+i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err[i, j, j, k] -= 1.0
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        e = err[i, j, jp, k]
                        dU[i*m+j, c] += e * vc[jp*p+k] * wc[k*n+i]
                        dV[jp*p+k, c] += e * uc[i*m+j] * wc[k*n+i]
                        dW[k*n+i, c] += e * uc[i*m+j] * vc[jp*p+k]
    return err, dU, dV, dW
=======
def _compute_error_and_grad(U, V, W, n, m, p, rank):
    """Compute error tensor and gradients using vectorized NumPy operations."""
    # Reshape for easier broadcasting
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # err[i, j, jp, k] = sum_c U[i,j,c] * V[jp,k,c] * W[k,i,c]
    # Use einsum: 'ijc,jkc,kic -> ijjpk' no...
    # err[i,j,jp,k] = sum_c U[i,j,c] * V[jp,k,c] * W[k,i,c]
    err = np.einsum('ijc,jkc,kic->ijpk', U_r, V_r, W_r, optimize=True)
    
    # Subtract identity: err[i,j,j,k] -= 1
    idx = np.arange(n)
    err[idx, idx, idx, :] -= 1.0  # This is wrong indexing
    # Actually err has shape (n, m, m, p). We need err[i, j, j, k] -= 1 for all i,j,k
    # Let's use a different approach
    err = np.einsum('ijc,jkc,kic->ijpk', U_r, V_r, W_r, optimize=True)
    # Subtract delta_{j,j'} from err[i,j,j',k]
    identity = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                identity[i, j, j, k] = 1.0
    err -= identity
    
    # Gradients:
    # dU[i,j,c] = sum_{jp,k} err[i,j,jp,k] * V[jp,k,c] * W[k,i,c]
    # dV[jp,k,c] = sum_{i,j} err[i,j,jp,k] * U[i,j,c] * W[k,i,c]
    # dW[k,i,c] = sum_{i,j,jp} err[i,j,jp,k] * U[i,j,c] * V[jp,k,c]
    
    dU = np.einsum('ijpk,jkc,kic->ijc', err, V_r, W_r, optimize=True)
    dV = np.einsum('ijpk,ijc,kic->jkc', err, U_r, W_r, optimize=True)
    dW = np.einsum('ijpk,ijc,jkc->kic', err, U_r, V_r, optimize=True)
    
    return err, dU.reshape(-1, rank), dV.reshape(-1, rank), dW.reshape(-1, rank)
>>>>>>> REPLACE

