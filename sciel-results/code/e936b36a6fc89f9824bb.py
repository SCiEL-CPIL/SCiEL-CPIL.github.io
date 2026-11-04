def objective_and_grad(x, R, n, m, p):
    U = x[:n*m*R].reshape(n, m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
    W = x[n*m*R + m*p*R :].reshape(p, n, R)
    
    T = np.zeros((n, m, p))
    for r in range(R):
        T += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
        
    error = T - 1.0
    loss = np.sum(error * error)
    
    grad = np.zeros_like(x)
    
    # grad_U[i, j, r] = sum_k 2 * error[i, j, k] * V[j, k, r] * W[k, i, r]
    # grad_V[j, k, r] = sum_i 2 * error[i, j, k] * U[i, j, r] * W[k, i, r]
    # grad_W[k, i, r] = sum_j 2 * error[i, j, k] * U[i, j, r] * V[j, k, r]
    
    # Vectorized computation
    # For U:
    # grad_U[i, j, r] = 2 * sum_k error[i, j, k] * V[j, k, r] * W[k, i, r]
    # We can use einsum.
    # error is (n, m, p)
    # V is (m, p, R)
    # W is (p, n, R)
    
    # grad_U[i, j, r] = 2 * einsum('ijk,jkr,kr->ijr', error, V, W)
    # Note: W index is (k, i, r). So W[k, i, r].
    # In einsum: 'ijk, jkr, kir -> ijr' ? No, W is stored as (p, n, R) i.e. (k, i, r).
    # So W[k, i, r] corresponds to index 0=k, 1=i, 2=r.
    # So: np.einsum('ijk,jkr,kir->ijr', error, V, W)
    
    grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', error, V, W)
    
    # grad_V[j, k, r] = 2 * sum_i error[i, j, k] * U[i, j, r] * W[k, i, r]
    # np.einsum('ijk,ijr,kir->jkr', error, U, W)
    grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', error, U, W)
    
    # grad_W[k, i, r] = 2 * sum_j error[i, j, k] * U[i, j, r] * V[j, k, r]
    # np.einsum('ijk,ijr,jkr->kir', error, U, V)
    grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', error, U, V)
    
    grad[:n*m*R] = grad_U.flatten()
    grad[n*m*R : n*m*R + m*p*R] = grad_V.flatten()
    grad[n*m*R + m*p*R :] = grad_W.flatten()
    
    return loss, grad
