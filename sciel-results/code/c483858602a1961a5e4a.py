def compute_grads(U, V, W, n, m, p, R):
    # Reshape to 3D tensors for easier manipulation?
    # U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R)
    
    U_t = U.reshape(n, m, R)
    V_t = V.reshape(m, p, R)
    W_t = W.reshape(p, n, R)
    
    # Compute T = sum_c U_ijk V_jk W_ki
    # We need T_ijk = 1 for all i,j,k.
    
    # grad_U_ijk = sum_c 2 * (T_ijk - 1) * V_jk_c * W_ki_c
    # This is a bit tricky to vectorize fully without einsum.
    
    # Let's use einsum
    # T_ijk = sum_c U_ijk_c V_jk_c W_ki_c
    # Note: V is indexed by j,k. W by k,i.
    
    # V_t is (m, p, R). We need V_jk for fixed j,k.
    # W_t is (p, n, R). We need W_ki for fixed k,i.
    
    # Sum over c:
    # prod = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    # This computes the full tensor T.
    
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Error E_ijk = T_ijk - 1
    E = T - 1.0
    
    # Gradients
    # dL/dU_ijk = sum_c 2 * E_ijk * V_jk_c * W_ki_c
    # This is: 2 * E_ijk * (V_t[j,k,:] * W_t[k,i,:]).sum(axis=-1) ? No.
    # It's a sum over c.
    # grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', E, V_t, W_t)
    
    grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', E, V_t, W_t)
    grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', E, U_t, W_t)
    grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', E, U_t, V_t)
    
    return grad_U.reshape(n*m, R), grad_V.reshape(m*p, R), grad_W.reshape(p*n, R)
