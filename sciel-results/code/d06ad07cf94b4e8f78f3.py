def obj_tensor(c):
    # I2 = c^T A c. A is approx Identity.
    # Actually, let's compute A too.
    # A[i,j] = sum_alpha w_alpha Y[i,alpha] Y[j,alpha]
    A = np.einsum('ia,ja,a->ij', Y, Y, weights, optimize=True)
    
    # I4 = sum c_i c_j c_k c_l T[i,j,k,l]
    # I4 = np.einsum('i,j,k,l,ijkl->', c, c, c, c, T)
    
    # Grad I4: 4 * sum_j k l c_j c_k c_l T[i,j,k,l]
    # G[i] = 4 * np.einsum('j,k,l,jkl->i', c, c, c, T[i,:,:,:]) ? No.
    # G = 4 * np.einsum('jkl,ijkl->i', c@c@c, T) ?
    # c_outer = np.einsum('i,j->ij', c, c)
    # c_outer2 = np.einsum('i,j->ij', c, c)
    # term = np.einsum('ij,kl,ijkl->i', c_outer, c_outer, T)
    # grad_I4 = 4 * term
    
    # Grad I2: 2 * A @ c
    
    # R = (I4 / I2^2) ** 0.25 ? No.
    # R = I4^0.25 / I2^0.5
    # Minimize R.
