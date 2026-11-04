        U_mat = U6.reshape(6, 6, -1)
        V_mat = V6.reshape(6, 6, -1)
        W_mat = W6.reshape(6, 6, -1)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
        expected = np.ones((6, 6, 6))
