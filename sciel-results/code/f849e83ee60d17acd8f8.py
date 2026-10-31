    def verify(U_mat, V_mat, W_mat):
        """Check if U,V,W exactly decompose T using numpy."""
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Use einsum: 'ir,jr,kr->ijk'
        reconstructed = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
        return np.allclose(reconstructed, T, atol=1e-10)
