            # Symmetric matrix H
            # H[i,j] = eps_inv_diff[i,j] * |k+G_i| * |k+G_j|
            # |k+G_i| is sqrt(k_g_sq[i])
            k_g_mags = np.sqrt(k_g_sq)
            H = eps_inv_diff * k_g_mags[:, None] * k_g_mags[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.real(eigvals)
                # Filter positive real eigenvalues
                eigvals = eigvals[eigvals > 1e-6]
                if len(eigvals) >= 5:
                    eigvals = np.sort(eigvals)
                    omega4_sq = eigvals[3]
                    omega5_sq = eigvals[4]
                    
                    if omega4_sq > max_omega4_sq:
                        max_omega4_sq = omega4_sq
                    if omega5_sq < min_omega5_sq:
                        min_omega5_sq = omega5_sq
            except:
                pass
