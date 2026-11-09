          # Replace:
          # coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
          # for i in range(num_pw):
          #     for j in range(num_pw):
          #         dx = int(G_diff[i, j, 0])
          #         dy = int(G_diff[i, j, 1])
          #         ix = dx % size
          #         iy = dy % size
          #         coeff_matrix[i, j] = fft_inv_eps[ix, iy]
          
          # With:
          dx = G_diff[:, :, 0]
          dy = G_diff[:, :, 1]
          ix = dx % size
          iy = dy % size
          coeff_matrix = fft_inv_eps[ix, iy]
          