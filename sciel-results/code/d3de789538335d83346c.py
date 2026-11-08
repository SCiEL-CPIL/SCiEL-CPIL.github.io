            try:
                U_new_flat = np.linalg.solve(G, h[..., np.newaxis])
                U_new = U_new_flat.squeeze(-1)
            except np.linalg.LinAlgError:
                U_new = U_t.copy()
