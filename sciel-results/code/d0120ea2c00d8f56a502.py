            # h_sigma calculation
            norm_Cs = np.linalg.norm(Cs)
            t_gen = calls / lam
            if norm_Cs / np.sqrt(1 - (1 - c_c) ** t_gen) < (1.4 + 2 / (d + 1)) * (1 - (1 - c_c) ** t_gen):
                h_sig = 1.0
            else:
                h_sig = 0.0
