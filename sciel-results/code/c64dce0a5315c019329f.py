                updates = np.where(is_true_old, -1, 1)
                lit_true_counts[c_idx_arr] += updates
                unsat_count = int((lit_true_counts == 0).sum())
