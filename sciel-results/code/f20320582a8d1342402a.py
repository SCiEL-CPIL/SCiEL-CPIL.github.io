                if len(u1_idxs) > 0:
                    first_u1 = u1_idxs[0]
                    cand1 = np.where(cov_matrix[:, first_u1] == 1)[0]
                    cand1 = [c for c in cand1 if c not in sel_set]
                    # Filter: must cover all u1_idxs
                    cand1 = [c for c in cand1 if (cov_matrix[c, u1_idxs] == 1).all()]
                else:
                    cand1 = [c for c in range(num_k) if c not in sel_set]
