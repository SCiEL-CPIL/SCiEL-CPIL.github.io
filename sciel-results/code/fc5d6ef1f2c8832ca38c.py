<<<<<<< SEARCH
        # Build points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_final

        # Evaluate all points
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
=======
        # Build points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_final

        # Evaluate all points
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        # Local optimization: adjust f1 values to maximize HV
        # Use a simple iterative approach: for each point, try small perturbations
        # and keep if HV improves
        def compute_hv(f1_vals):
            f2_vals = 1.0 - np.sqrt(f1_vals)
            order = np.argsort(f1_vals)
            f1_s = f1_vals[order]
            f2_s = f2_vals[order]
            # HV = sum (R - f1_i) * (f2_{i-1} - f2_i) with f2_0 = R
            f2_prev = np.concatenate(([ref], f2_s[:-1]))
            hv = np.sum((ref - f1_s) * (f2_prev - f2_s))
            return hv
        
        current_f1 = compute_f1(x1_final)
        current_hv = compute_hv(current_f1)
        
        # Try a few iterations of local refinement
        n_local_iters = 3
        step_size = 0.001
        for it in range(n_local_iters):
            improved = False
            for i in range(n_evals):
                if time.time() > deadline:
                    break
                for delta in [-step_size, step_size]:
                    new_f1 = current_f1.copy()
                    new_f1[i] = current_f1[i] + delta
                    if new_f1[i] < f1_min or new_f1[i] > 1.0:
                        continue
                    new_hv = compute_hv(new_f1)
                    if new_hv > current_hv + 1e-10:
                        current_f1[i] = new_f1[i]
                        current_hv = new_hv
                        improved = True
                        break
                if time.time() > deadline:
                    break
            if not improved:
                break
            step_size *= 0.5
        
        # Re-invert final f1 values to x1
        f1_targets_final = current_f1
        x1_final2 = np.zeros(n_evals)
        x1_from_left2 = np.interp(f1_targets_final, f1_left[::-1], x1_left[::-1])
        x1_from_right2 = np.interp(f1_targets_final, f1_right, x1_right)
        f1_check_left2 = compute_f1(x1_from_left2)
        f1_check_right2 = compute_f1(x1_from_right2)
        err_left2 = np.abs(f1_check_left2 - f1_targets_final)
        err_right2 = np.abs(f1_check_right2 - f1_targets_final)
        use_left2 = err_left2 <= err_right2
        x1_final2[use_left2] = x1_from_left2[use_left2]
        x1_final2[~use_left2] = x1_from_right2[~use_left2]
        
        # Newton refine again
        for _ in range(5):
            fp2 = f1_prime(x1_final2)
            mask2 = np.abs(fp2) > 1e-15
            x1_final2[mask2] -= (compute_f1(x1_final2[mask2]) - f1_targets_final[mask2]) / fp2[mask2]
            x1_final2 = np.clip(x1_final2, 0.0, 1.0)
        
        # Rebuild and re-evaluate
        points2 = np.zeros((n_evals, dim))
        points2[:, 0] = x1_final2
        
        recorded2 = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points2[i]
            f = problem(x)
            recorded2.append(f)
        
        if len(recorded2) > len(recorded):
            return recorded2
        return recorded
>>>>>>> REPLACE
