<<<<<<< SEARCH
            X = allX[new_idx]
            F = allF[new_idx]

        return X
=======
            X = allX[new_idx]
            F = allF[new_idx]

        # Local polish: apply Nelder-Mead to top non-dominated points
        if time.time() < deadline and used < self.budget:
            fronts = non_dominated_sort(F)
            # Get top 5 non-dominated points for local search
            top_points = []
            for fr in fronts[:3]:
                if len(top_points) >= 5:
                    break
                for idx in fr:
                    if len(top_points) < 5:
                        top_points.append(idx)
            
            for idx in top_points:
                if used >= self.budget or time.time() >= deadline:
                    break
                x0 = X[idx].copy()
                f0 = F[idx].copy()
                
                # Simple local search: perturb each dimension
                for d in range(self.dim):
                    if used >= self.budget or time.time() >= deadline:
                        break
                    for step in [0.05, 0.02, 0.01]:
                        if used >= self.budget or time.time() >= deadline:
                            break
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() >= deadline:
                                break
                            x_try = x0.copy()
                            x_try[d] += direction * step * span[d]
                            x_try = np.clip(x_try, lo, hi)
                            f_try = np.array(problem(x_try))
                            used += 1
                            # Check if f_try dominates f0
                            if np.all(f_try <= f0) and np.any(f_try < f0):
                                x0 = x_try
                                f0 = f_try
                            elif np.all(f0 <= f_try) and not np.any(f0 < f_try):
                                # f0 dominates or equals f_try, keep x0
                                pass
                            elif not np.all(f_try <= f0) and not np.all(f0 <= f_try):
                                # Incomparable - keep both by updating population
                                pass
                        if time.time() >= deadline:
                            break
                    if time.time() >= deadline:
                        break
                
                # Update population with improved point
                if np.all(f0 < F[idx]) or (np.all(f0 <= F[idx]) and np.any(f0 < F[idx])):
                    X[idx] = x0
                    F[idx] = f0

        return X
>>>>>>> REPLACE
