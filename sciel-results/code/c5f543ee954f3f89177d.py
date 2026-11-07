            if stagnation_count > stagnation_limit:
                # Only restart if we have enough budget for NP-1 evaluations
                if evals_used + NP - 1 <= self.budget:
                    # Restart
                    pop[best_idx] = x_best
                    fit[best_idx] = f_best
                    for i in range(NP):
                        if i != best_idx:
                            if evals_used >= self.budget:
                                break
                            pop[i] = x_best + rng.uniform(-2, 2, self.dim)
                            pop[i] = np.clip(pop[i], -5, 5)
                            fit[i] = func(pop[i])
                            evals_used += 1
                            if fit[i] < f_best:
                                f_best = fit[i]
                                x_best = pop[i].copy()
                                best_idx = i
                    stagnation_count = 0
