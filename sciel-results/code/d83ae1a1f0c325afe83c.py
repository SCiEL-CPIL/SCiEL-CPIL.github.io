<<<<<<< SEARCH
    # Phase 4: Local search
    if time.time() < deadline - 60:
        k = len(best)
        best_set = set(map(tuple, best.tolist()))
        remaining = [tuple(g) for g in grid if tuple(g) not in best_set]
        rng2 = np.random.RandomState(777)

        improved = True
        while improved and time.time() < deadline - 60:
            improved = False
            for idx in range(min(k, 30)):
                if time.time() > deadline - 60:
                    break
                trial = np.delete(best, idx, axis=0)
                trial_sq = (trial.astype(np.float64) ** 2).sum(axis=1) if len(trial) > 0 else np.empty(0)
                best_trial_set = set(map(tuple, trial.tolist()))
                for cand in remaining:
                    if len(trial) >= 500:
                        break
                    if cand in best_trial_set:
                        continue
                    p = np.array(cand, dtype=np.float64)
                    p_sq = (p ** 2).sum()
                    if can_add(trial, p, p_sq, trial_sq):
                        trial = np.vstack([trial, p])
                        trial_sq = np.append(trial_sq, p_sq)
                        best_trial_set.add(cand)
                        remaining.remove(cand)
                        improved = True
                        break
                if len(trial) > len(best):
                    best = trial
                    best_set = set(map(tuple, best.tolist()))
                    remaining = [tuple(g) for g in grid if tuple(g) not in best_set]
                    improved = True
                    break

    # Phase 5: More random restarts
=======
    # Phase 4: Removed due to crash risk. 
    # Phase 5: More random restarts
>>>>>>> REPLACE
