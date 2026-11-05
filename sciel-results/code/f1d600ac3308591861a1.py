                scores = M.dot(uncovered)
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                if len(candidates) > 1:
                    best_c = np.random.choice(candidates)
                else:
                    best_c = candidates[0]
