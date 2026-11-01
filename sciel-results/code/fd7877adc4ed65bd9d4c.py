            gains = coverage @ uncovered
            max_gain = np.max(gains)
            if max_gain == 0: break
            best_indices = np.where(gains == max_gain)[0]
            best_idx = best_indices[rng.integers(len(best_indices))]
            current_blocks.append(best_idx)
            uncovered &= (1 - coverage[best_idx])
