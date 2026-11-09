            if np.isfinite(energy) and energy < best_energy:
                # Check validity
                min_d = np.inf
                for a in range(size):
                    for b in range(a+1, size):
                        d = np.linalg.norm(positions[a] - positions[b])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    best_energy = energy
                    best_positions = positions.copy()
