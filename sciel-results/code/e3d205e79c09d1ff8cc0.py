<<<<<<< SEARCH
            # Restart/Escape: Blend with a random base heuristic including Fiedler
            w_d = rng.uniform(-1, 0)
            w_c = rng.uniform(-0.5, 0.5)
            w_s = rng.uniform(-0.5, 0.5)
            w_f = rng.uniform(0, 0.5)
            base = w_d * D_n + w_c * Clus_n + w_s * Spectral_n + w_f * Fiedler_n
            # Blend with current best to stay in the neighborhood
            alpha = rng.uniform(0.5, 1.0)
            new_scores = alpha * best_scores + (1 - alpha) * base
=======
            # Restart/Escape: Blend with a random base heuristic including Fiedler and Removal Degree
            w_d = rng.uniform(-1, 0)
            w_c = rng.uniform(-0.5, 0.5)
            w_s = rng.uniform(-0.5, 0.5)
            w_f = rng.uniform(0, 0.5)
            w_r = rng.uniform(-0.5, 0)  # Negative weight for removal degree
            w_h = rng.uniform(0, 0.3)   # Small positive weight for hub-blocking
            base = w_d * D_n + w_c * Clus_n + w_s * Spectral_n + w_f * Fiedler_n + w_r * RemovalDeg_n + w_h * HubBlock_n
            # Blend with current best to stay in the neighborhood
            alpha = rng.uniform(0.5, 1.0)
            new_scores = alpha * best_scores + (1 - alpha) * base
>>>>>>> REPLACE

