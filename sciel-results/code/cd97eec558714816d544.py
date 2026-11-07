<<<<<<< SEARCH
    def _sobol_2d(self, n, seed=0):
        """Generate n 2D points using a scrambled-like lattice (independent random shifts
        per dimension plus a low-discrepancy base grid). Returns (n, 2) array in [0,1)^2."""
        # Use a simple 2D low-discrepancy construction: combine two 1D low-discrepancy
        # sequences (van der Corput / Halton) with independent random shifts to decorrelate.
        # This gives a 2D sequence with good spread and no visible grid alignment.
        rng = np.random.RandomState(seed)
        shift0 = rng.random()
        shift1 = rng.random()

        def halton_seq(n_pts, base):
            """Return n_pts Halton values in [0,1)."""
            out = np.empty(n_pts)
            for i in range(n_pts):
                idx = i + 1
                f = 1.0
                r = 0.0
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                out[i] = r
            return out

        h2 = halton_seq(n, 2)
        h3 = halton_seq(n, 3)
        x0 = (h2 + shift0) % 1.0
        x1 = (h3 + shift1) % 1.0
        return np.column_stack((x0, x1))
=======
    def _maxmin_facility_location(self, n, seed=0, n_iter=50):
        """Generate n 2D points using max-min facility location to maximize minimum
        pairwise distance. Returns (n, 2) array in [0,1)^2."""
        rng = np.random.RandomState(seed)
        
        # Start with a few random points
        points = np.zeros((n, 2))
        points[0] = rng.random(2)
        
        for i in range(1, n):
            # Generate many candidate points
            n_candidates = 200
            candidates = rng.random((n_candidates, 2))
            
            # For each candidate, compute the minimum distance to existing points
            # Vectorized: for each candidate, find min dist to all placed points
            # dist[j, k] = ||candidates[j] - points[k]||
            # min_dist[j] = min_k dist[j, k]
            
            # Use broadcasting: candidates[:, np.newaxis, :] - points[np.newaxis, :, :]
            # Shape: (n_candidates, n_placed, 2)
            diffs = candidates[:, np.newaxis, :] - points[:i, np.newaxis, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n_candidates, i)
            min_dists = np.min(dists, axis=1)  # (n_candidates,)
            
            # Pick the candidate with maximum minimum distance
            best_idx = np.argmax(min_dists)
            points[i] = candidates[best_idx]
        
        return points
>>>>>>> REPLACE

