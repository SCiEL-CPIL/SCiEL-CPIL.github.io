    def _jitter_polish(pos, radii, n_restarts=5, max_iter=100):
        """
        Perform multiple SLSQP polishes on jittered positions to escape local minima.
        """
        if time_left() < 15:
            return pos, radii
        
        best_val = np.sum(radii)
        best_p = pos.copy()
        best_r = radii.copy()
        
        rng = np.random.RandomState(int(time.time() * 1000))
        
        for k in range(n_restarts):
            if time_left() < 10:
                break
            
            # Jitter positions
            jitter_scale = 0.01
            p_jitter = pos + rng.normal(0, jitter_scale, pos.shape)
            # Clip to [0,1]
            p_jitter = np.clip(p_jitter, 0.0, 1.0)
            
            # Run SLSQP on jittered positions
            # We reuse _slsqp_polish but with fewer iterations
            p_new, r_new = _slsqp_polish(p_jitter, radii, max_iter=max_iter)
            
            # Check validity
            if _is_valid(p_new, r_new, tol=1e-9):
                val = np.sum(r_new)
                if val > best_val:
                    best_val = val
                    best_p = p_new
                    best_r = r_new
        
        return best_p, best_r
