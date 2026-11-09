<<<<<<< SEARCH
        # --- Phase 1: Random sampling for initial diversity ---
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            update_best(x, f)
            if time.time() > deadline:
                break
=======
        # --- Phase 1: Quasi-random sampling for initial diversity ---
        # Use a Halton-like sequence for better coverage
        n_init = min(30, budget // 8)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        for idx in range(1, n_init + 1):
            if time.time() > deadline:
                break
            x = np.array([2.0 * halton(idx, base) - 5.0 for base in [2, 3, 5, 7, 11][:dim]])
            f = func(x)
            update_best(x, f)
        
        # Add a few random points for stochastic coverage
        n_random = min(10, budget - calls_made) if 'calls_made' in dir() else min(10, budget - n_init)
        calls_made = n_init
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)
>>>>>>> REPLACE

