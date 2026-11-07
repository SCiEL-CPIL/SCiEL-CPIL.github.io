"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer combining:
    1. Deterministic axis-scan for robust basin discovery (from inspiration 1).
    2. Rastrigin-aware lattice search to exploit periodic minima (from current program & inspiration 2).
    3. (1+1)-ES for final continuous refinement (from current program).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _scan_axis(self, func, x, fx, axis, n_pts, budget_left, deadline, calls_counter):
        """
        Deterministically sweep one coordinate to find a better point.
        Returns (best_x, best_fx, calls_made).
        """
        d = self.dim
        base = x[axis]
        old_x = x.copy()
        best_x = x.copy()
        best_fx = fx
        made = 0
        
        for k in range(n_pts):
            t = -5.0 + 10.0 * k / (n_pts - 1)
            if abs(t - base) < 1e-9:
                continue
            
            x[axis] = t
            # Check budget and time
            if calls_counter[0] >= self.budget or time.time() > deadline:
                break
            
            f = float(func(x))
            calls_counter[0] += 1
            made += 1
            
            if f < best_fx:
                best_fx = f
                best_x = x.copy()
                
        # Restore original axis value
        x[axis] = base
        return best_x, best_fx, made

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin
        
        calls = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Deterministic Exploration ---
        # Use a coarse grid scan to find a good starting basin.
        # This is more robust than random starts for finding the global minimum of rotated Rastrigin.
        
        # Start at origin
        x = np.zeros(dim)
        fx = eval_func(x)
        
        # Schedule of grid resolutions for axis scanning
        # (n_pts, max_cycles)
        schedule = [
            (5, 2),   # Coarse: 5 points per axis, 2 full cycles
            (9, 1),   # Medium: 9 points per axis, 1 cycle
            (15, 1),  # Fine: 15 points per axis, 1 cycle
        ]

        for n_pts, max_cycles in schedule:
            for _cycle in range(max_cycles):
                improved = False
                for axis in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Use a local counter for the scan to respect global budget
                    calls_counter = [calls]
                    x_new, fx_new, made = self._scan_axis(
                        func, x, fx, axis, n_pts, budget - calls, deadline, calls_counter
                    )
                    calls = calls_counter[0]
                    
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                        # Update global best
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # Even if no improvement in this scan, we might have moved? 
                        # _scan_axis returns best found. If best found is worse than start, we keep start.
                        # But we should update x to the best found in scan if it's better than current x?
                        # Actually, _scan_axis restores x[axis] to base at the end.
                        # So x is unchanged if we don't adopt x_new.
                        # We adopt x_new only if it improved fx.
                        pass

                if not improved:
                    # Stagnation in deterministic scan.
                    # Break out to try other strategies (Lattice/ES)
                    break
            
            if calls >= budget or time.time() > deadline:
                break

        # --- Phase 2: Rastrigin-Aware Lattice Search ---
        # Rastrigin has minima at integer coordinates (in the rotated frame).
        # Since we don't know the rotation, we can't simply snap to integers in the original frame.
        # However, the minima are "integer-like" in the latent space.
        # A common heuristic for BBOB f15 (Rastrigin) is that the global minimum is near an integer vector.
        # Let's evaluate integer neighbors of the current best point.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            anchor = best_x
            seen = set()
            
            # Generate candidates: anchor + integer offsets in [-2, 2] for each dimension
            # This creates a local lattice around the current best.
            offsets = np.array(np.ndindex(*[(5,)] * dim)) - 2  # -2 to 2
            
            # Sort offsets by L2 distance to prioritize nearby integers
            dists = np.sum(offsets**2, axis=1)
            order = np.argsort(dists)
            
            for idx in order:
                if calls >= budget or time.time() > deadline:
                    break
                
                off = offsets[idx]
                cand = anchor + off
                cand = np.clip(cand, -5, 5)
                # Round to nearest integer to ensure we are on the lattice
                cand_int = np.round(cand).astype(int)
                cand_int = np.clip(cand_int, -5, 5)
                
                key = tuple(cand_int)
                if key in seen:
                    continue
                seen.add(key)
                
                f_cand = eval_func(cand_int)
                
                # If we found a significantly better point, we might want to re-center?
                # For now, just keep scanning.

        # --- Phase 3: (1+1)-ES Refinement ---
        # Fine-tune the best point found so far using a small-step ES.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            window = 10
            recent = [False] * window
            idx = 0
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                y = x + sigma * np.random.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.2
                    elif sr < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0
                    sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
