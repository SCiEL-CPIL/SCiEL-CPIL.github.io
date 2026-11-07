"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate descent with golden-section search for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within an instance
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Track number of evaluations used
        eval_count = 1

        # Golden-section search parameters
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
        inv_phi = 1.0 / phi
        inv_phi2 = 1.0 / (phi * phi)

        # For each coordinate, we'll do a golden-section search over the full range [-5, 5]
        # But we need to be careful with the budget.
        # Strategy: multiple passes over coordinates, each pass doing a 1D search.
        # Each 1D golden-section search takes ~30-40 evaluations to converge well.
        # With dim=5, one full pass = 5 * 40 = 200 evals.
        # Budget is 500, so we can do ~2 full passes plus some refinement.

        # Actually, let's think more carefully. Golden section search on [-5, 5] 
        # with tolerance ~1e-6 requires about log(1e-6/log(phi)) / log(phi) ≈ 30-35 iterations.
        # But we don't need that much precision. Let's use a fixed number of iterations per coordinate.

        # Let's do a smarter approach: 
        # Phase 1: Coarse coordinate descent - for each coordinate, do a 1D search with ~20 iterations
        # Phase 2: Fine coordinate descent - for each coordinate, do a 1D search with ~15 iterations
        # Total: 5 * 20 + 5 * 15 = 175 evals, leaving room for more passes or random restarts.

        # Actually, let's just do iterative coordinate descent with golden-section,
        # continuing until budget is exhausted.

        # Golden section search on coordinate i, given current x and fx
        def golden_search_coord(coord_idx, x_current, fx_current, max_iter=30):
            nonlocal eval_count, best_x, best_fx
            
            # We search over x[coord_idx] in [-5, 5], keeping other coords fixed
            a, b = -5.0, 5.0
            # Initial points
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            
            # Evaluate at c and d
            x_c = x_current.copy()
            x_c[coord_idx] = c
            x_d = x_current.copy()
            x_d[coord_idx] = d
            
            fc = func(x_c)
            fd = func(x_d)
            eval_count += 2
            
            if fc < best_fx:
                best_fx = fc
                best_x = x_c.copy()
            if fd < best_fx:
                best_fx = fd
                best_x = x_d.copy()
            
            # Keep track of best point found during this search
            local_best_x = x_current.copy()
            local_best_fx = fx_current
            if fc < local_best_fx:
                local_best_fx = fc
                local_best_x = x_c.copy()
            if fd < local_best_fx:
                local_best_fx = fd
                local_best_x = x_d.copy()
            
            for _ in range(max_iter - 1):
                if eval_count >= budget:
                    break
                    
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - inv_phi * (b - a)
                    x_c = local_best_x.copy()
                    x_c[coord_idx] = c
                    fc = func(x_c)
                    eval_count += 1
                    if fc < local_best_fx:
                        local_best_fx = fc
                        local_best_x = x_c.copy()
                        if fc < best_fx:
                            best_fx = fc
                            best_x = x_c.copy()
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + inv_phi * (b - a)
                    x_d = local_best_x.copy()
                    x_d[coord_idx] = d
                    fd = func(x_d)
                    eval_count += 1
                    if fd < local_best_fx:
                        local_best_fx = fd
                        local_best_x = x_d.copy()
                        if fd < best_fx:
                            best_fx = fd
                            best_x = x_d.copy()
            
            return local_best_x, local_best_fx

        # Main loop: iterate through coordinates repeatedly
        # Use the best point found so far as the base for the next coordinate search
        current_x = best_x.copy()
        current_fx = best_fx

        while eval_count < budget:
            # Determine how many iterations to use for this coordinate search
            # Use fewer iterations as we get closer to budget
            remaining = budget - eval_count
            if remaining < 5:
                break
            # Use up to 25 iterations per coordinate, but not more than remaining/2
            max_iter = min(25, max(5, remaining // 2))
            
            for i in range(dim):
                if eval_count >= budget:
                    break
                current_x, current_fx = golden_search_coord(i, current_x, current_fx, max_iter)
            
            # If we haven't improved in a full pass, we might want to try a different strategy
            # But for now, just continue

        # Final return
        return best_x, best_fx
# EVOLVE-BLOCK-END
