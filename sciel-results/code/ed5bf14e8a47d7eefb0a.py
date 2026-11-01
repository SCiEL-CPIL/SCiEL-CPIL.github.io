"""Binary search on radius with violation-minimization feasibility checks (L-BFGS-B)."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _violation_loss(x, r):
    """
    Loss = sum of squared overlaps (pairwise) + squared wall penetrations.
    x is shape (2N,), centres are x.reshape(N,2).
    """
    c = x.reshape(N, 2)
    # Wall penalties: for each coordinate, penetration = max(0, |c| - (0.5 - r))
    wall_pen_x = np.maximum(0.0, np.abs(c[:, 0]) - (0.5 - r))
    wall_pen_y = np.maximum(0.0, np.abs(c[:, 1]) - (0.5 - r))
    wall_loss = np.sum(wall_pen_x**2) + np.sum(wall_pen_y**2)

    # Pairwise: for each pair, overlap = max(0, 2r - dist)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    overlaps = np.maximum(0.0, 2.0 * r - dists)
    pair_loss = np.sum(overlaps**2)

    return wall_loss + pair_loss

def _feasible(x, r, tol=1e-8):
    """Check if configuration x (flat) is feasible for radius r."""
    return _violation_loss(x, r) < tol

def _try_radius(r, init_cents, time_budget):
    """
    Try to find a feasible configuration for radius r starting from init_cents.
    Returns (success, best_cents, best_loss).
    """
    x0 = init_cents.reshape(-1).copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    start = time.time()
    max_iter = 2000
    tol = 1e-8
    
    result = minimize(
        _violation_loss, x0, args=(r,),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    elapsed = time.time() - start
    x_best = result.x
    loss_best = _violation_loss(x_best, r)
    
    success = loss_best < tol
    return success, x_best.reshape(N, 2), loss_best

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Step 1: Hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0

    # Step 2: Binary search on radius
    # Lower bound: current hexagonal radius
    # Upper bound: theoretical max ~ 1/(2*sqrt(50)) ~ 0.0707, but use 0.08 as safe upper
    lo = best_r
    hi = 0.08

    # First, try to improve from hexagonal with a slightly larger radius
    # to get a better starting point for binary search
    # Actually, let's just do binary search directly.
    # We need a feasible starting point for each radius.
    # Strategy: keep the last feasible configuration as the start for the next radius.
    # If infeasible, fall back to the hexagonal grid scaled or perturbed.

    feasible_cents = best_cents.copy()

    # Do a few iterations of binary search
    n_bisect = 0
    max_bisect = 15  # Limit number of bisection steps to ensure time
    
    while (time.time() - start_time) < deadline - 20 and n_bisect < max_bisect:
        mid = (lo + hi) / 2.0
        
        # Try to find feasible config for radius mid
        # Use the last feasible config as init if it's close, else use hexagonal
        # If mid > best_r, the hexagonal config is infeasible, so we need a good init.
        # Let's try using the last feasible config (which was feasible for lo) as init.
        # This should be close enough.
        
        success, new_cents, loss = _try_radius(mid, feasible_cents, time_budget=20)
        
        if success:
            lo = mid
            feasible_cents = new_cents
            best_r = mid
            best_cents = new_cents
        else:
            hi = mid
        
        n_bisect += 1

    # Step 3: Local polish using the best feasible config found
    # If we have time, do a quick local optimization on the final config to squeeze out more
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        # Use a simple local optimization on the final best_cents to see if we can improve
        # We'll do a few steps of gradient descent on the "min distance" objective
        # Or just return best_cents if time is tight.
        
        # Let's try a quick L-BFGS-B on the actual score (min distance) to polish
        # But score is non-smooth. Let's stick to the violation method for a slightly larger r.
        
        # Try to push r a bit more from current best_r
        r_push = best_r * 1.001
        if r_push < 0.08:
            success, new_cents, loss = _try_radius(r_push, best_cents, time_budget=min(remaining - 5, 15))
            if success:
                best_r = r_push
                best_cents = new_cents

    # Ensure we return the best valid config
    # If best_r is still the initial hexagonal, return that.
    # Otherwise return the best found.
    
    # Final safety check: ensure all points are within bounds
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
# EVOLVE-BLOCK-END
