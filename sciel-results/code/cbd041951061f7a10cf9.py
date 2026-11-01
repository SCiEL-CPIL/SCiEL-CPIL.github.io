"""
Strategy: Force-directed repulsion model with Simulated Annealing.
Instead of minimizing a smooth proxy objective via gradient descent (L-BFGS-B),
we model the packing as a physical system where circles repel each other and
the walls with a force proportional to 1/d^2. We then use Simulated Annealing
to find a low-energy configuration. The energy function is:
E = sum_{i<j} 1/d_ij^2 + sum_i (1/wall_x_i^2 + 1/wall_y_i^2)
This energy is minimized when all distances are large. Simulated Annealing
allows the system to jump over energy barriers, escaping local minima that
trap gradient-based methods.
"""
import numpy as np
import time

N = 50

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
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

def _energy(cents):
    """
    Compute the total repulsion energy.
    E = sum_{i<j} 1/d_ij^2 + sum_i (1/wall_x_i^2 + 1/wall_y_i^2)
    Lower energy = larger minimum distances.
    """
    n = len(cents)
    eps = 1e-12
    
    # Pairwise repulsion
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(d_sq, 1.0)  # Avoid div by zero
    # Only count each pair once (i<j)
    tri_idx = np.triu_indices(n, k=1)
    pair_energy = np.sum(1.0 / (d_sq[tri_idx] + eps))
    
    # Wall repulsion
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    wall_energy = np.sum(1.0 / (wx + eps)**2) + np.sum(1.0 / (wy + eps)**2)
    
    return pair_energy + wall_energy

def _force(cents):
    """
    Compute the net repulsive force on each circle.
    Force is the negative gradient of the energy.
    F_i = -dE/dc_i
    """
    n = len(cents)
    eps = 1e-12
    
    # Pairwise forces
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(d_sq, 1.0)
    
    # dE/dx_i = sum_{j != i} -2*(x_i - x_j) / (d_ij^2 + eps)^2
    # Force = -dE/dx_i = sum_{j != i} 2*(x_i - x_j) / (d_ij^2 + eps)^2
    weights = 2.0 / (d_sq + eps)**2
    weights[np.diag_indices(n)] = 0.0
    
    # F_c[i] = sum_j weights[i,j] * (c[i] - c[j])
    sum_weights = weights.sum(axis=1)
    weighted_c = weights @ cents
    force_c = cents * sum_weights[:, np.newaxis] - weighted_c
    
    # Wall forces
    # dE/dx_i = 2*sign(x_i) / (0.5 - |x_i|)^3
    # F_x_i = -dE/dx_i = -2*sign(x_i) / (0.5 - |x_i|)^3
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    force_c[:, 0] -= 2.0 * np.sign(cents[:, 0]) / (wx + eps)**3
    force_c[:, 1] -= 2.0 * np.sign(cents[:, 1]) / (wy + eps)**3
    
    return force_c

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

def _simulated_annealing(cents0, max_iters=50000, T0=0.1, T_min=1e-6, cooling=0.9995, step_scale=0.01):
    """
    Simulated Annealing for the force-directed packing.
    """
    rng = np.random.default_rng(123)
    cents = cents0.copy()
    E = _energy(cents)
    best_cents = cents.copy()
    best_E = E
    best_r = _score(best_cents)
    
    T = T0
    n = len(cents)
    
    for it in range(max_iters):
        # Generate a random move: perturb one circle
        idx = rng.integers(0, n)
        delta = rng.normal(0, step_scale, size=2)
        new_cents = cents.copy()
        new_cents[idx] += delta
        
        # Keep within bounds (soft constraint)
        new_cents = np.clip(new_cents, -0.499, 0.499)
        
        new_E = _energy(new_cents)
        dE = new_E - E
        
        if dE < 0 or rng.random() < np.exp(-dE / T):
            cents = new_cents
            E = new_E
            
            if E < best_E:
                best_E = E
                best_cents = cents.copy()
                best_r = _score(best_cents)
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Periodically reset step scale based on energy
        if it % 1000 == 0:
            step_scale = 0.01 * max(0.1, 1.0 / (1.0 + E))
    
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Step 1: Hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Multi-restart Simulated Annealing
    rng = np.random.default_rng(42)
    
    inits = []
    # 1. Hexagonal
    inits.append(cents.copy())
    
    # 2. Perturbed Hexagonal
    for _ in range(3):
        pert = cents + rng.normal(0, 0.02, size=cents.shape)
        pert = np.clip(pert, -0.49, 0.49)
        inits.append(pert)
        
    # 3. Random uniform
    for _ in range(2):
        rand_cents = rng.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(rand_cents)
        
    # 4. Grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    for i, init_cents in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Adjust iterations based on remaining time
        time_left = deadline - time.time()
        # Estimate time per iteration: ~100 microseconds for N=50
        # So we can do about 10000 iterations per second
        max_iters = int(time_left * 8000)
        max_iters = max(5000, min(max_iters, 50000))
        
        try:
            cents_opt, r_opt = _simulated_annealing(
                init_cents, 
                max_iters=max_iters,
                T0=0.05,
                T_min=1e-7,
                cooling=0.9998,
                step_scale=0.005
            )
            
            # Ensure valid
            if not np.all(np.isfinite(cents_opt)):
                continue
                
            # Clip to be safe
            cents_opt = np.clip(cents_opt, -0.5, 0.5)
            
            r_opt = _score(cents_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_cents = cents_opt.copy()
                
        except Exception as e:
            continue
    
    # Final safety check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
