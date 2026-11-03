import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin for 150s limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # ------------------------------------------------------------------
    # Exact radius
    # ------------------------------------------------------------------
    def compute_radius(positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(N, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        diff = positions[:, None, :] - positions[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    # ------------------------------------------------------------------
    # Smooth surrogate with analytic gradient
    # ------------------------------------------------------------------
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def smooth_obj_grad(x_flat, beta):
        c = x_flat.reshape(N, 3)

        # Pairwise distances
        diff = c[tri_i] - c[tri_j]
        norms = np.linalg.norm(diff, axis=1)
        norms_safe = np.where(norms < 1e-12, 1e-12, norms)
        g_pairs = 0.5 * norms_safe

        # Wall distances
        # 6 walls: x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5
        # g = 0.5 - |c| is not differentiable at 0, but we can use the 6 linear constraints
        # g_1 = 0.5 - c_x, g_2 = 0.5 + c_x, etc.
        g_walls = np.concatenate([
            0.5 - c[:, 0], 0.5 + c[:, 0],
            0.5 - c[:, 1], 0.5 + c[:, 1],
            0.5 - c[:, 2], 0.5 + c[:, 2],
        ])

        g_all = np.concatenate([g_pairs, g_walls])

        # Log-sum-exp for stability
        # val = -1/beta * log(sum(exp(-beta * g_all)))
        max_g = np.max(g_all)
        log_sum_exp = np.log(np.sum(np.exp(-beta * (g_all - max_g)))) + beta * max_g
        val = -log_sum_exp / beta

        # Weights
        log_w = -beta * g_all - log_sum_exp
        weights = np.exp(log_w)
        
        # Gradient
        grad_c = np.zeros_like(c)

        # Pairwise contribution
        wp = weights[:n_pairs]
        mask = wp > 1e-14
        if np.any(mask):
            # grad_c[i] -= 0.5 * w * (c_i - c_j) / ||c_i - c_j||
            # grad_c[j] += 0.5 * w * (c_i - c_j) / ||c_i - c_j||
            factors = np.zeros(n_pairs)
            factors[mask] = 0.5 * wp[mask] / norms_safe[mask]
            contrib = factors[:, None] * diff
            np.add.at(grad_c, tri_i, -contrib)
            np.add.at(grad_c, tri_j, contrib)

        # Wall contribution
        ws = weights[n_pairs:]
        # x-dim
        grad_c[:, 0] -= ws[0:N]     # 0.5 - c_x -> -1
        grad_c[:, 0] += ws[N:2*N]   # 0.5 + c_x -> +1
        # y-dim
        grad_c[:, 1] -= ws[2*N:3*N]
        grad_c[:, 1] += ws[3*N:4*N]
        # z-dim
        grad_c[:, 2] -= ws[4*N:5*N]
        grad_c[:, 2] += ws[5*N:6*N]

        return val, (-grad_c).flatten()

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    bounds = [(-0.499999, 0.499999)] * (N * 3)

    def local_smooth(c_init, beta, maxiter=200):
        if remaining() < 5:
            return c_init
        x0 = c_init.flatten()

        def f(x):
            v, _ = smooth_obj_grad(x, beta)
            return v

        def g(x):
            _, gr = smooth_obj_grad(x, beta)
            return gr

        try:
            res = opt.minimize(f, x0, jac=g, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def local_direct(c_init, maxiter=300):
        if remaining() < 5:
            return c_init
        x0 = c_init.flatten()

        def f(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.minimize(f, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def make_bcc():
        # Generate BCC lattice points
        # Simple cubic grid with spacing 1, plus body center
        # We want to fit N=50 points.
        # A 4x4x4 grid has 64 points. BCC adds 27 points (3x3x3). Total 91.
        # We can generate a larger grid and pick the N closest to the center?
        # Or just generate a BCC structure and scale it.
        
        pts = []
        # Cubic corners
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    pts.append([x, y, z])
        # Body centers
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    pts.append([x + 0.5, y + 0.5, z + 0.5])
        
        arr = np.array(pts)
        # Center it
        arr -= np.array([2.0, 2.0, 2.0])
        
        # Scale to fit in cube [-0.5, 0.5]
        # The max coordinate is 2.5 (from 4.5 - 2.0). 
        # To fit in 0.5, scale by 0.5/2.5 = 0.2
        # But we want to fill the space. 
        # Let's just scale so max abs is 0.45 to be safe.
        mx = np.max(np.abs(arr))
        if mx > 0:
            arr *= 0.45 / mx
            
        # We have 125 + 64 = 189 points. We need 50.
        # Sort by distance to center? No, that clusters them.
        # Better: Randomly select 50? Or take a subset?
        # Taking the first 50 from the sorted list of coordinates might be biased.
        # Let's just take a random subset of the BCC points, ensuring diversity.
        # Actually, for packing, we want them spread out.
        # The BCC points are already spread.
        # Let's just pick 50 points that are roughly uniform.
        # A simple way: sort by x, y, z?
        # Let's just use a random seed to pick 50 from the 189.
        idx = np.random.RandomState(42).choice(len(arr), N, replace=False)
        return arr[idx]

    def make_grid():
        # Simple grid
        # 50 points. 4x4x4 = 64.
        # Take first 50.
        m = 4
        t = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in t for y in t for x in t])
        return grid[:N].copy()

    def make_random(seed):
        return np.random.RandomState(seed).uniform(-0.45, 0.45, size=(N, 3))

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    best_pos = None
    best_r = -1.0

    def update(pos):
        nonlocal best_pos, best_r
        r = compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        return r

    # Generate a diverse set of initializations
    inits = [make_bcc(), make_grid()]
    for seed in range(10):
        inits.append(make_random(seed))

    for init in inits:
        if remaining() < 15:
            break
        pos = np.clip(init, -0.499999, 0.499999)

        # Beta-annealed smooth optimisation
        if remaining() > 25:
            # Start with low beta (smooth) to find global structure, then high beta (sharp) for local
            for beta in [1.0, 5.0, 20.0, 100.0, 500.0]:
                if remaining() < 6:
                    break
                pos = local_smooth(pos, beta, maxiter=200)

        # Direct L-BFGS-B polish
        if remaining() > 10:
            pos = local_direct(pos, maxiter=350)

        update(pos)

    # --- Perturbation refinement around best ---
    if best_pos is not None and remaining() > 18:
        for k in range(10):
            if remaining() < 5:
                break
            rng = np.random.RandomState(2000 + k)
            scale = 0.01 * (1.0 - k / 11.0)
            pert = np.clip(best_pos + rng.normal(0, scale, size=best_pos.shape),
                           -0.499999, 0.499999)

            if remaining() > 12:
                for beta in [50.0, 200.0]:
                    if remaining() < 4:
                        break
                    pert = local_smooth(pert, beta, maxiter=150)

            if remaining() > 6:
                pert = local_direct(pert, maxiter=250)

            update(pert)

    # --- Final high-precision polish ---
    if best_pos is not None and remaining() > 10:
        try:
            fp = local_direct(best_pos, maxiter=500)
            update(fp)
        except Exception:
            pass

    # --- Fallback ---
    if best_pos is None:
        best_pos = np.clip(make_bcc(), -0.499999, 0.499999)

    return best_pos
