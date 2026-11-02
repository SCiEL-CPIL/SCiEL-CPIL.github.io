"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 130.0  # seconds; leave margin under 150s wall-clock


def _score(c):
    """Return the feasible radius r for centres c (N,2) in [-0.5,0.5]^2."""
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    # distance to walls
    wall = 0.5 - np.max(np.abs(c), axis=1)
    r_wall = np.min(wall)
    if r_wall <= 0:
        return 0.0
    # pairwise distances
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    return min(r_wall, r_pair)


def _grid_init(m=7):
    """7x7 grid, drop 2 corners to get 50 points."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    # Drop two corner points (the ones farthest from centre? or just last two)
    # Better: drop the two corners that are least critical. Actually for 7x7=49 we need 50.
    # Wait, 7*7=49. We need 50. So 8x8 grid? 8*8=64, drop 14.
    # Or 7x7 + 1?
    # Let's use a better initial: 8x8 grid, keep best 50?
    # Actually, the known good packing is roughly 7x7 with modifications.
    # Let's start with a dense grid and let the optimizer work.
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    # Select 50 points: keep the ones closest to the center? Or just first 50?
    # First 50 of 64 is fine as a start.
    return pts[:N].copy()


def _refine_slsqp(c0, max_iter=2000, ftol=1e-12):
    """Maximize r subject to constraints using SLSQP."""
    from scipy.optimize import minimize

    n = c0.shape[0]
    x0 = np.concatenate([c0.flatten(), [0.05]])  # last var is r

    def objective(x):
        # Minimize -r
        return -x[-1]

    def constraints(x):
        c = x[:-1].reshape(n, 2)
        r = x[-1]
        cons = []
        # Wall constraints: 0.5 - |c_i| >= r
        for i in range(n):
            for k in range(2):
                # 0.5 - abs(c_ik) - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k: 0.5 - abs(x[2*i+k]) - x[-1]})
        # Pairwise constraints: |c_i - c_j| - 2r >= 0
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - 2*x[-1]})
        return cons

    # This is too slow with 1225+50 constraints as separate lambdas.
    # SLSQP can handle vectorized constraints.
    # Let's define a single constraint function that returns an array.
    # But scipy.optimize.minimize with method='SLSQP' expects a list of dicts or a single dict with 'fun' returning array for multiple constraints.
    
    # Let's rewrite for efficiency.
    def cons_fun(x):
        c = x[:-1].reshape(n, 2)
        r = x[-1]
        vals = []
        # Walls
        for i in range(n):
            vals.append(0.5 - abs(c[i, 0]) - r)
            vals.append(0.5 - abs(c[i, 1]) - r)
        # Pairs
        for i in range(n):
            for j in range(i+1, n):
                vals.append(np.linalg.norm(c[i] - c[j]) - 2*r)
        return np.array(vals)

    # SLSQP requires each constraint to be a dict. For many constraints, it's better to use a single dict with 'fun' returning array?
    # Actually, scipy SLSQP accepts a list of constraint dicts. If we pass a single dict with 'fun' returning an array, it treats it as multiple constraints?
    # No, it expects 'fun' to return a scalar for each dict, OR if 'fun' returns an array, it's treated as multiple inequalities?
    # Documentation: "fun : callable, fun(x, *args) -> ndarray. The function to be optimized. ... constraints : sequence of Constraint or dict."
    # If I use a list of dicts, it's slow.
    # Alternative: use penalty method or trust-constr?
    # Let's stick to a simpler local optimization: gradient ascent on r with projection.
    return None


def _gradient_ascent(c, steps=5000, lr=0.001):
    """Simple gradient ascent on r with projection onto feasible set."""
    n = c.shape[0]
    r = _score(c)
    
    # Precompute pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
            
        # Compute current r
        wall = 0.5 - np.max(np.abs(c), axis=1)
        r_wall = np.min(wall)
        
        d = np.zeros((n, n))
        for i, j in pairs:
            d[i, j] = d[j, i] = np.linalg.norm(c[i] - c[j])
        r_pair = np.min(d[i, j] for i, j in pairs) / 2.0
        
        r = min(r_wall, r_pair)
        
        # Gradient of r wrt c
        # This is complex. Let's use a different approach:
        # Push apart points that are too close, pull in from walls if too close.
        
        # 1. Push apart pairs with dist < 2*r + eps
        eps = 1e-6
        for i, j in pairs:
            dist = np.linalg.norm(c[i] - c[j])
            if dist < 2*r + eps:
                diff = c[i] - c[j]
                if dist > 1e-10:
                    diff /= dist
                else:
                    diff = np.array([1.0, 0.0])
                c[i] += diff * (2*r + eps - dist) * 0.5
                c[j] -= diff * (2*r + eps - dist) * 0.5
        
        # 2. Push in from walls
        for i in range(n):
            for k in range(2):
                if c[i, k] > 0.5 - r:
                    c[i, k] = 0.5 - r
                elif c[i, k] < -0.5 + r:
                    c[i, k] = -0.5 + r
        
        # 3. Try to increase r slightly
        # This is a heuristic, not a true gradient ascent.
        
    return c


def construct():
    start_time = time.time()
    
    # Initial guess
    c = _grid_init()
    best_c = c.copy()
    best_r = _score(c)
    
    # Use a more sophisticated local optimization
    # Let's try a simple iterative relaxation:
    # 1. Compute current r.
    # 2. For each pair, if dist < 2r, push apart.
    # 3. For each point, if wall dist < r, push in.
    # 4. Repeat.
    
    n = c.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute pair indices for vectorization if possible, but N=50 is small.
    
    for iteration in range(10000):
        if time.time() > DEADLINE - 5:
            break
            
        # Compute r
        wall = 0.5 - np.max(np.abs(c), axis=1)
        r_wall = np.min(wall)
        
        min_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_dist:
                min_dist = d
        r_pair = min_dist / 2.0
        r = min(r_wall, r_pair)
        
        # Update best
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Relaxation step
        # Push apart pairs that are violating or close to violating
        target_dist = 2 * r + 1e-8
        for i, j in pairs:
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < target_dist:
                if dist > 1e-10:
                    diff /= dist
                else:
                    diff = np.array([1.0, 0.0])
                push = (target_dist - dist) * 0.5
                c[i] += diff * push
                c[j] -= diff * push
        
        # Push in from walls
        for i in range(n):
            for k in range(2):
                if c[i, k] > 0.5 - r:
                    c[i, k] = 0.5 - r
                elif c[i, k] < -0.5 + r:
                    c[i, k] = -0.5 + r
        
        # Small random perturbation to escape local minima?
        # If we are stuck, add noise.
        if iteration % 100 == 0 and iteration > 0:
            # Check if we've improved in last 100 steps
            pass # Simplified: always add small noise
            c += np.random.normal(0, 1e-5, c.shape)
            # Re-project to ensure validity
            for i in range(n):
                for k in range(2):
                    c[i, k] = np.clip(c[i, k], -0.5 + 1e-9, 0.5 - 1e-9)
    
    # Final polish: try to increase r by moving points
    # This is a basic approach. Let's try to use scipy.optimize if available and fast enough.
    # But the simple relaxation might be sufficient for a jump from 1.0.
    
    # Ensure final validity
    for i in range(n):
        for k in range(2):
            best_c[i, k] = np.clip(best_c[i, k], -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Check final score
    final_r = _score(best_c)
    if final_r < best_r:
        # If clipping reduced score, use previous best
        pass
    else:
        best_r = final_r
    
    return best_c
