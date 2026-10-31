"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 138.0


def _validity(circles, W, H):
    """Check disjoint interiors and bounds. Returns (ok, perimeter, sum_radii)."""
    n = len(circles)
    if n != N:
        return False, 0.0, 0.0
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per > PERIMETER + 1e-6:
        return False, per, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
            if d < circles[i][2] + circles[j][2] - 1e-7:
                return False, per, 0.0
    s = sum(c[2] for c in circles)
    return True, per, s


def _scale_to_perimeter(circles, target_per=PERIMETER):
    """Scale the whole packing so bounding-box perimeter equals target_per."""
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per <= 0:
        return circles
    s = target_per / per
    out = []
    for (x, y, r) in circles:
        nx = (x - minx) * s
        ny = (y - miny) * s
        nr = r * s
        out.append((nx, ny, nr))
    return out


def _delaunay_edges(circles):
    """
    Compute Delaunay triangulation edges from circle centers.
    Returns list of (i, j) pairs.
    """
    try:
        from scipy.spatial import Delaunay
        centers = np.array([(c[0], c[1]) for c in circles])
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                i = simplex[a]
                j = simplex[(a + 1) % 3]
                if i > j:
                    i, j = j, i
                edges.add((i, j))
        return list(edges)
    except Exception:
        # Fallback: use k-nearest neighbors
        from scipy.spatial.distance import cdist
        centers = np.array([(c[0], c[1]) for c in circles])
        dists = cdist(centers, centers)
        edges = set()
        k = 6
        for i in range(N):
            idx = np.argsort(dists[i])[:k]
            for j in idx:
                if i != j:
                    if i > j:
                        i, j = j, i
                    edges.add((i, j))
        return list(edges)


def _delaunay_constrained_optimize(circles, W, H, max_iter=200):
    """
    Optimize using Delaunay graph constraints.
    For each edge (i,j) in the Delaunay graph, enforce d_ij >= r_i + r_j.
    This creates a sparse constraint structure that can be exploited.
    """
    circles = [list(c) for c in circles]
    edges = _delaunay_edges(circles)
    
    # Build adjacency list
    adj = {i: [] for i in range(N)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Use SLSQP with sparse constraints
    def objective(x):
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:3*N]
        
        # Objective: maximize sum of radii (minimize negative)
        val = -np.sum(r)
        
        # Penalty for wall violations
        mu = 50.0
        for i in range(N):
            if cx[i] - r[i] < 0:
                val += mu * (cx[i] - r[i]) ** 2
            if cy[i] - r[i] < 0:
                val += mu * (cy[i] - r[i]) ** 2
            if cx[i] + r[i] > W:
                val += mu * (cx[i] + r[i] - W) ** 2
            if cy[i] + r[i] > H:
                val += mu * (cy[i] + r[i] - H) ** 2
        
        # Penalty for Delaunay edge violations
        for i, j in edges:
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if d < min_d:
                val += mu * (min_d - d) ** 2
        
        return val
    
    def objective_grad(x):
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:3*N]
        
        grad = np.zeros(3 * N)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] = -1.0
        
        mu = 50.0
        
        # Wall constraint gradients
        for i in range(N):
            if cx[i] - r[i] < 0:
                viol = cx[i] - r[i]
                grad[i] += 2 * mu * viol
                grad[2*N + i] -= 2 * mu * viol
            if cy[i] - r[i] < 0:
                viol = cy[i] - r[i]
                grad[N + i] += 2 * mu * viol
                grad[2*N + i] -= 2 * mu * viol
            if cx[i] + r[i] > W:
                viol = cx[i] + r[i] - W
                grad[i] += 2 * mu * viol
                grad[2*N + i] += 2 * mu * viol
            if cy[i] + r[i] > H:
                viol = cy[i] + r[i] - H
                grad[N + i] += 2 * mu * viol
                grad[2*N + i] += 2 * mu * viol
        
        # Delaunay edge constraint gradients
        for i, j in edges:
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if d < min_d and d > 1e-12:
                viol = min_d - d
                # d/dx_i of (min_d - d)^2
                grad[i] += 2 * mu * viol * (dx / d)
                grad[N + i] += 2 * mu * viol * (dy / d)
                grad[j] -= 2 * mu * viol * (dx / d)
                grad[N + j] -= 2 * mu * viol * (dy / d)
                grad[2*N + i] += 2 * mu * viol
                grad[2*N + j] += 2 * mu * viol
        
        return grad
    
    x0 = np.concatenate([
        np.array([c[0] for c in circles]),
        np.array([c[1] for c in circles]),
        np.array([c[2] for c in circles])
    ])
    
    bounds = []
    for i in range(N):
        bounds.append((0, W))
    for i in range(N):
        bounds.append((0, H))
    for i in range(N):
        bounds.append((1e-4, min(W, H) / 2))
    
    try:
        res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10})
        
        cx_opt = res.x[:N]
        cy_opt = res.x[N:2*N]
        r_opt = res.x[2*N:3*N]
        
        circles_opt = list(zip(cx_opt, cy_opt, r_opt))
        ok, per, s = _validity(circles_opt, W, H)
        if ok:
            return circles_opt, s
    except Exception:
        pass
    
    return None, -float('inf')


def _force_directed_pack(W, H, seed=0):
    """
    Pack N circles in a W x H rectangle using force-directed relaxation.
    Returns list of (x, y, r) or None if failed.
    """
    rng = random.Random(seed)
    
    # Initial configuration: Hexagonal lattice
    points = []
    r_est = np.sqrt(W * H / N) * 0.8
    r = r_est
    
    y = r
    row = 0
    while y < H - r:
        x = r
        if row % 2 == 1:
            x += r
        while x < W - r:
            points.append((x, y, r))
            x += 2 * r
        y += r * np.sqrt(3)
        row += 1
        
    if len(points) > N:
        center_x, center_y = W/2, H/2
        points.sort(key=lambda p: (p[0]-center_x)**2 + (p[1]-center_y)**2)
        points = points[:N]
    elif len(points) < N:
        while len(points) < N:
            points.append((rng.uniform(r, W-r), rng.uniform(r, H-r), r))
            
    circles = [list(p) for p in points]
    
    dt = 0.05
    max_iter = 500
    repulsion_strength = 5.0
    boundary_strength = 10.0
    growth_rate = 0.01
    
    for step in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        forces = [[0.0, 0.0] for _ in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                d = np.hypot(dx, dy)
                min_d = circles[i][2] + circles[j][2]
                
                if d < 1e-6:
                    dx = 1e-3
                    dy = 1e-3
                    d = np.hypot(dx, dy)
                    
                if d < min_d:
                    overlap = min_d - d
                    f = repulsion_strength * overlap
                    fx = f * dx / d
                    fy = f * dy / d
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    
        for i in range(N):
            x, y, r = circles[i]
            if x < r:
                forces[i][0] += boundary_strength * (r - x)
            if x > W - r:
                forces[i][0] -= boundary_strength * (x - (W - r))
            if y < r:
                forces[i][1] += boundary_strength * (r - y)
            if y > H - r:
                forces[i][1] -= boundary_strength * (y - (H - r))
                
        for i in range(N):
            circles[i][0] += forces[i][0] * dt
            circles[i][1] += forces[i][1] * dt
            circles[i][0] = max(circles[i][2], min(W - circles[i][2], circles[i][0]))
            circles[i][1] = max(circles[i][2], min(H - circles[i][2], circles[i][1]))
            
        total_force_mag = sum(np.hypot(f[0], f[1]) for f in forces)
        if total_force_mag < 1.0:
             for i in range(N):
                 circles[i][2] += growth_rate
             
    # Final relaxation
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                d = np.hypot(dx, dy)
                min_d = circles[i][2] + circles[j][2]
                if d < min_d - 1e-8:
                    if d < 1e-8:
                        dx = 1.0
                        dy = 0.0
                        d = 1.0
                    push = (min_d - d) / 2.0 + 1e-4
                    nx_i = circles[i][0] + (dx/d)*push
                    ny_i = circles[i][1] + (dy/d)*push
                    nx_j = circles[j][0] - (dx/d)*push
                    ny_j = circles[j][1] - (dy/d)*push
                    
                    if nx_i >= circles[i][2] and nx_i <= W-circles[i][2] and ny_i >= circles[i][2] and ny_i <= H-circles[i][2]:
                        circles[i][0] = nx_i
                        circles[i][1] = ny_i
                    if nx_j >= circles[j][2] and nx_j <= W-circles[j][2] and ny_j >= circles[j][2] and ny_j <= H-circles[j][2]:
                        circles[j][0] = nx_j
                        circles[j][1] = ny_j
                    changed = True
        if not changed:
            break
            
    return circles


def _refine_positions(circles, W, H, steps=1000, lr0=0.02):
    """Local search: nudge each circle to increase its radius while keeping constraints."""
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(12345)
    
    for step in range(steps):
        if time.time() > DEADLINE - 5:
            break
        lr = lr0 * (1.0 - step / steps)
        i = rng.randrange(n)
        x, y, r = circles[i]
        
        for _ in range(3):
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            norm = np.hypot(dx, dy)
            if norm < 1e-9:
                continue
            dx, dy = dx / norm, dy / norm
            nx = x + dx * lr
            ny = y + dy * lr
            nx = min(max(nx, 0.0), W)
            ny = min(max(ny, 0.0), H)
            
            nr = min(nx, W - nx, ny, H - ny)
            for j in range(n):
                if j == i:
                    continue
                px, py, pr = circles[j]
                d = np.hypot(nx - px, ny - py)
                cand = d - pr
                if cand < nr:
                    nr = cand
            nr = max(nr, 0.0)
            
            if nr > r + 1e-12:
                circles[i] = [nx, ny, nr]
                break
    
    return circles


def _differential_evolution_pack(W, H, pop_size=40, max_iter=100):
    """
    Differential evolution to explore the configuration space.
    Each individual is (cx_1..cx_N, cy_1..cy_N, r_1..r_N).
    """
    rng = np.random.RandomState(42)
    
    def fitness(ind):
        cx = ind[:N]
        cy = ind[N:2*N]
        r = ind[2*N:3*N]
        
        # Check validity
        circles = list(zip(cx, cy, r))
        ok, per, s = _validity(circles, W, H)
        if not ok:
            return -1e6
        
        # Penalize if perimeter is too small (we want to use the full rectangle)
        penalty = 0.0
        if per < PERIMETER * 0.9:
            penalty = (PERIMETER * 0.9 - per) * 10
        
        return s - penalty
    
    # Initialize population
    pop = []
    for _ in range(pop_size):
        cx = rng.uniform(0, W, N)
        cy = rng.uniform(0, H, N)
        r = rng.uniform(0.05, min(W, H) / 4, N)
        ind = np.concatenate([cx, cy, r])
        pop.append(ind)
    
    best_ind = None
    best_fit = -float('inf')
    
    for gen in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        
        for i in range(pop_size):
            if rng.random() < 0.8:
                a, b, c = rng.choice(pop_size, 3, replace=False)
                F = rng.uniform(0.5, 1.5)
                trial = pop[a] + F * (pop[b] - pop[c])
                # Clip to bounds
                trial[:N] = np.clip(trial[:N], 0, W)
                trial[N:2*N] = np.clip(trial[N:2*N], 0, H)
                trial[2*N:3*N] = np.clip(trial[2*N:3*N], 0.01, min(W, H) / 2)
            else:
                trial = pop[i].copy()
                # Random mutation
                idx = rng.randint(0, 2*N-1)
                if idx < N:
                    trial[idx] = rng.uniform(0, W)
                elif idx < 2*N:
                    trial[idx] = rng.uniform(0, H)
                else:
                    trial[idx] = rng.uniform(0.01, min(W, H) / 2)
            
            trial_fit = fitness(trial)
            if trial_fit > best_fit:
                best_fit = trial_fit
                best_ind = trial.copy()
            
            if trial_fit >= fitness(pop[i]):
                pop[i] = trial
    
    if best_ind is not None:
        cx = best_ind[:N]
        cy = best_ind[N:2*N]
        r = best_ind[2*N:3*N]
        circles = list(zip(cx, cy, r))
        ok, per, s = _validity(circles, W, H)
        if ok:
            return circles, s
    
    return None, -float('inf')


def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 138.0
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    aspect_choices = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.15, 0.85]
    random.shuffle(aspect_choices)
    
    for W in aspect_choices:
        if time.time() > DEADLINE - 15:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Strategy 1: Force-directed packing
        circles = _force_directed_pack(W, H, seed=int(W * 100))
        if circles is not None:
            circles_scaled = _scale_to_perimeter(circles, PERIMETER)
            ok, per, s = _validity(circles_scaled, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = circles_scaled
        
        # Strategy 2: Delaunay-constrained optimization
        if time.time() < DEADLINE - 20:
            # Start from force-directed result
            if circles is not None:
                circles_delaunay, s_delaunay = _delaunay_constrained_optimize(circles, W, H, max_iter=150)
                if circles_delaunay is not None and s_delaunay > best_val:
                    circles_delaunay = _scale_to_perimeter(circles_delaunay, PERIMETER)
                    ok, per, s = _validity(circles_delaunay, 2.0, 2.0)
                    if ok and s > best_val:
                        best_val = s
                        best = circles_delaunay
        
        # Strategy 3: Differential evolution
        if time.time() < DEADLINE - 25:
            circles_de, s_de = _differential_evolution_pack(W, H, pop_size=30, max_iter=50)
            if circles_de is not None and s_de > best_val:
                circles_de = _scale_to_perimeter(circles_de, PERIMETER)
                ok, per, s = _validity(circles_de, 2.0, 2.0)
                if ok and s > best_val:
                    best_val = s
                    best = circles_de
        
        # Refine the best so far
        if best is not None and time.time() < DEADLINE - 5:
            best_refined = _refine_positions(best, 2.0, 2.0, steps=500, lr0=0.01)
            best_refined = _scale_to_perimeter(best_refined, PERIMETER)
            ok, per, s = _validity(best_refined, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = best_refined
    
    if best is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        best = _scale_to_perimeter(best, PERIMETER)
    
    return np.array(best)


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
