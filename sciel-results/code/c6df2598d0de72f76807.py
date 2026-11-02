"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9

def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr

def _valid_check(pos, r, tol=1e-7):
    """Check validity: in square, no overlaps."""
    n = len(r)
    for i in range(n):
        if r[i] < -tol:
            return False
        if pos[i, 0] < r[i] - tol or pos[i, 0] > 1.0 - r[i] + tol:
            return False
        if pos[i, 1] < r[i] - tol or pos[i, 1] > 1.0 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _delaunay_edges(pos):
    """Build Delaunay triangulation of centers, return edge list (i, j) with i < j."""
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _build_constraints(n, edges):
    """Build constraint functions for SLSQP."""
    cons = []
    for i in range(n):
        xi = 2 * i
        yi = 2 * i + 1
        ri = 2 * n + i
        # x_i - r_i >= 0
        def c1(p, xi=xi, ri=ri):
            return p[xi] - p[ri]
        def j1(p, xi=xi, ri=ri):
            v = np.zeros(3 * n)
            v[xi] = 1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c1, 'jac': j1})
        # 1 - x_i - r_i >= 0
        def c2(p, xi=xi, ri=ri):
            return 1.0 - p[xi] - p[ri]
        def j2(p, xi=xi, ri=ri):
            v = np.zeros(3 * n)
            v[xi] = -1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c2, 'jac': j2})
        # y_i - r_i >= 0
        def c3(p, yi=yi, ri=ri):
            return p[yi] - p[ri]
        def j3(p, yi=yi, ri=ri):
            v = np.zeros(3 * n)
            v[yi] = 1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c3, 'jac': j3})
        # 1 - y_i - r_i >= 0
        def c4(p, yi=yi, ri=ri):
            return 1.0 - p[yi] - p[ri]
        def j4(p, yi=yi, ri=ri):
            v = np.zeros(3 * n)
            v[yi] = -1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c4, 'jac': j4})
        # r_i >= 0
        def c5(p, ri=ri):
            return p[ri]
        def j5(p, ri=ri):
            v = np.zeros(3 * n)
            v[ri] = 1.0
            return v
        cons.append({'type': 'ineq', 'fun': c5, 'jac': j5})

    for (i, j) in edges:
        xi, yi = 2 * i, 2 * i + 1
        xj, yj = 2 * j, 2 * j + 1
        ri, rj = 2 * n + i, 2 * n + j

        def c_overlap(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]
            dy = p[yi] - p[yj]
            d = np.hypot(dx, dy)
            return d - p[ri] - p[rj]

        def j_overlap(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]
            dy = p[yi] - p[yj]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
            v = np.zeros(3 * n)
            v[xi] = dx / d
            v[yi] = dy / d
            v[xj] = -dx / d
            v[yj] = -dy / d
            v[ri] = -1.0
            v[rj] = -1.0
            return v

        cons.append({'type': 'ineq', 'fun': c_overlap, 'jac': j_overlap})

    return cons

def _slsqp_step(pos, r, edges):
    """One SLSQP step."""
    n = len(r)
    x0 = np.concatenate([pos.flatten(), r])

    def objective(p):
        return -np.sum(p[2 * n:])

    cons = _build_constraints(n, edges)
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 100, 'ftol': 1e-10})
        if res.success or res.fun < objective(x0):
            new_pos = res.x[:2 * n].reshape(n, 2)
            new_r = res.x[2 * n:]
            if _valid_check(new_pos, new_r):
                return new_pos, new_r
    except Exception:
        pass
    return pos, r

def _grow_and_relax(pos, r, edges, t_start, deadline):
    """Alternating growth and relaxation."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)

    for iteration in range(50):
        if time.time() > t_start + deadline:
            break

        # Grow radii (vectorized)
        for i in range(n):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)

        # SLSQP relaxation
        new_pos, new_r = _slsqp_step(pos, r, edges)
        pos = new_pos
        r = new_r
        edges = _delaunay_edges(pos)

        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()

    return best_pos, best_r

def _polish(pos, r, t_start, deadline):
    """Final polish with more iterations."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)

    for _ in range(12):
        if time.time() > t_start + deadline:
            break
        edges = _delaunay_edges(pos)
        new_pos, new_r = _slsqp_step(pos, r, edges)
        if np.sum(new_r) > best_score:
            best_score = np.sum(new_r)
            best_pos = new_pos.copy()
            best_r = new_r.copy()
        pos = best_pos
        r = best_r

    return best_pos, best_r

def _soft_penalty_objective(pos, r, penalty_w=100.0):
    """Soft penalty objective for simulated annealing."""
    n = len(r)
    score = np.sum(r)
    penalty = 0.0
    # Wall violations
    for i in range(n):
        if pos[i, 0] < r[i]:
            penalty += (r[i] - pos[i, 0]) ** 2
        if pos[i, 0] > 1.0 - r[i]:
            penalty += (pos[i, 0] - (1.0 - r[i])) ** 2
        if pos[i, 1] < r[i]:
            penalty += (r[i] - pos[i, 1]) ** 2
        if pos[i, 1] > 1.0 - r[i]:
            penalty += (pos[i, 1] - (1.0 - r[i])) ** 2
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            overlap = r[i] + r[j] - d
            if overlap > 0:
                penalty += overlap ** 2
    return -score + penalty_w * penalty

def _anneal(pos, r, t_start, deadline):
    """Simulated annealing search."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    T = 0.05
    T_min = 1e-6
    cooling = 0.998
    rng = np.random.default_rng(42)
    
    current_obj = _soft_penalty_objective(pos, r)
    
    while time.time() < t_start + deadline and T > T_min:
        # Propose move
        if rng.random() < 0.5:
            # Move a position
            i = rng.integers(0, n)
            pos[i, 0] += rng.normal(0, T * 0.5)
            pos[i, 1] += rng.normal(0, T * 0.5)
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
        else:
            # Change a radius
            i = rng.integers(0, n)
            r[i] = np.clip(r[i] + rng.normal(0, T * 0.3), 0.0, 0.4)
        
        new_obj = _soft_penalty_objective(pos, r)
        delta = new_obj - current_obj
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_obj = new_obj
            score = np.sum(r)
            if _valid_check(pos, r) and score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_r = r.copy()
        
        T *= cooling
    
    return best_pos, best_r

def _error_repair(pos, r, t_start, deadline):
    """Repair invalid configurations by shrinking overlapping circles."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    for _ in range(20):
        if time.time() > t_start + deadline:
            break
        
        if _valid_check(pos, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_r = r.copy()
            break
        
        # Find worst overlap and repair
        worst_i, worst_j, worst_overlap = -1, -1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                overlap = r[i] + r[j] - d
                if overlap > worst_overlap:
                    worst_overlap = overlap
                    worst_i, worst_j = i, j
        
        if worst_i >= 0:
            # Shrink the smaller circle
            if r[worst_i] < r[worst_j]:
                r[worst_i] = max(0.0, r[worst_i] - worst_overlap * 0.6)
            else:
                r[worst_j] = max(0.0, r[worst_j] - worst_overlap * 0.6)
        else:
            # Wall violation: move toward center
            for i in range(n):
                if pos[i, 0] < r[i]:
                    pos[i, 0] = (pos[i, 0] + r[i]) / 2
                elif pos[i, 0] > 1.0 - r[i]:
                    pos[i, 0] = (pos[i, 0] + (1.0 - r[i])) / 2
                if pos[i, 1] < r[i]:
                    pos[i, 1] = (pos[i, 1] + r[i]) / 2
                elif pos[i, 1] > 1.0 - r[i]:
                    pos[i, 1] = (pos[i, 1] + (1.0 - r[i])) / 2
    
    return best_pos, best_r

def _vectorized_penalty(pos, r, beta=50.0):
    """Vectorized penalty objective for faster SA."""
    n = len(r)
    score = np.sum(r)
    
    # Wall violations (vectorized)
    wall_pen = 0.0
    for i in range(n):
        for coord in [0, 1]:
            if pos[i, coord] < r[i]:
                wall_pen += (r[i] - pos[i, coord]) ** 2
            if pos[i, coord] > 1.0 - r[i]:
                wall_pen += (pos[i, coord] - (1.0 - r[i])) ** 2
    
    # Overlap violations (vectorized using broadcasting)
    dx = pos[:, 0:1] - pos[:, 0:1].T
    dy = pos[:, 1:2] - pos[:, 1:2].T
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    overlap_matrix = r[:, None] + r[None, :] - dist
    overlap_pen = np.sum(np.maximum(0, overlap_matrix) ** 2)
    
    return -score + beta * (wall_pen + overlap_pen)

def _fast_anneal(pos, r, t_start, deadline):
    """Faster simulated annealing with vectorized penalty."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    T = 0.03
    T_min = 1e-6
    cooling = 0.9995
    rng = np.random.default_rng(12345)
    
    current_obj = _vectorized_penalty(pos, r)
    
    while time.time() < t_start + deadline and T > T_min:
        # Propose move
        if rng.random() < 0.6:
            # Move a position
            i = rng.integers(0, n)
            pos[i, 0] += rng.normal(0, T * 0.4)
            pos[i, 1] += rng.normal(0, T * 0.4)
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
        else:
            # Change a radius
            i = rng.integers(0, n)
            r[i] = np.clip(r[i] + rng.normal(0, T * 0.25), 0.0, 0.4)
        
        new_obj = _vectorized_penalty(pos, r)
        delta = new_obj - current_obj
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_obj = new_obj
            if _valid_check(pos, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_r = r.copy()
        
        T *= cooling
    
    return best_pos, best_r

def construct():
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    best_score = -1.0
    best_result = None

    num_restarts = 5

    for restart in range(num_restarts):
        if time.time() > t_start + DEADLINE - 20.0:
            break

        rng = np.random.default_rng(2000 + restart)

        if restart == 0:
            # 5x5 grid with one removed, use 25 positions
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side
                pos[i, 1] = (row + 0.5) / side
                r[i] = 0.36 / side
        elif restart == 1:
            # Perturbed grid
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side + rng.normal(0, 0.02)
                pos[i, 1] = (row + 0.5) / side + rng.normal(0, 0.02)
                r[i] = 0.33 / side
            pos = np.clip(pos, 0.05, 0.95)
        elif restart == 2:
            # Hexagonal packing: 5 rows, alternating 5-6-5-6-5 = 27, use 26
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            row_counts = [5, 6, 5, 6, 4]
            idx = 0
            for row_idx, count in enumerate(row_counts):
                if idx >= N:
                    break
                row_y = (row_idx + 0.5) / 5.0
                for col_idx in range(count):
                    if idx >= N:
                        break
                    offset = 0.5 / 6.0 if row_idx % 2 == 1 else 0.0
                    pos[idx, 0] = (col_idx + 0.5 + offset) / 6.0
                    pos[idx, 1] = row_y
                    r[idx] = 0.35 / 6.0
                    idx += 1
        elif restart == 3:
            # Random with larger initial radii
            pos = rng.random((N, 2)) * 0.8 + 0.1
            r = np.full(N, 0.02)
        else:
            # Another perturbed grid variant
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                # Add some structured perturbation
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, 0.03)
                pos[i, 0] = (col + 0.5) / side + dist * np.cos(angle)
                pos[i, 1] = (row + 0.5) / side + dist * np.sin(angle)
                r[i] = 0.34 / side
            pos = np.clip(pos, 0.04, 0.96)

        if not _valid_check(pos, r):
            scale = 0.5
            while not _valid_check(pos, r * scale) and scale > 0.005:
                scale *= 0.7
            r = r * scale

        edges = _delaunay_edges(pos)

        # Phase 1: Grow and relax
        pos, r = _grow_and_relax(pos, r, edges, t_start, DEADLINE - 25.0)

        # Phase 2: Error repair
        pos, r = _error_repair(pos, r, t_start, DEADLINE - 15.0)

        # Phase 3: Polish with more iterations
        pos, r = _polish(pos, r, t_start, DEADLINE - 8.0)

        # Phase 4: Fast SA to escape local minima
        if time.time() < t_start + DEADLINE - 5.0:
            pos, r = _fast_anneal(pos, r, t_start, DEADLINE - 3.0)
            pos, r = _polish(pos, r, t_start, DEADLINE - 2.0)

        score = np.sum(r)
        if _valid_check(pos, r) and score > best_score:
            best_score = score
            best_result = np.column_stack([pos, r])

        if best_score > 2.95:
            break

    if best_result is None:
        side = 5
        pos = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            r[i] = 0.36 / side
        best_result = np.column_stack([pos, r])

    return best_result

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
