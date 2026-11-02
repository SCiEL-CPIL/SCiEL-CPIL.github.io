"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Physics-based Position Based Dynamics (PBD) simulation with
outward pressure, Gauss-Seidel constraint projection, and Delaunay contact graph.
"""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 26
DEADLINE = 143.0  # safety margin below 150s kill


def _max_radius(pos, rad):
    """Maximal radius for each circle given current positions and radii."""
    n = len(rad)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - rad[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    for i in range(len(arr)):
        x, y, r = arr[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _delaunay_edges(pos):
    """Return set of (i,j) pairs that are Delaunay neighbors."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return edges
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _pbd_simulate(pos, rad, time_budget=10.0):
    """
    PBD simulation: grow radii via outward pressure, then project constraints.
    Returns improved (pos, rad).
    """
    t0 = time.time()
    pos = pos.copy()
    rad = rad.copy()
    
    # Adaptive parameters
    pressure = 0.002  # initial growth rate per iteration
    n_inner = 5  # Gauss-Seidel passes per outer iteration
    
    for outer in range(2000):
        if time.time() - t0 > time_budget:
            break
        
        # --- Growth phase: increase radii (outward pressure) ---
        rad = np.minimum(rad + pressure, 0.5)
        
        # --- Constraint projection phase: Gauss-Seidel relaxation ---
        for _ in range(n_inner):
            # Boundary constraints
            for i in range(N):
                # Left wall
                if pos[i, 0] < rad[i]:
                    pos[i, 0] = rad[i]
                # Right wall
                if pos[i, 0] > 1.0 - rad[i]:
                    pos[i, 0] = 1.0 - rad[i]
                # Bottom wall
                if pos[i, 1] < rad[i]:
                    pos[i, 1] = rad[i]
                # Top wall
                if pos[i, 1] > 1.0 - rad[i]:
                    pos[i, 1] = 1.0 - rad[i]
            
            # Pairwise separation constraints (all pairs for correctness)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.hypot(dx, dy)
                    min_dist = rad[i] + rad[j]
                    if dist < min_dist:
                        if dist < 1e-12:
                            # Degenerate: push apart arbitrarily
                            dx = 0.01
                            dy = 0.01
                            dist = np.hypot(dx, dy)
                        # Equal mass split (or mass-proportional)
                        # Using mass proportional to r^2
                        mi = rad[i] ** 2
                        mj = rad[j] ** 2
                        total_m = mi + mj
                        if total_m < 1e-12:
                            w_i = 0.5
                            w_j = 0.5
                        else:
                            w_i = mj / total_m
                            w_j = mi / total_m
                        overlap = min_dist - dist
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= w_i * overlap * ux
                        pos[i, 1] -= w_i * overlap * uy
                        pos[j, 0] += w_j * overlap * ux
                        pos[j, 1] += w_j * overlap * uy
            
            # Re-clamp to boundaries after pairwise pushes
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], rad[i], 1.0 - rad[i])
                pos[i, 1] = np.clip(pos[i, 1], rad[i], 1.0 - rad[i])
        
        # --- Adapt pressure: if constraints are violated, reduce growth ---
        # Quick violation check
        violation = 0.0
        for i in range(N):
            if pos[i, 0] < rad[i] - 1e-6 or pos[i, 0] > 1.0 - rad[i] + 1e-6:
                violation += 1
            if pos[i, 1] < rad[i] - 1e-6 or pos[i, 1] > 1.0 - rad[i] + 1e-6:
                violation += 1
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad[i] + rad[j] - 1e-6:
                    violation += 1
        
        if violation > 5:
            pressure *= 0.7  # reduce growth
        elif violation == 0 and pressure < 0.005:
            pressure *= 1.05  # slowly increase growth if stable
        
        # Clamp pressure
        pressure = np.clip(pressure, 1e-5, 0.01)
    
    return pos, rad


def _pbd_with_contact_graph(pos, rad, time_budget=10.0):
    """
    PBD simulation using Delaunay contact graph to limit pairwise checks.
    More efficient than all-pairs for dense packings.
    """
    t0 = time.time()
    pos = pos.copy()
    rad = rad.copy()
    
    pressure = 0.0015
    n_inner = 4
    contact_refresh = 10  # refresh Delaunay every N outer iterations
    
    for outer in range(3000):
        if time.time() - t0 > time_budget:
            break
        
        # Refresh contact graph periodically
        if outer % contact_refresh == 0:
            edges = _delaunay_edges(pos)
            # Also add nearby pairs that might be in contact
            # Use a bounding box approach: for each i, check j within 2*(ri+rj)
            edge_list = list(edges)
            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) in edges:
                        continue
                    # Check if potentially close
                    max_r = max(rad[i], rad[j])
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.hypot(dx, dy)
                    if dist < 2.0 * (rad[i] + rad[j] + max_r):
                        edge_list.append((i, j))
        
        # --- Growth phase ---
        rad = np.minimum(rad + pressure, 0.5)
        
        # --- Constraint projection: Gauss-Seidel ---
        for _ in range(n_inner):
            # Boundary constraints
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], rad[i], 1.0 - rad[i])
                pos[i, 1] = np.clip(pos[i, 1], rad[i], 1.0 - rad[i])
            
            # Pairwise separation on contact graph
            for (i, j) in edge_list:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.hypot(dx, dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        dx = 0.01
                        dy = 0.01
                        dist = np.hypot(dx, dy)
                    mi = rad[i] ** 2 + 1e-6
                    mj = rad[j] ** 2 + 1e-6
                    total_m = mi + mj
                    w_i = mj / total_m
                    w_j = mi / total_m
                    overlap = min_dist - dist
                    ux = dx / dist
                    uy = dy / dist
                    pos[i, 0] -= w_i * overlap * ux
                    pos[i, 1] -= w_i * overlap * uy
                    pos[j, 0] += w_j * overlap * ux
                    pos[j, 1] += w_j * overlap * uy
            
            # Re-clamp
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], rad[i], 1.0 - rad[i])
                pos[i, 1] = np.clip(pos[i, 1], rad[i], 1.0 - rad[i])
        
        # Adapt pressure based on constraint satisfaction
        # Sample a few random pairs for quick check
        n_check = min(20, len(edge_list))
        if n_check > 0:
            check_idx = np.random.choice(len(edge_list), size=n_check, replace=False)
            violations = 0
            for ci in check_idx:
                i, j = edge_list[ci]
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad[i] + rad[j] - 1e-5:
                    violations += 1
            if violations > n_check * 0.3:
                pressure *= 0.75
            elif violations == 0 and pressure < 0.004:
                pressure *= 1.03
        
        pressure = np.clip(pressure, 1e-5, 0.008)
    
    return pos, rad


def _local_refine(pos, rad, time_budget=3.0):
    """
    Local refinement: after PBD, do a few iterations of max-radius + small PBD steps
    to squeeze out more value.
    """
    t0 = time.time()
    pos = pos.copy()
    rad = rad.copy()
    
    for it in range(50):
        if time.time() - t0 > time_budget:
            break
        # Compute max possible radii
        rad_max = _max_radius(pos, rad)
        rad = np.maximum(rad, rad_max * 0.999)
        
        # Small PBD step to resolve any conflicts
        pos, rad = _pbd_simulate(pos, rad, time_budget=0.3)
        
        # Final max radius
        rad = _max_radius(pos, rad)
    
    return pos, rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(123)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr):
            # Try to fix by shrinking radii
            for scale in [0.999, 0.995, 0.99, 0.98]:
                rad_try = rad * scale
                arr_try = np.column_stack([pos, rad_try])
                if _valid_check(arr_try):
                    rad = rad_try
                    arr = arr_try
                    break
            else:
                return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr.copy()

    # ---- Generate diverse initializations ----
    starts = []

    # 5x5 grid (25 cells) + 1 extra
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    # Add one more at center of a random cell (perturbed)
    cells.append((0.5 + 0.05, 0.5 + 0.05))
    pos = np.array(cells[:N], dtype=float)
    rad = np.full(N, 0.09)
    rad = _max_radius(pos, rad)
    starts.append((pos.copy(), rad.copy()))

    # 4x6 grid
    cells = []
    for i in range(4):
        for j in range(6):
            cells.append(((i + 0.5) / 4, (j + 0.5) / 6))
    pos = np.array(cells[:N], dtype=float)
    rad = np.full(N, 0.08)
    rad = _max_radius(pos, rad)
    starts.append((pos.copy(), rad.copy()))

    # 6x4 grid
    cells = []
    for i in range(6):
        for j in range(4):
            cells.append(((i + 0.5) / 6, (j + 0.5) / 4))
    pos = np.array(cells[:N], dtype=float)
    rad = np.full(N, 0.08)
    rad = _max_radius(pos, rad)
    starts.append((pos.copy(), rad.copy()))

    # Hexagonal packing
    r_hex = 0.1
    dx = 2 * r_hex * 1.05
    dy = r_hex * np.sqrt(3) * 1.05
    hex_pos = []
    for row in range(8):
        for col in range(8):
            x = col * dx + (row % 2) * dx / 2 + 0.12
            y = row * dy + 0.12
            if 0.08 < x < 0.92 and 0.08 < y < 0.92:
                hex_pos.append((x, y))
    if len(hex_pos) >= N:
        hp = np.array(hex_pos[:N], dtype=float)
        hr = _max_radius(hp, np.full(N, 0.08))
        starts.append((hp, hr))

    # Random initializations
    for k in range(6):
        p0 = rng.random((N, 2)) * 0.7 + 0.15
        r0 = np.full(N, 0.06)
        r0 = _max_radius(p0, r0)
        starts.append((p0, r0))

    # Perturbed grid
    p0 = np.array([((i + 0.5) / 5 + 0.03 * (rng.random() - 0.5),
                    (j + 0.5) / 5 + 0.03 * (rng.random() - 0.5))
                   for i in range(5) for j in range(5)])
    p0 = np.clip(p0, 0.1, 0.9)
    r0 = _max_radius(p0, np.full(N, 0.09))
    starts.append((p0, r0))

    # ---- Run PBD simulation from each start ----
    n_starts = len(starts)
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 15:
            break
        # Allocate budget: more for earlier starts
        remaining = DEADLINE - 15 - time.time()
        budget = min(12.0, max(3.0, remaining / max(1, n_starts - idx)))
        
        # First pass: full PBD
        pos_r, rad_r = _pbd_with_contact_graph(pos0, rad0, time_budget=budget * 0.7)
        consider(pos_r, rad_r)
        
        # Second pass: local refinement
        if time.time() < DEADLINE - 10:
            pos_l, rad_l = _local_refine(pos_r, rad_l if 'rad_l' in dir() else rad_r, time_budget=min(3.0, budget * 0.3))
            consider(pos_l, rad_l)

    # ---- Polish the best solution with more PBD iterations ----
    if best is not None and time.time() < DEADLINE - 8:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        # More aggressive PBD with smaller pressure
        pos_p, rad_p = _pbd_simulate(pos_b, rad_b, time_budget=min(8.0, DEADLINE - 12 - time.time()))
        consider(pos_p, rad_p)
        
        # Final local refinement
        if time.time() < DEADLINE - 5:
            pos_f, rad_f = _local_refine(pos_p, rad_p, time_budget=min(3.0, DEADLINE - 6 - time.time()))
            consider(pos_f, rad_f)

    # ---- Perturbation-based diversification on best ----
    if best is not None and time.time() < DEADLINE - 12:
        for k in range(5):
            if time.time() > DEADLINE - 6:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            # Perturb random subset
            n_perturb = rng.integers(3, 8)
            idx_perturb = rng.choice(N, size=n_perturb, replace=False)
            pos_p[idx_perturb] += 0.015 * (rng.random((n_perturb, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.05, 0.95)
            rad_p = _max_radius(pos_p, rad_p)
            
            budget = min(5.0, DEADLINE - 8 - time.time())
            if budget > 1.0:
                pos_f, rad_f = _pbd_with_contact_graph(pos_p, rad_p, time_budget=budget)
                consider(pos_f, rad_f)

    if best is None:
        # Fallback: 5x5 grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.095)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
