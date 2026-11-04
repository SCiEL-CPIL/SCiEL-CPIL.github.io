import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    # Unit hexagon vertices (flat-top, circumradius 1)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """Min side of regular hexagon (center 0,0, rot outer_angle) containing all inner hexagons."""
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k*math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k*math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            projs = verts @ normals.T
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
        return max_proj * 2.0 / SQRT3

    def sat_gap_fast(poly1, poly2):
        """SAT gap: positive if separated, negative if overlapping (min penetration)."""
        min_gap = float('inf')
        # Combine edge normals from both polygons
        axes = []
        for poly in (poly1, poly2):
            for i in range(6):
                edge = poly[(i+1)%6] - poly[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    axes.append(n / norm)
        for axis in axes:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap >= 0:
                return gap  # separated
            min_gap = min(min_gap, gap)
        return min_gap

    def check_validity(positions, angles, tol=1e-4):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap_fast(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- Epigraph objective: minimize t subject to containment + disjointness ---
    # For smoothness, use log-sum-exp soft-max for the containment projection.
    def soft_max(vals, beta=50.0):
        m = np.max(vals)
        return m + np.log(np.sum(np.exp(beta*(vals - m)))) / beta

    def objective_epigraph(params, outer_angle=0.0):
        """params = [positions (2N), angles (N), t]. Minimize t."""
        if time.time() > deadline - 3:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        t = params[3*N]
        
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k*math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k*math.pi/3)] for k in range(6)])
        
        # Containment: for each inner hex, max projection over its vertices <= t*SQRT3/2
        # Use soft-max to smooth
        all_proj = []
        for i in range(N):
            verts = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            projs = verts @ normals.T  # (6, 6)
            all_proj.append(np.max(projs, axis=1))  # max over vertices for each normal
        all_proj = np.array(all_proj)  # (N, 6)
        # For each normal, max over all inner hexagons
        max_per_normal = np.max(all_proj, axis=0)  # (6,)
        # t must be >= max_per_normal * 2 / SQRT3
        required_t = np.max(max_per_normal) * 2.0 / SQRT3
        # Penalty if t < required_t
        if t < required_t:
            return 1e6 + (required_t - t)**2
        return t

    def make_disjoint_constraint(i, j):
        def constraint(params):
            if time.time() > deadline - 3:
                return 1.0
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            poly_i = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            poly_j = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
            return sat_gap_fast(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_disjoint_constraint(i, j)})

    # --- Initializations ---
    def honeycomb_init():
        """12 hexagons in honeycomb, exactly touching (distance 2)."""
        # Honeycomb lattice: (1.5*i + 0.5*j, (sqrt3/2)*i + sqrt3*j)
        # Nearest neighbors at distance 2.
        # Pick 12 points forming a compact cluster.
        # Center at (0,0), then 6 around at distance 2, then 5 more.
        pts = [(0, 0)]
        # 6 neighbors at distance 2
        for k in range(6):
            a = k * math.pi / 3
            pts.append((2*math.cos(a), 2*math.sin(a)))
        # 5 more: place in the "gaps" at distance ~ 2*sqrt(3) from center
        # Actually, let's use proper honeycomb coordinates.
        # Lattice points (i,j) -> (1.5*i + 0.5*j, (sqrt3/2)*i + sqrt3*j)
        # Distance between (i,j) and (i',j') = 2*sqrt((i-i')^2 + (i-i')(j-j') + (j-j')^2)
        # Let's just pick 12 close lattice points.
        pts2 = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = 1.5*i + 0.5*j
                y = (SQRT3/2)*i + SQRT3*j
                pts2.append((x, y, x*x + y*y))
        pts2.sort(key=lambda p: p[2])
        # Take 12 closest
        chosen = pts2[:12]
        positions = np.array([[p[0], p[1]] for p in chosen])
        positions -= np.mean(positions, axis=0)
        angles = np.zeros(N)
        t_init = calc_min_outer_side(positions, angles, 0.0) + 0.5
        return np.concatenate([positions.flatten(), angles, [t_init]])

    def ring_init():
        """Center + 6 ring + 5 outer."""
        positions = np.zeros((N, 2))
        positions[0] = [0, 0]
        for i in range(1, 7):
            a = (i-1) * math.pi / 3
            positions[i] = [2*math.cos(a), 2*math.sin(a)]
        for i in range(7, 12):
            a = (i-7) * 2*math.pi/5 + math.pi/5
            positions[i] = [3.5*math.cos(a), 3.5*math.sin(a)]
        positions -= np.mean(positions, axis=0)
        angles = np.zeros(N)
        t_init = calc_min_outer_side(positions, angles, 0.0) + 0.5
        return np.concatenate([positions.flatten(), angles, [t_init]])

    def grid_init():
        """3x4 grid with hex spacing."""
        positions = np.zeros((N, 2))
        idx = 0
        for i in range(3):
            for j in range(4):
                x = (j - 1.5) * 2.1
                y = (i - 1) * SQRT3 * 1.05
                positions[idx] = [x, y]
                idx += 1
        positions -= np.mean(positions, axis=0)
        angles = np.zeros(N)
        t_init = calc_min_outer_side(positions, angles, 0.0) + 0.5
        return np.concatenate([positions.flatten(), angles, [t_init]])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        positions = rng.uniform(-3, 3, size=(N, 2))
        angles = rng.uniform(0, 360, size=N)
        t_init = calc_min_outer_side(positions, angles, 0.0) + 1.0
        return np.concatenate([positions.flatten(), angles, [t_init]])

    initial_configs = [
        honeycomb_init(),
        ring_init(),
        grid_init(),
        random_init(1),
        random_init(42),
        random_init(123),
    ]

    bounds = []
    for i in range(N):
        bounds.append((-8, 8))
        bounds.append((-8, 8))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((2.0, 10.0))  # t

    best_score = float('inf')
    best_params = None

    # --- Fast Local Search (Hill Climbing with Random Restarts) ---
    # Replaces slow SLSQP and heavy SA with a fast, robust local search.
    
    def fast_local_search(start_params, max_iter=5000, step_size=0.5):
        """
        Hill climbing that only accepts valid moves.
        Fast because it rejects invalid moves immediately without calculating score.
        """
        params = start_params.copy()
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        # Ensure start is valid
        if not check_validity(positions, angles):
            return None, float('inf')
        
        current_score = calc_min_outer_side(positions, angles, 0.0)
        best_params_ls = params.copy()
        best_score_ls = current_score
        
        rng = np.random.RandomState(123)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            new_params = params.copy()
            # Perturb one or two hexagons
            num_perturb = rng.randint(1, 3)
            for _ in range(num_perturb):
                i = rng.randint(N)
                new_params[2*i] += rng.uniform(-step_size, step_size)
                new_params[2*i+1] += rng.uniform(-step_size, step_size)
                if rng.random() < 0.1:
                    new_params[2*N+i] = (new_params[2*N+i] + rng.uniform(-10, 10)) % 360
            
            new_positions = new_params[:2*N].reshape(N, 2)
            new_angles = new_params[2*N:3*N]
            
            # Fast validity check
            if not check_validity(new_positions, new_angles):
                continue
            
            new_score = calc_min_outer_side(new_positions, new_angles, 0.0)
            
            if new_score < current_score:
                params = new_params
                current_score = new_score
                if new_score < best_score_ls:
                    best_score_ls = new_score
                    best_params_ls = params.copy()
                # Reduce step size slightly on improvement to converge
                step_size = max(0.05, step_size * 0.95)
            else:
                # Occasionally reset step size to escape local minima
                if rng.random() < 0.01:
                    step_size = 0.5
        
        return best_params_ls, best_score_ls

    # Run local search from multiple initial configurations
    for x0 in initial_configs:
        if time.time() > deadline - 10:
            break
        
        # First, ensure the initial config is valid. If not, try to fix it with a few random moves?
        # For simplicity, if invalid, skip or try a quick repair.
        pos0 = x0[:2*N].reshape(N, 2)
        ang0 = x0[2*N:3*N]
        if not check_validity(pos0, ang0):
            # Try to repair by pushing overlapping hexagons apart
            repaired = x0.copy()
            for _ in range(100):
                pos_r = repaired[:2*N].reshape(N, 2)
                ang_r = repaired[2*N:3*N]
                if check_validity(pos_r, ang_r):
                    break
                # Find overlapping pair and push apart
                polys = [get_hex_verts(pos_r[i][0], pos_r[i][1], 1.0, ang_r[i]) for i in range(N)]
                moved = False
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap_fast(polys[i], polys[j])
                        if gap < 0:
                            # Push i and j apart
                            dx = pos_r[i][0] - pos_r[j][0]
                            dy = pos_r[i][1] - pos_r[j][1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 1e-6:
                                dx, dy = 1, 0
                                dist = 1
                            else:
                                dx /= dist
                                dy /= dist
                            push = (2.0 - dist) * 0.5 + 0.1
                            repaired[2*i] += dx * push
                            repaired[2*i+1] += dy * push
                            repaired[2*j+1] -= dy * push
                            repaired[2*j+1] -= dy * push # Corrected: 2*j+1 is y
                            repaired[2*j] -= dx * push
                            moved = True
                            break
                    if moved:
                        break
            if check_validity(repaired[:2*N].reshape(N, 2), repaired[2*N:3*N]):
                x0 = repaired
            else:
                continue # Skip if we can't repair quickly
        
        ls_params, ls_score = fast_local_search(x0, max_iter=2000)
        if ls_params is not None and ls_score < best_score:
            best_score = ls_score
            best_params = ls_params

    # --- Finalize ---
    if best_params is not None:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:3*N]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Fallback: honeycomb
        positions = np.zeros((N, 2))
        positions[0] = [0, 0]
        for i in range(1, 7):
            a = (i-1) * math.pi / 3
            positions[i] = [2*math.cos(a), 2*math.sin(a)]
        for i in range(7, 12):
            a = (i-7) * 2*math.pi/5 + math.pi/5
            positions[i] = [3.5*math.cos(a), 3.5*math.sin(a)]
        positions -= np.mean(positions, axis=0)
        final_positions = positions
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
