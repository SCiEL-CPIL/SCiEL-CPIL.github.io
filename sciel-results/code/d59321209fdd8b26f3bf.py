import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 144.0  # Safety margin before 150s wall-clock kill

def _cmaes_minimize(objective, x0, sigma0, maxiter=200, time_limit=None, t0=None, seed=None):
    """Simple CMA-ES implementation for the smooth surrogate."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    n = len(x0)
    x = np.array(x0, dtype=float)
    sigma = sigma0
    C = np.eye(n)
    C_inv = np.eye(n)
    
    # Learning rates
    mu = max(1, n // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 - mu_eff/n + c_sigma) / (n + 4 - mu_eff/n + c_sigma)
    c_1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0/mu_eff) / ((n + 2)**2 + mu_eff))
    
    # State
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    
    # Eigen decomposition cache
    eigvals, eigvecs = np.linalg.eigh(C)
    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
    C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
    
    best_x = x.copy()
    best_f = objective(x)
    
    for it in range(maxiter):
        if time_limit is not None and t0 is not None:
            if time.time() - t0 > time_limit:
                break
        
        # Generate population
        pop_x = np.zeros((mu, n))
        pop_y = np.zeros((mu, n))
        for i in range(mu):
            z = rng.randn(n)
            pop_y[i] = z
            pop_x[i] = x + sigma * (C_sqrt @ z)
        
        # Evaluate
        pop_f = np.array([objective(xi) for xi in pop_x])
        
        # Sort
        idx = np.argsort(pop_f)
        pop_x = pop_x[idx]
        pop_y = pop_y[idx]
        pop_f = pop_f[idx]
        
        # Update mean
        old_x = x.copy()
        x = np.sum(weights[:, None] * pop_x, axis=0)
        
        # Update sigma
        d = (np.linalg.norm(x - old_x) / (sigma * np.linalg.norm(C_sqrt @ pop_y[0]))) if np.linalg.norm(C_sqrt @ pop_y[0]) > 1e-15 else 0
        p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((C_sqrt_inv @ (x - old_x)) / sigma)
        h_sigma = 1.0 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1-c_sigma)**(2*(it+1)))) / chi_n < 1.4 + 2/(n+1) else 0.0
        sigma *= math.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Update C
        p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ (x - old_x) / sigma)
        
        rank_mu = np.outer(C_sqrt_inv @ pop_x, C_sqrt_inv @ pop_x)
        rank_mu = np.sum(weights[:, None, None] * rank_mu, axis=0)
        
        C = (1 + c_1 + c_mu - c_c) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
        C_inv = np.linalg.inv(C)
        
        # Re-estimate eigen decomposition periodically
        if it % 5 == 0:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Track best
        if pop_f[0] < best_f:
            best_f = pop_f[0]
            best_x = pop_x[0].copy()
        
        # Check convergence
        if sigma < 1e-10:
            break
    
    return best_x, best_f

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Vectorized coarse scan: evaluate 120 angles (0.5° step) at once
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)  # (120, 6)
        ny = np.sin(normal_angles)  # (120, 6)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)  # (120, 72, 6)
        max_projs = np.max(abs_proj, axis=1)  # (120, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (120, 6)
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx * 0.5)
        
        # Fine tune with vectorized sub-degree scan (much faster than minimize_scalar)
        try:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            # 500 points over 1 degree = 0.002° resolution
            fine_phis = np.linspace(lo, hi, 500)
            fine_rad = np.radians(fine_phis)
            fine_normals = fine_rad[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            fnx = np.cos(fine_normals)  # (500, 6)
            fny = np.sin(fine_normals)  # (500, 6)
            fproj = offsets[None, :, 0:1] * fnx[:, None, :] + offsets[None, :, 1:2] * fny[:, None, :]
            fabs = np.abs(fproj)  # (500, 72, 6)
            fmax = np.max(fabs, axis=1)  # (500, 6)
            fsides = (2.0 / sqrt3) * fmax  # (500, 6)
            fbest_idx = np.argmin(np.max(fsides, axis=1))
            fbest_side = np.max(fsides[fbest_idx])
            fbest_phi = float(fine_phis[fbest_idx])
            
            # Second refinement: 200 points over ±0.002° around best
            if fbest_side < best_side:
                lo2 = max(0.0, fbest_phi - 0.002)
                hi2 = min(60.0, fbest_phi + 0.002)
                fine_phis2 = np.linspace(lo2, hi2, 200)
                fine_rad2 = np.radians(fine_phis2)
                fine_normals2 = fine_rad2[:, None] + np.pi/6 + np.arange(6) * np.pi/3
                fnx2 = np.cos(fine_normals2)
                fny2 = np.sin(fine_normals2)
                fproj2 = offsets[None, :, 0:1] * fnx2[:, None, :] + offsets[None, :, 1:2] * fny2[:, None, :]
                fabs2 = np.abs(fproj2)
                fmax2 = np.max(fabs2, axis=1)
                fsides2 = (2.0 / sqrt3) * fmax2
                fbest_idx2 = np.argmin(np.max(fsides2, axis=1))
                fbest_side2 = np.max(fsides2[fbest_idx2])
                fbest_phi2 = float(fine_phis2[fbest_idx2])
                
                if fbest_side2 < best_side:
                    best_side = fbest_side2
                    best_phi = fbest_phi2
                else:
                    best_side = fbest_side
                    best_phi = fbest_phi
            else:
                best_side = fbest_side
                best_phi = fbest_phi
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def smooth_surrogate_objective(params, temperature=0.05):
        """Smooth min-max surrogate using log-sum-exp for differentiability."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        # Check collisions - if any, return large penalty
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Soft penalty: how much they overlap
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    return 100.0 + max(0, 2*sqrt3 - dist) * 10
        
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # For a fixed outer hexagon orientation, compute required side in 6 directions
        # Use smooth aggregation over the 6 face normals
        phi = 0.0  # We'll optimize phi separately; here use best estimate
        # Actually, let's include phi in the smooth objective by averaging over a range
        # For CMA-ES, we fix phi to the current best and optimize positions/angles
        
        normal_angles = np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)  # (6,)
        side_vals = (2.0 / sqrt3) * max_projs  # (6,)
        
        # Smooth max via log-sum-exp
        beta = 1.0 / temperature
        max_sv = np.max(side_vals)
        smooth_max = max_sv + (1.0/beta) * np.log(np.sum(np.exp(beta * (side_vals - max_sv))))
        
        return smooth_max

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 2*sqrt3 * math.cos(a)
            positions[k, 1] = 2*sqrt3 * math.sin(a)
        positions[6] = [0, 0]
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3.5*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3.5*sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles

    def make_triangular():
        # Triangular arrangement: 3-4-3 or similar
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        # Row 1: 3 hexagons
        for k in range(3):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
        # Row 2: 4 hexagons
        for k in range(4):
            positions[3+k, 0] = k * 2*sqrt3 + sqrt3/2
            positions[3+k, 1] = 3.0
        # Row 3: 3 hexagons
        for k in range(3):
            positions[7+k, 0] = k * 2*sqrt3
            positions[7+k, 1] = 6.0
        return positions, angles

    def make_compact_hex():
        # More compact: center + 6 around + 5 more
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        for k in range(6):
            a = k * np.pi / 3
            positions[1+k, 0] = 2*sqrt3 * math.cos(a)
            positions[1+k, 1] = 2*sqrt3 * math.sin(a)
        # 5 more at distance ~3*sqrt3
        for k in range(5):
            a = k * 2*np.pi/5
            positions[7+k, 0] = 3*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3*sqrt3 * math.sin(a)
        return positions, angles

    # Evaluate multiple initial configurations
    inits = [make_honeycomb(), make_ring_config(), make_two_rows(), make_triangular(), make_compact_hex()]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos_init, ang_init in inits:
        s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if s < best_side:
            best_side = s
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # Strategy: Run SA for 65% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.65
    
    # Start SA from best initial configuration
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side = best_side
    current_phi = best_phi
    current_cx = best_cx
    current_cy = best_cy

    T = 0.35
    T_min = 0.00003
    cooling = 0.99997
    step_size = 0.3
    step_decay = 0.99998
    
    step_count = 0
    max_steps = 300000
    accept_count = 0
    total_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 4.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue
            
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        
        # Adaptive step size
        if total_count % 300 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.15:
                step_size *= 0.85
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.45:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. CMA-ES with smooth surrogate objective
        cmaes_time = min(remaining_time * 0.5, 40.0)
        
        if cmaes_time > 5.0:
            # Build initial params from best configuration
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
            
            # Create objective that uses current best phi for the smooth surrogate
            current_phi_for_cmaes = best_phi
            
            def cmaes_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                
                # Collision check
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            dx = h1[0] - h2[0]
                            dy = h1[1] - h2[1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            return 100.0 + max(0, 2*sqrt3 - dist) * 10
                
                all_verts = get_all_vertices(pos, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                offsets = verts_flat - center
                
                # Evaluate at current best phi
                phi = np.radians(current_phi_for_cmaes)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                
                # Smooth max via log-sum-exp
                beta = 20.0  # temperature = 0.05
                max_sv = np.max(side_vals)
                smooth_max = max_sv + (1.0/beta) * np.log(np.sum(np.exp(beta * (side_vals - max_sv))))
                
                return smooth_max
            
            try:
                cmaes_t0 = time.time()
                cmaes_maxiter = 100
                cmaes_result_x, cmaes_result_f = _cmaes_minimize(
                    cmaes_objective, init_params, sigma0=0.15,
                    maxiter=cmaes_maxiter, time_limit=cmaes_time - 1.0, t0=cmaes_t0, seed=42
                )
                
                # Extract and validate
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = cmaes_result_x[3*i]
                    new_pos[i, 1] = cmaes_result_x[3*i+1]
                    new_ang[i] = cmaes_result_x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
            except Exception:
                pass
        
        # 2. Coordinate Descent Polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            def polish_coordinate_descent(pos, ang, time_limit):
                current_side, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()
                
                t_start = time.time()
                step = 0.02
                min_step = 0.0003
                
                while time.time() - t_start < time_limit and step > min_step:
                    improved = False
                    for idx in range(N):
                        for dim in range(3):
                            for direction in [1, -1]:
                                new_pos = best_pos_cd.copy()
                                new_ang = best_ang_cd.copy()
                                
                                if dim == 0:
                                    new_pos[idx, 0] += direction * step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * step
                                else:
                                    new_ang[idx] += direction * step
                                
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                
                                if new_side < best_side_cd - 1e-10:
                                    best_pos_cd = new_pos
                                    best_ang_cd = new_ang
                                    best_side_cd = new_side
                                    improved = True
                                    break
                                
                                if time.time() - t_start > time_limit:
                                    return best_side_cd, best_pos_cd, best_ang_cd
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        step *= 0.75
                        continue
                    break
                
                return best_side_cd, best_pos_cd, best_ang_cd

            cd_time = min(remaining_time * 0.4, 15.0)
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
            
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 3. Second CMA-ES round with updated phi (if time allows)
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 10.0:
            cmaes_time2 = min(remaining_time * 0.5, 25.0)
            
            if cmaes_time2 > 5.0:
                init_params2 = np.zeros(3*N)
                for i in range(N):
                    init_params2[3*i] = best_positions[i, 0]
                    init_params2[3*i+1] = best_positions[i, 1]
                    init_params2[3*i+2] = best_angles[i]
                
                current_phi_for_cmaes2 = best_phi
                
                def cmaes_objective2(params):
                    pos = np.zeros((N, 2))
                    ang = np.zeros(N)
                    for i in range(N):
                        pos[i, 0] = params[3*i]
                        pos[i, 1] = params[3*i+1]
                        ang[i] = params[3*i+2]
                    
                    angles_rad = np.radians(ang)
                    for i in range(N):
                        for j in range(i+1, N):
                            h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                            h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                            if not hexagons_disjoint_fast(h1, h2):
                                dx = h1[0] - h2[0]
                                dy = h1[1] - h2[1]
                                dist = math.sqrt(dx*dx + dy*dy)
                                return 100.0 + max(0, 2*sqrt3 - dist) * 10
                    
                    all_verts = get_all_vertices(pos, angles_rad)
                    verts_flat = all_verts.reshape(-1, 2)
                    center = np.mean(verts_flat, axis=0)
                    offsets = verts_flat - center
                    
                    phi = np.radians(current_phi_for_cmaes2)
                    normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                    nx = np.cos(normal_angles)
                    ny = np.sin(normal_angles)
                    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                    abs_proj = np.abs(proj)
                    max_projs = np.max(abs_proj, axis=0)
                    side_vals = (2.0 / sqrt3) * max_projs
                    
                    beta = 30.0
                    max_sv = np.max(side_vals)
                    smooth_max = max_sv + (1.0/beta) * np.log(np.sum(np.exp(beta * (side_vals - max_sv))))
                    
                    return smooth_max
                
                try:
                    cmaes_t0_2 = time.time()
                    cmaes_result_x2, cmaes_result_f2 = _cmaes_minimize(
                        cmaes_objective2, init_params2, sigma0=0.05,
                        maxiter=80, time_limit=cmaes_time2 - 1.0, t0=cmaes_t0_2, seed=123
                    )
                    
                    new_pos2 = np.zeros((N, 2))
                    new_ang2 = np.zeros(N)
                    for i in range(N):
                        new_pos2[i, 0] = cmaes_result_x2[3*i]
                        new_pos2[i, 1] = cmaes_result_x2[3*i+1]
                        new_ang2[i] = cmaes_result_x2[3*i+2]
                    
                    if is_valid_configuration(new_pos2, new_ang2):
                        new_side2, new_phi2, new_cx2, new_cy2 = compute_min_outer_side(new_pos2, new_ang2)
                        if new_side2 < best_side:
                            best_side = new_side2
                            best_positions = new_pos2
                            best_angles = new_ang2
                            best_phi = new_phi2
                            best_cx = new_cx2
                            best_cy = new_cy2
                except Exception:
                    pass
        
        # 4. Final high-precision coordinate descent with multi-start local search
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            final_cd_time = min(remaining_time * 0.6, 12.0)
            
            def polish_coordinate_descent_final(pos, ang, time_limit):
                current_side, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()
                
                t_start = time.time()
                step = 0.008
                min_step = 0.00005
                
                while time.time() - t_start < time_limit and step > min_step:
                    improved = False
                    for idx in range(N):
                        for dim in range(3):
                            for direction in [1, -1]:
                                new_pos = best_pos_cd.copy()
                                new_ang = best_ang_cd.copy()
                                
                                if dim == 0:
                                    new_pos[idx, 0] += direction * step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * step
                                else:
                                    new_ang[idx] += direction * step
                                
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                
                                if new_side < best_side_cd - 1e-12:
                                    best_pos_cd = new_pos
                                    best_ang_cd = new_ang
                                    best_side_cd = new_side
                                    improved = True
                                    break
                                
                                if time.time() - t_start > time_limit:
                                    return best_side_cd, best_pos_cd, best_ang_cd
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        step *= 0.65
                        continue
                    break
                
                return best_side_cd, best_pos_cd, best_ang_cd

            final_side, final_pos, final_ang = polish_coordinate_descent_final(best_positions, best_angles, final_cd_time)
            if final_side < best_side:
                best_side = final_side
                best_positions = final_pos
                best_angles = final_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
            
            # 5. Second polish round if time remains
            remaining_time = DEADLINE - (time.time() - t0)
            if remaining_time > 3.0:
                second_cd_time = min(remaining_time * 0.5, 6.0)
                second_side, second_pos, second_ang = polish_coordinate_descent_final(best_positions, best_angles, second_cd_time)
                if second_side < best_side:
                    best_side = second_side
                    best_positions = second_pos
                    best_angles = second_ang
                    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos_f, ang_f in inits:
            if is_valid_configuration(pos_f, ang_f):
                best_positions = pos_f.copy()
                best_angles = ang_f.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
