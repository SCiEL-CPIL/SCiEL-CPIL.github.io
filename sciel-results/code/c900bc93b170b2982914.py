"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines multi-phase local optimization with a compact structural
parameterization (corner + center + rings) that reduces the global-search
dimensionality so Differential Evolution can actually explore it."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth log-sum-exp penalty."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations - smooth log-sum-exp penalty
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bot_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)
    
    def _soft_max(v, beta=30.0):
        # Approximate max(0, v) using log-sum-exp for smoothness
        # We want to penalize positive violations heavily
        # Use: (1/beta) * log(sum(exp(beta * v))) for v > 0, but we need max(0,v)
        # Better: use a smooth hinge. 
        # Let's use: (v + sqrt(v^2 + eps)) / 2 is C1 but not C2.
        # Log-sum-exp of (v, 0) is smooth.
        # LSE(v, 0) = log(exp(v) + 1). 
        # If v >> 0, LSE ~ v. If v << 0, LSE ~ exp(v) ~ 0.
        # This is a smooth approximation of max(v, 0).
        # To make it sharper, scale v by beta.
        # LSE(beta*v, 0)/beta = (1/beta) * log(exp(beta*v) + 1)
        # For numerical stability: (1/beta) * (max(beta*v, 0) + log1p(exp(-abs(beta*v))))
        
        v_scaled = beta * v
        # max(v_scaled, 0)
        vmax = np.maximum(v_scaled, 0.0)
        # log1p(exp(-abs(v_scaled)))
        # If v_scaled is very negative, exp(-abs(v_scaled)) is 0, log1p(0)=0.
        # If v_scaled is very positive, exp(-abs(v_scaled)) is 0, log1p(0)=0.
        # Wait, if v_scaled > 0, abs = v_scaled, exp(-v_scaled) is small.
        # If v_scaled < 0, abs = -v_scaled, exp(v_scaled) is small.
        # So:
        term = vmax + np.log1p(np.exp(-np.abs(v_scaled)))
        return term / beta

    left_s = _soft_max(left_v)
    right_s = _soft_max(right_v)
    bot_s = _soft_max(bot_v)
    top_s = _soft_max(top_v)
    bpen = np.sum(left_s) + np.sum(right_s) + np.sum(bot_s) + np.sum(top_s)

    # Overlap violations - smooth penalty
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)
    viol_s = _soft_max(viol)
    np.fill_diagonal(viol_s, 0.0)
    open_pen = np.sum(viol_s)

    return -np.sum(radii) + 50.0 * (bpen + open_pen)


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Structural seeds ----------------

def _make_grid(n, seed=42):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            idx = row * 5 + col
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


# ---------------- Post-processing & polish ----------------

def repair_and_validate(pos, radii, deadline):
    """
    Robustly repair a configuration to ensure all circles are inside the unit square
    and have disjoint interiors. Returns (pos, radii, is_valid).
    """
    pos = pos.copy()
    radii = radii.copy()
    n = len(radii)
    tol = 1e-9
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        valid = True
        
        # 1. Enforce boundary constraints by shrinking radii
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if max_r < 1e-9:
                # If center is on boundary or outside, move it in and shrink
                pos[i, 0] = np.clip(pos[i, 0], 1e-6, 1.0 - 1e-6)
                pos[i, 1] = np.clip(pos[i, 1], 1e-6, 1.0 - 1e-6)
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            
            if radii[i] > max_r:
                radii[i] = max_r * (1.0 - tol)
                valid = False
        
        # 2. Enforce disjoint interiors
        # We iterate through pairs and resolve overlaps.
        # Strategy: If dist < r_i + r_j, we must either move centers or shrink radii.
        # Moving centers is preferred to preserve area, but shrinking is safer for termination.
        # We use a mix: try to separate if possible, otherwise shrink.
        
        # First pass: check for overlaps
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist - tol:
                    overlaps.append((i, j, dist_sq, min_dist))
        
        if not overlaps:
            break
        
        valid = False
        
        # Resolve overlaps
        # We process overlaps in order of severity (largest overlap first)
        # Sort by (min_dist - dist)
        def overlap_size(item):
            return item[3] - np.sqrt(item[2])
        
        overlaps.sort(key=overlap_size, reverse=True)
        
        for i, j, dist_sq, min_dist in overlaps:
            dist = np.sqrt(dist_sq)
            if dist < 1e-9:
                # Centers coincide, push them apart slightly
                pos[i, 0] += 1e-6
                pos[j, 0] -= 1e-6
                continue
            
            overlap = min_dist - dist
            
            # Determine how much to shrink or move
            # We will shrink both radii slightly and move them apart slightly.
            # This ensures progress towards validity.
            
            # Move apart
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            
            # We want to increase dist by 'overlap'.
            # Let's move i by +overlap/2 * dir and j by -overlap/2 * dir
            # But we must respect boundaries.
            
            move_dist = overlap * 0.5
            dir_x = dx / dist
            dir_y = dy / dist
            
            new_pos_i = [pos[i, 0] + move_dist * dir_x, pos[i, 1] + move_dist * dir_y]
            new_pos_j = [pos[j, 0] - move_dist * dir_x, pos[j, 1] - move_dist * dir_y]
            
            # Check if new positions are valid w.r.t boundaries
            # If not, we shrink radii instead
            valid_i = (new_pos_i[0] >= radii[i] - tol and new_pos_i[0] <= 1.0 - radii[i] + tol and
                       new_pos_i[1] >= radii[i] - tol and new_pos_i[1] <= 1.0 - radii[i] + tol)
            valid_j = (new_pos_j[0] >= radii[j] - tol and new_pos_j[0] <= 1.0 - radii[j] + tol and
                       new_pos_j[1] >= radii[j] - tol and new_pos_j[1] <= 1.0 - radii[j] + tol)
            
            if valid_i and valid_j:
                pos[i] = new_pos_i
                pos[j] = new_pos_j
            else:
                # Shrink radii to eliminate overlap
                # We need r_i + r_j <= dist
                # Let r_i' + r_j' = dist * (1 - tol)
                # Distribute shrinkage proportional to current radii? Or equally?
                # Let's shrink the smaller one more to keep larger ones stable?
                # Or just shrink both to fit.
                target_sum = dist * (1.0 - tol)
                current_sum = radii[i] + radii[j]
                
                if current_sum > 0:
                    scale_i = radii[i] / current_sum
                    scale_j = radii[j] / current_sum
                    radii[i] = target_sum * scale_i
                    radii[j] = target_sum * scale_j
                else:
                    radii[i] = 0.0
                    radii[j] = 0.0
        
        # Final check for this iteration
        # If we exited the loop without breaking, we need to check again
        # The loop will restart and re-check.
        
    # Final validation
    is_valid = _is_valid(pos, radii, tol)
    
    # If still not valid (e.g. due to time limit or numerical issues),
    # perform a drastic fallback: shrink all radii by a factor
    if not is_valid:
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            radii[i] = min(radii[i], max_r) * 0.95
            
        # Repeat a few quick checks
        for _ in range(10):
            if time.time() > deadline:
                break
            valid = True
            for i in range(n):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                if radii[i] > max_r:
                    radii[i] = max_r * 0.99
                    valid = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Shrink both
                        shrink_i = overlap * 0.5
                        shrink_j = overlap * 0.5
                        radii[i] = max(1e-9, radii[i] - shrink_i)
                        radii[j] = max(1e-9, radii[j] - shrink_j)
                        valid = False
            if valid:
                break
        
        is_valid = _is_valid(pos, radii, tol)
        
    return pos, radii, is_valid


def _polish(pos, radii, deadline):
    """Multi-stage high-precision polish: L-BFGS-B -> SLSQP -> coordinate descent."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.3), min(1.0, 1.0 - radii[i] * 0.3)))
        bounds.append((max(0.0, radii[i] * 0.3), min(1.0, 1.0 - radii[i] * 0.3)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    best_p, best_r = pos.copy(), radii.copy()
    best_s = np.sum(radii)

    # Stage 1: L-BFGS-B
    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp, is_valid_pp = repair_and_validate(pos_opt, radii_opt, deadline)
        if is_valid_pp and np.sum(radii_pp) > best_s:
            best_s = np.sum(radii_pp)
            best_p, best_r = pos_pp.copy(), radii_pp.copy()
    except Exception:
        pass

    # Stage 2: SLSQP from best
    if time.time() < deadline - 4:
        try:
            x2 = np.concatenate([best_p.flatten(), best_r])
            bounds2 = []
            for i in range(N):
                bounds2.append((max(0.0, best_r[i] * 0.6), min(1.0, 1.0 - best_r[i] * 0.6)))
                bounds2.append((max(0.0, best_r[i] * 0.6), min(1.0, 1.0 - best_r[i] * 0.6)))
            for i in range(N):
                bounds2.append((max(1e-4, best_r[i] * 0.85), min(0.35, best_r[i] * 1.15 + 0.01)))
            res2 = minimize(lambda x: _objective(x, deadline), x2, method='SLSQP',
                            bounds=bounds2,
                            options={'maxiter': 500, 'ftol': 1e-15, 'disp': False})
            pos_opt2 = res2.x[:2 * N].reshape(N, 2)
            radii_opt2 = res2.x[2 * N:]
            pos_pp2, radii_pp2, is_valid_pp2 = repair_and_validate(pos_opt2, radii_opt2, deadline)
            if is_valid_pp2 and np.sum(radii_pp2) > best_s:
                best_s = np.sum(radii_pp2)
                best_p, best_r = pos_pp2.copy(), radii_pp2.copy()
        except Exception:
            pass

    # Stage 3: Coordinate descent on positions and radii
    if time.time() < deadline - 1:
        p_cd = best_p.copy()
        r_cd = best_r.copy()
        for _ in range(5):
            improved = False
            for i in range(N):
                if time.time() > deadline - 0.5:
                    break
                # Try growing radius
                for growth in [1.003, 1.001, 1.0005, 1.0001]:
                    new_r = r_cd[i] * growth
                    max_r = min(p_cd[i, 0], 1.0 - p_cd[i, 0], p_cd[i, 1], 1.0 - p_cd[i, 1])
                    if new_r > max_r:
                        new_r = max_r * 0.9999
                    old_r = r_cd[i]
                    r_cd[i] = new_r
                    if _is_valid(p_cd, r_cd):
                        improved = True
                        break
                    else:
                        r_cd[i] = old_r
                # Try small position adjustments
                for dx in [0.001, -0.001, 0.0005, -0.0005]:
                    for dy in [0.001, -0.001, 0.0005, -0.0005]:
                        old_x, old_y = p_cd[i, 0], p_cd[i, 1]
                        p_cd[i, 0] = old_x + dx
                        p_cd[i, 1] = old_y + dy
                        if p_cd[i, 0] < r_cd[i] or p_cd[i, 0] > 1 - r_cd[i]:
                            p_cd[i, 0] = old_x
                            continue
                        if p_cd[i, 1] < r_cd[i] or p_cd[i, 1] > 1 - r_cd[i]:
                            p_cd[i, 1] = old_y
                            continue
                        if _is_valid(p_cd, r_cd):
                            # Check if we can grow any circle
                            any_grown = False
                            for j in range(N):
                                max_rj = min(p_cd[j, 0], 1.0 - p_cd[j, 0], p_cd[j, 1], 1.0 - p_cd[j, 1])
                                if r_cd[j] < max_rj * 0.9999:
                                    r_cd[j] = max_rj * 0.9999
                                    any_grown = True
                            if any_grown or True:
                                improved = True
                                break
                        else:
                            p_cd[i, 0] = old_x
                            p_cd[i, 1] = old_y
                    if improved:
                        break
            if not improved:
                break
        if _is_valid(p_cd, r_cd) and np.sum(r_cd) > best_s:
            best_p, best_r = p_cd.copy(), r_cd.copy()

    return best_p, best_r


def _force_directed_anneal(pos, radii, deadline, max_iter=2000, rng_seed=123):
    """Force-directed annealing to explore topological rearrangements.
    Uses repulsive forces between overlapping circles and attractive forces
    toward the center of available space, with simulated annealing cooling."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(rng_seed)
    
    T = 0.08  # initial temperature
    cooling = 0.9995
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T < 1e-6:
            break
            
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces: push circles back inside
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 20.0
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 20.0
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 20.0
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 20.0
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    # Overlap: strong repulsion
                    overlap = min_dist - dist
                    f = overlap * 80.0 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                elif dist > min_dist and dist < min_dist * 1.3:
                    # Near-contact: mild attraction to maximize packing
                    f = (min_dist * 1.3 - dist) * 0.3 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
        
        # Apply forces with temperature-based noise (position-adaptive step)
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            # Smaller steps for larger circles (they have less room)
            local_step = 0.012 * T / (1.0 + radii[i] * 4.0)
            if f_norm > 1e-12:
                delta = local_step * (forces[i] / f_norm + rng.randn(2) * 0.5)
            else:
                delta = local_step * 0.4 * rng.randn(2)
            pos[i] += delta
            # Clamp to unit square with margin
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
        
        # Try to grow radii slightly if no overlaps
        pos_pp, radii_pp, is_valid_pp = repair_and_validate(pos, radii, deadline)
        if is_valid_pp:
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
            # Grow radii adaptively: more for small circles, less for large
            for i in range(N):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                growth = 1.002 if radii[i] < 0.08 else 1.001
                radii[i] = min(radii[i] * growth, max_r * 0.999)
        
        T *= cooling
    
    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp, is_valid_pp = repair_and_validate(pos, radii, deadline)
        if is_valid_pp:
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # ---------- Track A: compact structural parameterization ----------
    # DE over 10-dim structural space, then expand + polish.
    struct_bounds = [
        (0.06, 0.28),   # corner_r
        (0.06, 0.32),   # center_r
        (0.3, 0.7),     # cx
        (0.3, 0.7),     # cy
        (0.18, 0.45),   # ring1 offset
        (0.04, 0.16),   # ring1 r
        (0.04, 0.16),   # ring1 r2
        (0.30, 0.55),   # ring2 offset
        (0.03, 0.12),   # ring2 r
        (0.03, 0.12),   # ring2 r2
    ]

    if time.time() < deadline - 55:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=100, popsize=18, seed=7,
                                            tol=1e-8, polish=False,
                                            mutation=(0.4, 1.0), recombination=0.7,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                # polish the DE structural result in full space
                if time.time() < deadline - 25:
                    pos_p, radii_p = _polish(pos_s, radii_s, deadline)
                    if _is_valid(pos_p, radii_p):
                        s = np.sum(radii_p)
                        if s > best_score:
                            best_score = s
                            best_pos = pos_p.copy()
                            best_radii = radii_p.copy()
        except Exception:
            pass

    # L-BFGS-B on structural vector from a few good seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 35:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B',
                           bounds=struct_bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: direct 78-dim from structural seeds (safety net) ----------
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    # structural seed expanded
    pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
    configs.append((pos_s0, radii_s0))

    for pos, radii in configs:
        if time.time() > deadline - 25:
            break
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-11})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Track C: Force-directed annealing on best configuration ----------
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline,
                                                       max_iter=1500, rng_seed=456)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    # ---------- Track D: SLSQP Polish on Best ----------
    if best_pos is not None and time.time() < deadline - 10:
        try:
            x_sl = np.concatenate([best_pos.flatten(), best_radii])
            bounds_sl = []
            for i in range(N):
                bounds_sl.append((best_radii[i] * 0.5, best_radii[i] * 1.5))
                bounds_sl.append((best_radii[i] * 0.5, best_radii[i] * 1.5))
            for i in range(N):
                bounds_sl.append((best_radii[i] * 0.9, best_radii[i] * 1.1 + 0.01))
            
            res_sl = minimize(lambda x: _objective(x, deadline), x_sl, method='SLSQP',
                              bounds=bounds_sl, options={'maxiter': 1000, 'ftol': 1e-12})
            
            pos_sl = res_sl.x[:2 * N].reshape(N, 2)
            radii_sl = res_sl.x[2 * N:]
            
            pos_pp, radii_pp, is_valid_pp = repair_and_validate(pos_sl, radii_sl, deadline)
            if is_valid_pp:
                s = np.sum(radii_pp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # ---------- Final polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 8:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
        
        # Additional SLSQP pass if time permits
        if time.time() < deadline - 5:
            try:
                x_sl2 = np.concatenate([best_pos.flatten(), best_radii])
                bounds_sl2 = []
                for i in range(N):
                    bounds_sl2.append((best_radii[i] * 0.8, best_radii[i] * 1.2))
                    bounds_sl2.append((best_radii[i] * 0.8, best_radii[i] * 1.2))
                for i in range(N):
                    bounds_sl2.append((best_radii[i] * 0.95, best_radii[i] * 1.05 + 0.005))
                
                res_sl2 = minimize(lambda x: _objective(x, deadline), x_sl2, method='SLSQP',
                                   bounds=bounds_sl2, options={'maxiter': 500, 'ftol': 1e-14})
                
                pos_sl2 = res_sl2.x[:2 * N].reshape(N, 2)
                radii_sl2 = res_sl2.x[2 * N:]
                
                pos_pp2, radii_pp2, is_valid_pp2 = repair_and_validate(pos_sl2, radii_sl2, deadline)
                if is_valid_pp2:
                    s = np.sum(radii_pp2)
                    if s > best_score:
                        best_score = s
                        best_pos = pos_pp2.copy()
                        best_radii = radii_pp2.copy()
            except Exception:
                pass
        # Extra micro-polish: aggressive coordinate descent
        if time.time() < deadline - 1:
            p_mp = best_pos.copy()
            r_mp = best_radii.copy()
            for _ in range(10):
                improved = False
                for i in range(N):
                    if time.time() > deadline - 0.3:
                        break
                    # Try growing radius i
                    for growth in [1.005, 1.002, 1.001, 1.0005, 1.0001]:
                        new_r = r_mp[i] * growth
                        max_r = min(p_mp[i, 0], 1.0 - p_mp[i, 0],
                                    p_mp[i, 1], 1.0 - p_mp[i, 1])
                        if new_r > max_r:
                            new_r = max_r * 0.9999
                        old_r = r_mp[i]
                        r_mp[i] = new_r
                        if _is_valid(p_mp, r_mp):
                            improved = True
                            break
                        else:
                            r_mp[i] = old_r
                    # Try nudging position to create room
                    for dx in [0.002, -0.002, 0.001, -0.001]:
                        for dy in [0.002, -0.002, 0.001, -0.001]:
                            ox, oy = p_mp[i, 0], p_mp[i, 1]
                            p_mp[i, 0] = ox + dx
                            p_mp[i, 1] = oy + dy
                            if p_mp[i, 0] < r_mp[i] or p_mp[i, 0] > 1 - r_mp[i]:
                                p_mp[i, 0] = ox
                                continue
                            if p_mp[i, 1] < r_mp[i] or p_mp[i, 1] > 1 - r_mp[i]:
                                p_mp[i, 1] = oy
                                continue
                            if _is_valid(p_mp, r_mp):
                                # Try to grow all circles
                                any_grown = False
                                for j in range(N):
                                    max_rj = min(p_mp[j, 0], 1.0 - p_mp[j, 0],
                                                 p_mp[j, 1], 1.0 - p_mp[j, 1])
                                    if r_mp[j] < max_rj * 0.9999:
                                        r_mp[j] = max_rj * 0.9999
                                        any_grown = True
                                improved = True
                                break
                            else:
                                p_mp[i, 0] = ox
                                p_mp[i, 1] = oy
                        if improved:
                            break
                if not improved:
                    break
            if _is_valid(p_mp, r_mp):
                s = np.sum(r_mp)
                if s > best_score:
                    best_score = s
                    best_pos = p_mp.copy()
                    best_radii = r_mp.copy()

    # Final validation and repair of the best candidate
    if best_pos is not None:
        final_pos, final_radii, is_final_valid = repair_and_validate(best_pos, best_radii, deadline)
        if is_final_valid:
            best_pos = final_pos
            best_radii = final_radii
        else:
            # If even the best candidate is invalid after repair, use a safe fallback
            pos, radii = _make_grid(N)
            final_pos, final_radii, _ = repair_and_validate(pos, radii, deadline)
            best_pos = final_pos
            best_radii = final_radii
    else:
        pos, radii = _make_grid(N)
        best_pos, best_radii, _ = repair_and_validate(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
