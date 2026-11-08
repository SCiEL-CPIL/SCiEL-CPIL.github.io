"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _make_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _feasible_grow(pos, radii, grow, pairs):
    """Check if we can grow all radii by 'grow' amount."""
    n = len(pos)
    new_radii = radii + grow
    for i in range(n):
        if new_radii[i] > 0.5:
            return False, new_radii
        if pos[i, 0] < new_radii[i] - 1e-9 or pos[i, 0] > 1 - new_radii[i] + 1e-9:
            return False, new_radii
        if pos[i, 1] < new_radii[i] - 1e-9 or pos[i, 1] > 1 - new_radii[i] + 1e-9:
            return False, new_radii
    for i, j in pairs:
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < new_radii[i] + new_radii[j] - 1e-9:
            return False, new_radii
    return True, new_radii


def _clamp_boundaries(pos, radii):
    for i in range(len(pos)):
        pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
        pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))


def _relax(pos, radii, pairs, max_iters=2000, time_budget=30.0, start_time=None):
    """Iterative overlap resolution with boundary clamping."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        _clamp_boundaries(pos, radii)
        
        changed = False
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                changed = True
                if dist < 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = (min_dist - dist) / 2
                    pos[i, 0] -= push * np.cos(angle)
                    pos[i, 1] -= push * np.sin(angle)
                    pos[j, 0] += push * np.cos(angle)
                    pos[j, 1] += push * np.sin(angle)
                else:
                    push = (min_dist - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    pos[i, 0] -= push * nx
                    pos[i, 1] -= push * ny
                    pos[j, 0] += push * nx
                    pos[j, 1] += push * ny
        
        _clamp_boundaries(pos, radii)
        
        if not changed:
            break
    
    return pos, radii


def _grow_radii(pos, radii, pairs, time_budget=5.0, start_time=None):
    """Try to grow radii as much as possible."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    grow = 0.002
    max_grow_attempts = 500
    
    for _ in range(max_grow_attempts):
        if time.time() - start_time > time_budget:
            break
        
        feasible, new_radii = _feasible_grow(pos, radii, grow, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=200,
                                  time_budget=min(1.0, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
        else:
            lo, hi = 0.0, grow
            for _ in range(20):
                mid = (lo + hi) / 2
                feasible_mid, _ = _feasible_grow(pos, radii, mid, pairs)
                if feasible_mid:
                    lo = mid
                else:
                    hi = mid
            if lo > 1e-6:
                _, new_radii = _feasible_grow(pos, radii, lo, pairs)
                radii = new_radii
                pos, radii = _relax(pos, radii, pairs, max_iters=200,
                                      time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                      start_time=start_time)
            grow *= 0.7
            if grow < 1e-6:
                break
    
    return pos, radii


def _local_search(pos, radii, pairs, time_budget=20.0, start_time=None):
    """Local search with random moves and growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    while time.time() - start_time < time_budget:
        feasible, new_radii = _feasible_grow(pos, radii, 0.001, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=100,
                                  time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(n)
            old_pos = pos[idx].copy()
            step_size = 0.01
            pos[idx] += rng.normal(0, step_size, size=2)
            pos[idx] = np.clip(pos[idx], radii[idx], 1 - radii[idx])
            
            valid = True
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            if valid:
                for j in range(n):
                    if j == idx:
                        continue
                    dx = pos[j, 0] - pos[idx, 0]
                    dy = pos[j, 1] - pos[idx, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[idx] + radii[j] - 1e-9:
                        valid = False
                        break
            
            if valid:
                feasible2, new_r = _feasible_grow(pos, radii, 0.0005, pairs)
                if feasible2:
                    radii = new_r
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
            else:
                pos[idx] = old_pos
    
    return best_pos, best_radii


def _optimize_min_gap(pos, radii, pairs, time_budget=10.0, start_time=None):
    """
    Structural Change: Rigid-body contact graph optimization.
    Maximizes the soft-minimum gap between circles.
    This drives the system to a 'fully stressed' configuration where many
    circles are in contact, allowing for denser packing.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    # Pre-compute indices for vectorization
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_count = len(i_idx)
    
    # We will optimize positions only, keeping radii fixed for the "gap" phase.
    # Then we will grow radii.
    
    def soft_min_gap(flat_pos, beta=100.0):
        """
        Returns (negative_soft_min_gap, grad).
        We maximize min_gap, so we minimize -min_gap.
        min_gap = min(d_ij - r_i - r_j, boundary_gaps)
        """
        pos_arr = flat_pos.reshape(-1, 2)
        
        # Pairwise gaps
        dx = pos_arr[i_idx, 0] - pos_arr[j_idx, 0]
        dy = pos_arr[i_idx, 1] - pos_arr[j_idx, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        gaps = dist - radii[i_idx] - radii[j_idx]
        
        # Boundary gaps (distance to wall - radius)
        # x >= r -> gap = x - r
        # x <= 1-r -> gap = (1-r) - x
        # y >= r -> gap = y - r
        # y <= 1-r -> gap = (1-r) - y
        b_gaps = np.concatenate([
            pos_arr[:, 0] - radii,
            (1 - radii) - pos_arr[:, 0],
            pos_arr[:, 1] - radii,
            (1 - radii) - pos_arr[:, 1]
        ])
        
        all_gaps = np.concatenate([gaps, b_gaps])
        
        # Soft min: -log-sum-exp(-beta * gaps) / beta
        # To avoid overflow, subtract max
        max_gap = np.max(all_gaps)
        # If max_gap is very large, soft min is close to max_gap? No.
        # Soft min approximates min.
        # min(x) = -1/beta * log(sum(exp(-beta * x)))
        
        # We want to maximize min_gap.
        # Objective: -min_gap.
        # Let's compute S = sum(exp(-beta * g)).
        # min_g = -1/beta * log(S).
        # We minimize -min_g = 1/beta * log(S).
        
        # Gradient of -min_g w.r.t. pos:
        # d(-min_g)/d(pos_k) = 1/beta * (1/S) * dS/d(pos_k)
        # dS/d(pos_k) = sum_j exp(-beta * g_j) * (-beta) * d(g_j)/d(pos_k)
        
        # Let's compute weights w_j = exp(-beta * g_j) / S.
        # Then d(-min_g)/d(pos_k) = - sum_j w_j * d(g_j)/d(pos_k)
        
        # Note: If gaps are large, exp(-beta*g) is small.
        # If gaps are negative (overlap), exp(-beta*g) is large.
        # This effectively focuses on the tightest constraints (smallest gaps).
        
        # Stability: shift by max_gap to prevent underflow/overflow?
        # Actually, if we shift, the weights change.
        # Standard softmin: m = -1/beta * log(sum(exp(-beta*g)))
        # Let g_max = max(g).
        # m = -1/beta * log(sum(exp(-beta*(g - g_max)))) - g_max
        # This is numerically stable.
        
        g_shifted = all_gaps - max_gap
        exp_terms = np.exp(-beta * g_shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
             return 0.0, np.zeros_like(flat_pos)
        
        weights = exp_terms / sum_exp
        
        # Objective value: -m = 1/beta * log(sum_exp) + max_gap
        obj_val = 1.0/beta * np.log(sum_exp) + max_gap
        
        # Gradients
        grad_flat = np.zeros_like(flat_pos)
        
        # Pairwise gradients
        # d(g_ij)/d(pos_i) = - (pos_i - pos_j) / dist = (pos_j - pos_i) / dist
        # d(g_ij)/d(pos_j) = (pos_i - pos_j) / dist
        
        dx_vec = pos_arr[i_idx] - pos_arr[j_idx]
        dist_vec = np.sqrt(np.sum(dx_vec**2, axis=1))
        # Handle dist=0
        dist_vec[dist_vec < 1e-12] = 1e-12
        unit_vec = dx_vec / dist_vec[:, None]
        
        # weights for pairs
        w_pairs = weights[:pair_count]
        
        # grad for i: - sum w_ij * (pos_j - pos_i)/dist = - sum w_ij * (-unit_vec) = sum w_ij * unit_vec
        # Wait. g_ij = dist - r_i - r_j.
        # d(dist)/d(pos_i) = (pos_i - pos_j)/dist = -unit_vec (if unit_vec = (pos_j-pos_i)/dist)
        # Let's define u_ij = (pos_j - pos_i) / dist.
        # d(dist)/d(pos_i) = -u_ij.
        # d(g_ij)/d(pos_i) = -u_ij.
        # d(-min_g)/d(pos_i) = - sum_j w_ij * d(g_ij)/d(pos_i) = - sum_j w_ij * (-u_ij) = sum_j w_ij * u_ij.
        
        # So grad_pos_i += sum_j w_ij * u_ij
        # grad_pos_j += sum_j w_ij * (-u_ij)
        
        # We need to accumulate these.
        # np.add.at is slow but robust.
        # Alternative: use sparse or loop if n is small (26 is small, pairs ~325).
        
        # Since n=26, loop is fine and faster than add.at overhead potentially.
        # But vectorized add.at is okay.
        
        # grad_flat[2*i_idx] += w_pairs * u_ij[:, 0]
        # grad_flat[2*i_idx+1] += w_pairs * u_ij[:, 1]
        # grad_flat[2*j_idx] -= w_pairs * u_ij[:, 0]
        # grad_flat[2*j_idx+1] -= w_pairs * u_ij[:, 1]
        
        np.add.at(grad_flat, 2 * i_idx, w_pairs * u_vec[:, 0])
        np.add.at(grad_flat, 2 * i_idx + 1, w_pairs * u_vec[:, 1])
        np.add.at(grad_flat, 2 * j_idx, -w_pairs * u_vec[:, 0])
        np.add.at(grad_flat, 2 * j_idx + 1, -w_pairs * u_vec[:, 1])
        
        # Boundary gradients
        # b_gaps indices:
        # 0..n-1: x - r -> d/dx = 1
        # n..2n-1: (1-r) - x -> d/dx = -1
        # 2n..3n-1: y - r -> d/dy = 1
        # 3n..4n-1: (1-r) - y -> d/dy = -1
        
        w_b = weights[pair_count:]
        w_x_low = w_b[:n]
        w_x_high = w_b[n:2*n]
        w_y_low = w_b[2*n:3*n]
        w_y_high = w_b[3*n:4*n]
        
        # d(-min_g)/d(x_i) = - (w_x_low[i] * 1 + w_x_high[i] * (-1)) = -w_x_low[i] + w_x_high[i]
        # d(-min_g)/d(y_i) = - (w_y_low[i] * 1 + w_y_high[i] * (-1)) = -w_y_low[i] + w_y_high[i]
        
        grad_flat[0::2] += (-w_x_low + w_x_high)
        grad_flat[1::2] += (-w_y_low + w_y_high)
        
        return obj_val, grad_flat
    
    # Optimization loop
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    current_pos_flat = pos.flatten()
    
    # We iterate: Optimize gaps -> Grow Radii -> Repeat
    max_iters = 50
    beta = 50.0
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
            
        # 1. Optimize positions to maximize min gap
        try:
            res = minimize(soft_min_gap, current_pos_flat, method='L-BFGS-B',
                           jac=True, options={'maxiter': 20, 'maxls': 10})
        except Exception:
            break
            
        new_pos = res.x.reshape(-1, 2)
        
        # Clamp to valid range (just in case)
        for i in range(n):
            new_pos[i, 0] = max(radii[i], min(1 - radii[i], new_pos[i, 0]))
            new_pos[i, 1] = max(radii[i], min(1 - radii[i], new_pos[i, 1]))
            
        # Check validity
        if _check_validity(new_pos, radii):
            pos = new_pos
            current_pos_flat = pos.flatten()
        else:
            # If invalid, revert and stop growing
            break
            
        # 2. Try to grow radii
        # Calculate current min gap
        dx = pos[i_idx, 0] - pos[j_idx, 0]
        dy = pos[i_idx, 1] - pos[j_idx, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        gaps = dist - radii[i_idx] - radii[j_idx]
        b_gaps = np.concatenate([
            pos[:, 0] - radii,
            (1 - radii) - pos[:, 0],
            pos[:, 1] - radii,
            (1 - radii) - pos[:, 1]
        ])
        min_gap = np.min(np.concatenate([gaps, b_gaps]))
        
        if min_gap < 1e-4:
            # No room to grow
            break
            
        # Grow by half the min gap (conservative)
        grow = min_gap * 0.5
        # Cap growth to avoid huge jumps
        grow = min(grow, 0.01)
        
        new_radii = radii + grow
        
        # Check if new radii are feasible with current positions
        # (Quick check: just boundaries and overlaps)
        feasible = True
        if np.any(new_radii > 0.5):
            feasible = False
        elif np.any(pos[:, 0] < new_radii - 1e-9) or np.any(pos[:, 0] > 1 - new_radii + 1e-9):
            feasible = False
        elif np.any(pos[:, 1] < new_radii - 1e-9) or np.any(pos[:, 1] > 1 - new_radii + 1e-9):
            feasible = False
        else:
            # Check overlaps
            dx = pos[i_idx, 0] - pos[j_idx, 0]
            dy = pos[i_idx, 1] - pos[j_idx, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if np.any(dist < new_radii[i_idx] + new_radii[j_idx] - 1e-9):
                feasible = False
        
        if feasible:
            radii = new_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # Try smaller grow
            grow *= 0.5
            if grow < 1e-5:
                break
            new_radii = radii + grow
            # Re-check quickly
            if np.any(dist < new_radii[i_idx] + new_radii[j_idx] - 1e-9):
                 break
            radii = new_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Increase beta to sharpen min
        beta *= 1.2
        if beta > 500:
            beta = 500
            
    return best_pos, best_radii


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def _build_corner(n):
    """Place circles at corners and edges for a structured start."""
    pos_list = []
    r_corner = 0.12
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])
    
    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])
    
    remaining = n - len(pos_list)
    if remaining > 0:
        side = int(np.ceil(np.sqrt(remaining)))
        for j in range(side):
            for i in range(side):
                if len(pos_list) >= n:
                    break
                x = 0.25 + (i + 0.5) * 0.5 / side
                y = 0.25 + (j + 0.5) * 0.5 / side
                pos_list.append([x, y])
    
    pos = np.array(pos_list[:n])
    radii = np.full(n, 0.08)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    pairs = _make_pairs(N)
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    for seed in [42, 123, 456, 789, 1000]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    # Process each configuration
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        remaining = deadline - elapsed - 5
        
        # Phase 1: Relax overlaps
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _relax(pos, radii, pairs, max_iters=2000,
                                 time_budget=time_alloc, start_time=start_time)
        
        # Phase 2: Grow radii
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _grow_radii(pos, radii, pairs, time_budget=time_alloc,
                                      start_time=start_time)
        
        # Phase 3: Rigid-body contact graph optimization (Min-Gap Max)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(20.0, remaining * 0.6)
        if time_alloc > 2:
            pos, radii = _optimize_min_gap(pos, radii, pairs,
                                            time_budget=time_alloc,
                                            start_time=start_time)
            _clamp_boundaries(pos, radii)
        
        # Phase 4: Local search polish (keep for final adjustments)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _local_search(pos, radii, pairs, time_budget=time_alloc,
                                        start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final polish if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        if remaining > 5:
            # Try min-gap optimization on best
            backup_pos = best_pos.copy()
            backup_radii = best_radii.copy()
            pos2, radii2 = _optimize_min_gap(best_pos, best_radii, pairs,
                                            time_budget=min(10.0, remaining * 0.6),
                                            start_time=start_time)
            _clamp_boundaries(pos2, radii2)
            if _check_validity(pos2, radii2):
                new_score = np.sum(radii2)
                if new_score > best_score:
                    best_pos = pos2
                    best_radii = radii2
                    best_score = new_score
            
            # Try local search on best
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 3
            if remaining > 3:
                pos3, radii3 = _local_search(best_pos, best_radii, pairs,
                                              time_budget=min(5.0, remaining),
                                              start_time=start_time)
                if _check_validity(pos3, radii3):
                    new_score = np.sum(radii3)
                    if new_score > best_score:
                        best_pos = pos3
                        best_radii = radii3
                        best_score = new_score
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
