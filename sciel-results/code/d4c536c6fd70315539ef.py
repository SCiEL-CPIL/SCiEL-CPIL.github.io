"""Physics-based soft-body simulation with Lennard-Jones potentials for circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _lj_energy(pos, radii, W, H, sigma_scale=1.0):
    """Lennard-Jones-like energy with wall constraints."""
    n = len(radii)
    energy = 0.0
    
    # Inter-circle: LJ-like repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            min_dist = radii[i] + radii[j]
            if dist < min_dist * 1.5:
                # LJ-like: strong repulsion when close
                ratio = min_dist / dist
                energy += 100.0 * (ratio ** 12 - ratio ** 6)
    
    # Wall repulsion
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        if x < r * 1.5:
            energy += 50.0 * ((r - x) / (r * 0.5)) ** 2
        if x > W - r * 1.5:
            energy += 50.0 * ((x - (W - r)) / (r * 0.5)) ** 2
        if y < r * 1.5:
            energy += 50.0 * ((r - y) / (r * 0.5)) ** 2
        if y > H - r * 1.5:
            energy += 50.0 * ((y - (H - r)) / (r * 0.5)) ** 2
    
    # Encourage larger radii
    energy -= 5.0 * np.sum(radii)
    
    return energy


def _simulate(W, H, n_restarts=5, n_steps=2000, seed=42):
    """Run physics simulation with multiple restarts."""
    rng = np.random.RandomState(seed)
    best_sum = 0.0
    best_result = None
    
    for restart in range(n_restarts):
        # Initialize: grid with jitter
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) * W / side + rng.uniform(-0.02, 0.02)
            pos[i, 1] = (row + 0.5) * H / side + rng.uniform(-0.02, 0.02)
        
        # Clamp to valid range
        pos[:, 0] = np.clip(pos[:, 0], 0.05, W - 0.05)
        pos[:, 1] = np.clip(pos[:, 1], 0.05, H - 0.05)
        
        r_max = min(W, H) / (2 * side)
        radii = np.full(N, r_max * 0.3)
        vel = np.zeros_like(pos)
        
        # Physics simulation with cooling
        dt = 0.005
        temp = 0.1
        cooling_rate = 0.9995
        
        for step in range(n_steps):
            # Compute forces
            force = np.zeros_like(pos)
            
            # Inter-circle repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist * 1.5:
                        ratio = min_dist / dist
                        # LJ-like force
                        f_mag = 100.0 * (12 * ratio ** 13 - 6 * ratio ** 7) / dist
                        force[i, 0] += f_mag * dx
                        force[i, 1] += f_mag * dy
                        force[j, 0] -= f_mag * dx
                        force[j, 1] -= f_mag * dy
            
            # Wall forces
            for i in range(N):
                r = radii[i]
                x, y = pos[i]
                if x < r * 1.5:
                    force[i, 0] += 50.0 * (r - x) / (r * 0.25)
                if x > W - r * 1.5:
                    force[i, 0] -= 50.0 * (x - (W - r)) / (r * 0.25)
                if y < r * 1.5:
                    force[i, 1] += 50.0 * (r - y) / (r * 0.25)
                if y > H - r * 1.5:
                    force[i, 1] -= 50.0 * (y - (H - r)) / (r * 0.25)
            
            # Update velocities with damping
            vel = 0.9 * vel + force * dt
            pos += vel * dt
            
            # Clamp positions
            pos[:, 0] = np.clip(pos[:, 0], 0.01, W - 0.01)
            pos[:, 1] = np.clip(pos[:, 1], 0.01, H - 0.01)
            
            # Gradually increase radii
            if step > n_steps // 4:
                growth = 0.0001 * (1.0 - step / n_steps)
                max_r = min(W, H) / 2.0
                radii = np.minimum(radii + growth, max_r)
            
            # Cool
            temp *= cooling_rate
        
        # Final: use scipy to polish
        x0 = np.concatenate([pos.flatten(), radii])
        
        def objective(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            e = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(p[i] - p[j]) + 1e-12
                    md = r[i] + r[j]
                    if d < md:
                        e += 1000.0 * (md - d) ** 2
            for i in range(N):
                r = r[i]
                if p[i, 0] < r:
                    e += 1000.0 * (r - p[i, 0]) ** 2
                if p[i, 0] > W - r:
                    e += 1000.0 * (p[i, 0] - (W - r)) ** 2
                if p[i, 1] < r:
                    e += 1000.0 * (r - p[i, 1]) ** 2
                if p[i, 1] > H - r:
                    e += 1000.0 * (p[i, 1] - (H - r)) ** 2
            e -= 10.0 * np.sum(r)
            return e
        
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300})
        
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = np.maximum(result.x[2 * N:], 1e-6)
        
        # Validate
        valid = True
        for i in range(N):
            if pos_opt[i, 0] < radii_opt[i] - 1e-9 or pos_opt[i, 0] > W - radii_opt[i] + 1e-9:
                valid = False
                break
            if pos_opt[i, 1] < radii_opt[i] - 1e-9 or pos_opt[i, 1] > H - radii_opt[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                    if d < radii_opt[i] + radii_opt[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            total = np.sum(radii_opt)
            if total > best_sum:
                best_sum = total
                best_result = (pos_opt.copy(), radii_opt.copy())
    
    return best_result, best_sum


def construct():
    best_overall = None
    best_total = 0.0
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.5]
    
    for aspect in aspect_ratios:
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        result, total = _simulate(W, H, n_restarts=3, n_steps=1000, seed=42)
        
        if result is not None and total > best_total:
            best_total = total
            best_overall = (result[0], result[1], W, H)
    
    if best_overall is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells)
    
    pos_opt, radii_opt, W, H = best_overall
    
    # Final safety
    for i in range(N):
        radii_opt[i] = min(radii_opt[i], pos_opt[i, 0], W - pos_opt[i, 0], pos_opt[i, 1], H - pos_opt[i, 1])
        radii_opt[i] = max(radii_opt[i], 1e-6)
    
    # Resolve overlaps
    for _ in range(50):
        resolved = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                md = radii_opt[i] + radii_opt[j]
                if d < md - 1e-10:
                    resolved = False
                    shrink = (md - d) / 2.0 + 1e-4
                    radii_opt[i] = max(radii_opt[i] - shrink, 1e-6)
                    radii_opt[j] = max(radii_opt[j] - shrink, 1e-6)
        if resolved:
            break
    
    for i in range(N):
        radii_opt[i] = min(radii_opt[i], pos_opt[i, 0], W - pos_opt[i, 0], pos_opt[i, 1], H - pos_opt[i, 1])
        radii_opt[i] = max(radii_opt[i], 1e-6)
    
    return np.column_stack([pos_opt, radii_opt])
# EVOLVE-BLOCK-END
