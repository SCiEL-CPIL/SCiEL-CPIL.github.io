"""Circle packing in perimeter-4 rectangle: Growth-and-Relax Strategy."""
import numpy as np
import time
from scipy.spatial import Delaunay

N = 21
_TIME_LIMIT = 142.0  # Safety margin before 150s kill


def _push_apart(z, max_iter=2000, tol=1e-7, time_limit=1.0):
    """
    Differentiable gradient-flow push-apart solver.
    z: [x1, y1, r1, ..., xN, yN, rN, w]
    """
    start = time.time()
    n = N
    z = z.copy()
    dt = 0.005
    min_dt = 1e-5
    
    for _ in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        if w < 0.1 or h < 0.1:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Vectorized pairwise forces
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx*dx + dy*dy) + 1e-10
        min_d = r[:, None] + r[None, :]
        
        # Only consider overlapping pairs (upper triangle)
        mask = (d < min_d) & (np.triu(np.ones((n, n), dtype=bool), k=1))
        
        # Force magnitude: proportional to overlap depth / distance
        # This approximates the gradient of a smooth convex potential
        force_mag = np.where(mask, (min_d - d) / d, 0.0)
        
        fx += np.sum(force_mag * dx, axis=1)
        fy += np.sum(force_mag * dy, axis=1)
        
        # Boundary forces: push circles back inside if they violate bounds
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        # Strong boundary penalty to keep circles inside
        fx -= 20.0 * left_viol
        fx += 20.0 * right_viol
        fy -= 20.0 * bottom_viol
        fy += 20.0 * top_viol
        
        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        
        # Clamp to bounds (hard constraint)
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)
        
        # Adaptive step size
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)))
        if max_force < tol:
            break
        elif max_force > 5.0:
            dt = max(min_dt, dt * 0.5)
        else:
            dt = min(0.01, dt * 1.05)
            
    return z


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _grow_and_relax(w0, max_time=5.0):
    """
    Strategy: Start with small circles, grow them uniformly, and relax positions.
    Returns (z, value) where z is the final state and value is sum(r).
    """
    start = time.time()
    n = N
    h0 = 2.0 - w0
    
    # Initial guess: Grid layout with small radii
    side = int(np.ceil(np.sqrt(N)))
    z = np.zeros(3 * N + 1)
    for i in range(N):
        row, col = divmod(i, side)
        cx = (col + 0.5) / side * w0
        cy = (row + 0.5) / side * h0
        r0 = 0.05
        z[3 * i] = cx
        z[3 * i + 1] = cy
        z[3 * i + 2] = r0
    z[-1] = w0
    
    # Relax initial configuration
    z = _push_apart(z, max_iter=500, time_limit=1.0)
    
    # Binary search on global radius scale
    # We want to find the maximum uniform scaling factor s such that
    # s * r_i fits in the box.
    # Actually, it's better to just grow radii and let push_apart handle it.
    # Let's try a "growth" loop: increase all radii by a small amount, then push apart.
    
    step = 0.005
    current_r = z[2::3][:N].copy()
    
    # Estimate max possible radius based on area
    area = w0 * h0
    max_r_area = np.sqrt(area / N) * 0.8
    min_r = 0.05
    
    # If current radii are already large, we are in a good spot.
    # Let's try to expand.
    
    for _ in range(100):
        if time.time() - start > max_time:
            break
        
        # Try to expand all radii
        new_r = current_r + step
        # Limit by box dimensions
        new_r = np.minimum(new_r, w0 / 2)
        new_r = np.minimum(new_r, h0 / 2)
        
        z[2::3][:N] = new_r
        
        # Relax
        z = _push_apart(z, max_iter=200, time_limit=0.5)
        
        # Check validity
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if _is_valid(x, y, r, w, h):
            current_r = r.copy()
            # If we could expand and stay valid, keep going
            # But maybe we should check if we can expand more?
            # The push_apart might have shrunk them? No, push_apart doesn't shrink radii.
            # It only moves x, y.
            # So if valid, we can try again.
        else:
            # Invalid. Try smaller step or break.
            step *= 0.8
            if step < 1e-5:
                break
            # Revert to previous valid state
            z[2::3][:N] = current_r
            z = _push_apart(z, max_iter=100, time_limit=0.2)
            break
            
    return z, np.sum(z[2::3][:N])


def _local_aspect_search(z_best, time_limit=10.0):
    """
    Fine-tune the aspect ratio w around the current best solution.
    """
    start = time.time()
    best_z = z_best.copy()
    best_val = np.sum(z_best[2::3][:N])
    w0 = z_best[-1]
    
    # Try small perturbations in w
    deltas = [0.05, -0.05, 0.02, -0.02, 0.01, -0.01]
    
    for d in deltas:
        if time.time() - start > time_limit:
            break
        
        w_new = w0 + d
        if w_new < 0.3 or w_new > 1.7:
            continue
        
        h_new = 2.0 - w_new
        
        # Scale positions and radii to fit new box?
        # Or just adjust w and let push_apart handle it?
        # Adjusting w changes the constraint.
        # Let's try to map the current solution to the new box aspect ratio.
        # Simple approach: Keep x, y, r as is, but change w.
        # This might cause boundary violations.
        
        z_trial = best_z.copy()
        z_trial[-1] = w_new
        
        # Relax in the new box
        z_trial = _push_apart(z_trial, max_iter=500, time_limit=1.0)
        
        # Try to grow again in the new box
        z_trial, val = _grow_and_relax(w_new, max_time=2.0)
        
        # Check validity
        x = z_trial[0::3][:N]
        y = z_trial[1::3][:N]
        r = z_trial[2::3][:N]
        w = z_trial[-1]
        h = 2.0 - w
        
        if _is_valid(x, y, r, w, h) and val > best_val:
            best_val = val
            best_z = z_trial.copy()
            
    return best_z, best_val


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    # Perimeter 4 => w + h = 2.
    # w in [0.5, 1.5]
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5]
    
    # Sort by closeness to 1.0 (square is often good for packing)
    aspect_ratios.sort(key=lambda x: abs(x - 1.0))
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        
        # Run growth strategy
        try:
            z, val = _grow_and_relax(w0, max_time=5.0)
            
            # Verify
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            
            if _is_valid(x, y, r, w, h) and val > best_value:
                best_value = val
                best_result = z.copy()
                
        except Exception:
            continue

    # Local aspect search on the best result found
    if best_result is not None and time.time() < deadline - 10:
        z_refined, val_refined = _local_aspect_search(best_result, time_limit=15.0)
        if val_refined > best_value:
            best_value = val_refined
            best_result = z_refined

    # Final polish: try to expand individual circles
    if best_result is not None and time.time() < deadline - 5:
        z = best_result.copy()
        n = N
        # Simple individual expansion
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            rng = np.random.default_rng()
            order = rng.permutation(n)
            for i in order:
                r_i = z[2*i+2]
                x_i = z[2*i]
                y_i = z[2*i+1]
                w = z[-1]
                h = 2.0 - w
                
                # Try to increase radius
                new_r_i = r_i + 0.001
                if new_r_i > w/2 or new_r_i > h/2:
                    continue
                
                # Check bounds
                if new_r_i > x_i or x_i + new_r_i > w or new_r_i > y_i or y_i + new_r_i > h:
                    continue
                
                # Check overlaps
                dx = x_i - z[0::3][:n]
                dy = y_i - z[1::3][:n]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = new_r_i + z[2::3][:n]
                min_d[i] = np.inf
                
                if np.any(d < min_d - 1e-9):
                    continue
                
                # Valid expansion
                z[2*i+2] = new_r_i
                improved = True
                break
                
            if not improved:
                break
                
        # Verify final
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_result = z

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    return np.column_stack((x, y, r))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(np.sum(res[:, 2]))
