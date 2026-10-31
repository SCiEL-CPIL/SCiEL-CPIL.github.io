import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
W = 1.0
H = 1.0

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    best_result = None
    best_score = -np.inf

    def _is_valid(circles, tol=1e-9):
        if len(circles) != N:
            return False
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs <= 0):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > W + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > H + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                rad_sum_sq = (rs[i] + rs[j])**2
                if dist_sq < rad_sum_sq - tol:
                    return False
        return True

    def evaluate_circles(circles):
        if not _is_valid(circles):
            return -np.inf
        return np.sum(circles[:, 2])

    def _vectorized_objective(params, w=1.0, h=1.0, lam=1000.0, mu=1e-3):
        # params: [x1, y1, r1, ..., xN, yN, rN]
        circles = params.reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Boundary penalties (smooth)
        pen = 0.0
        pen += np.sum(np.maximum(0.0, rs - xs)**2)
        pen += np.sum(np.maximum(0.0, xs - (w - rs))**2)
        pen += np.sum(np.maximum(0.0, rs - ys)**2)
        pen += np.sum(np.maximum(0.0, ys - (h - rs))**2)
        
        # Overlap penalties (smooth)
        # Use vectorized computation for speed
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx**2 + dy**2
        rad_sum = rs[:, None] + rs[None, :]
        overlap = np.maximum(0.0, rad_sum**2 - dist_sq)
        pen += np.sum(overlap**2)
        
        # Maximize sum of radii -> Minimize -sum(rs)
        return -np.sum(rs) + lam * pen

    def _force_relax_unified(params, w, h, steps=50, lr=0.05):
        # Simple force-directed relaxation to remove overlaps and push to boundaries
        circles = params.copy().reshape(N, 3)
        for _ in range(steps):
            forces = np.zeros_like(circles)
            xs = circles[:, 0]
            ys = circles[:, 1]
            rs = circles[:, 2]
            
            # Boundary forces
            forces[:, 0] += np.where(xs < rs, (rs - xs) * 10, 0)
            forces[:, 0] -= np.where(xs > w - rs, (xs - (w - rs)) * 10, 0)
            forces[:, 1] += np.where(ys < rs, (rs - ys) * 10, 0)
            forces[:, 1] -= np.where(ys > h - rs, (ys - (h - rs)) * 10, 0)
            
            # Overlap forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist and dist > 1e-9:
                        f = (min_dist - dist) * 5.0
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
            
            # Update
            circles[:, 0] += lr * forces[:, 0]
            circles[:, 1] += lr * forces[:, 1]
            # Don't update radii in this step, just positions
            
            # Clip to valid range
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
            
        return circles.flatten()

    def _polish_result(params, w, h, max_iters=1000, tol=1e-8):
        # Use L-BFGS-B to polish the configuration
        try:
            res = minimize(_vectorized_objective, params, args=(w, h), method='L-BFGS-B',
                           options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol})
            if res.fun < 1e-4:
                return res.x
        except Exception:
            pass
        return params

    def _expansion_relaxation_polish(params, w, h, max_steps=100):
        # Try to increase radii while maintaining validity
        circles = params.copy().reshape(N, 3)
        for _ in range(max_steps):
            improved = False
            for i in range(N):
                old_r = circles[i, 2]
                # Try increasing radius slightly
                new_r = old_r * 1.01 + 1e-5
                circles[i, 2] = new_r
                
                # Check if still valid
                if _is_valid(circles, tol=1e-9):
                    improved = True
                else:
                    circles[i, 2] = old_r
            
            if not improved:
                break
            
            # Relax positions to accommodate larger radii
            circles = _force_relax_unified(circles.flatten(), w, h, steps=20, lr=0.02)
            
        return circles.flatten()

    def _contact_graph_optimize(params, w, h, max_iters=500):
        # Identify contact graph and solve for exact tangencies
        # This is a heuristic implementation
        circles = params.copy().reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Find contacts
        contacts = []
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j] + 1e-6:
                    contacts.append((i, j))
        
        # Also check wall contacts
        wall_contacts = []
        for i in range(N):
            if abs(xs[i] - rs[i]) < 1e-6: wall_contacts.append((i, 'left'))
            if abs(xs[i] - (w - rs[i])) < 1e-6: wall_contacts.append((i, 'right'))
            if abs(ys[i] - rs[i]) < 1e-6: wall_contacts.append((i, 'bottom'))
            if abs(ys[i] - (h - rs[i])) < 1e-6: wall_contacts.append((i, 'top'))
        
        # For now, just use force relaxation to enforce contacts
        for _ in range(max_iters):
            forces = np.zeros_like(circles)
            
            # Enforce contact constraints
            for i, j in contacts:
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist = np.sqrt(dx**2 + dy**2)
                target = rs[i] + rs[j]
                if dist > 1e-9:
                    f = (target - dist) * 10.0
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
            
            # Boundary forces
            forces[:, 0] += np.where(xs < rs, (rs - xs) * 10, 0)
            forces[:, 0] -= np.where(xs > w - rs, (xs - (w - rs)) * 10, 0)
            forces[:, 1] += np.where(ys < rs, (rs - ys) * 10, 0)
            forces[:, 1] -= np.where(ys > h - rs, (ys - (h - rs)) * 10, 0)
            
            circles[:, 0] += 0.05 * forces[:, 0]
            circles[:, 1] += 0.05 * forces[:, 1]
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
            
        return circles.flatten()

    def init_hexagonal():
        # Hexagonal packing in 1x1 square
        cols = 5
        rows = 5
        xs = []
        ys = []
        rs = []
        r_est = 0.09
        for i in range(cols):
            for j in range(rows):
                x = (i + 0.5) * (W / cols)
                y = (j + 0.5) * (H / rows)
                if j % 2 == 1:
                    x += 0.5 * (W / cols)
                if x < W and y < H:
                    xs.append(x)
                    ys.append(y)
                    rs.append(r_est)
        init = np.array(list(zip(xs[:N], ys[:N], rs[:N])))
        return init.flatten()

    def init_random():
        xs = np.random.uniform(0.15, 0.85, N)
        ys = np.random.uniform(0.15, 0.85, N)
        rs = np.full(N, 0.08)
        return np.array(list(zip(xs, ys, rs))).flatten()

    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        return np.array(cells).flatten()

    # Try multiple initializations and optimizations
    inits = [init_hexagonal(), init_grid()]
    for _ in range(3):
        inits.append(init_random())

    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        # Step 1: Force relaxation
        relaxed = _force_relax_unified(init, W, H, steps=100, lr=0.05)
        
        # Step 2: Optimization
        polished = _polish_result(relaxed, W, H, max_iters=2000)
        
        # Step 3: Expansion
        expanded = _expansion_relaxation_polish(polished, W, H, max_steps=50)
        
        # Step 4: Contact graph optimization
        contact_opt = _contact_graph_optimize(expanded, W, H, max_iters=200)
        
        # Step 5: Final polish
        final = _polish_result(contact_opt, W, H, max_iters=1000)
        
        # Evaluate
        circles = final.reshape(N, 3)
        score = evaluate_circles(circles)
        
        if score > best_score:
            best_score = score
            best_result = circles.copy()

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
