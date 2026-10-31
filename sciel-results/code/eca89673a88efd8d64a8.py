"""Circle packing in perimeter-4 rectangle: topological contact-graph search + geometric optimization."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay, cKDTree
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    # ---------- Helper: compute radii given positions and rectangle ----------
    def compute_radii(coords, w, h):
        """Compute maximal radii for given center positions in rectangle [0,w]x[0,h]."""
        n = len(coords)
        radii = np.zeros(n)
        for i in range(n):
            x, y = coords[i]
            # Distance to walls
            min_dist = min(x, w - x, y, h - y)
            # Distance to other circles
            for j in range(n):
                if i == j:
                    continue
                dx = x - coords[j][0]
                dy = y - coords[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
            radii[i] = 0.5 * min_dist
        return radii
    
    # ---------- Helper: check validity ----------
    def check_validity(coords, radii, w, h, tol=1e-9):
        """Check if circles are disjoint and within rectangle."""
        n = len(coords)
        for i in range(n):
            x, y = coords[i]
            r = radii[i]
            if x < -tol or x > w + tol or y < -tol or y > h + tol:
                return False
            if r < -tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    # ---------- Helper: objective for SLSQP ----------
    def make_objective(w, h):
        def objective(params):
            coords = params.reshape(N, 2)
            # Radii are determined by positions
            radii = compute_radii(coords, w, h)
            return -np.sum(radii)
        return objective
    
    def make_jacobian(w, h):
        def jacobian(params):
            coords = params.reshape(N, 2)
            grad = np.zeros(2*N)
            for i in range(N):
                x, y = coords[i]
                min_dist = min(x, w - x, y, h - y)
                # Determine which wall gives min distance
                wall_x_left = x
                wall_x_right = w - x
                wall_y_bottom = y
                wall_y_top = h - y
                
                # Check other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = x - coords[j][0]
                    dy = y - coords[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
                
                # Gradient of -0.5 * min_dist with respect to coords[i]
                # This is complex; let's use finite differences or just let SLSQP handle it
                # Actually, SLSQP can use numerical Jacobian
                pass
            return None  # Use numerical
        return jacobian
    
    # ---------- Strategy 1: Grid-based initial solutions ----------
    def grid_solution(w, h, cols, rows):
        """Place circles on a grid."""
        coords = []
        for i in range(cols):
            for j in range(rows):
                if len(coords) >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                coords.append([x, y])
        # Fill remaining with random positions
        while len(coords) < N:
            x = random.uniform(0.1*w, 0.9*w)
            y = random.uniform(0.1*h, 0.9*h)
            coords.append([x, y])
        coords = np.array(coords[:N])
        return coords
    
    # ---------- Strategy 2: Random initial solutions ----------
    def random_solution(w, h):
        coords = np.random.uniform(0.1, 0.9, size=(N, 2))
        coords[:, 0] *= w
        coords[:, 1] *= h
        return coords
    
    # ---------- Strategy 3: Hexagonal packing ----------
    def hex_solution(w, h):
        """Hexagonal close packing pattern."""
        coords = []
        r_est = min(w, h) / (2 * 5)  # estimate
        dy = r_est * np.sqrt(3)
        dx = r_est * 2
        row = 0
        while len(coords) < N:
            offset = (row % 2) * dx / 2
            col = 0
            while len(coords) < N:
                x = dx/2 + offset + col * dx
                y = dy/2 + row * dy
                if 0 < x < w and 0 < y < h:
                    coords.append([x, y])
                col += 1
            row += 1
        coords = np.array(coords[:N])
        return coords
    
    # ---------- Optimize a single configuration ----------
    def optimize_config(coords0, w, h, max_time=10):
        """Run SLSQP optimization on a starting configuration."""
        params0 = coords0.flatten()
        
        def objective(params):
            coords = params.reshape(N, 2)
            # Clamp to valid region
            coords[:, 0] = np.clip(coords[:, 0], 0, w)
            coords[:, 1] = np.clip(coords[:, 1], 0, h)
            radii = compute_radii(coords, w, h)
            return -np.sum(radii)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
        
        try:
            t_start = time.time()
            result = minimize(
                objective, params0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            t_end = time.time()
            
            if result.success or result.fun < objective(params0):
                coords_opt = result.x.reshape(N, 2)
                radii = compute_radii(coords_opt, w, h)
                score = np.sum(radii)
                return coords_opt, radii, score, t_end - t_start
        except Exception:
            pass
        
        # Fallback: return original
        radii = compute_radii(coords0, w, h)
        score = np.sum(radii)
        return coords0, radii, score, 0
    
    # ---------- Force-directed relaxation ----------
    def force_relax(coords, w, h, n_iter=200, lr=0.01):
        """Simple force-directed method to resolve overlaps."""
        coords = coords.copy()
        for it in range(n_iter):
            if elapsed() > deadline - 5:
                break
            forces = np.zeros_like(coords)
            for i in range(N):
                # Wall repulsion
                if coords[i][0] < 0:
                    forces[i][0] += 1.0
                if coords[i][0] > w:
                    forces[i][0] -= 1.0
                if coords[i][1] < 0:
                    forces[i][1] += 1.0
                if coords[i][1] > h:
                    forces[i][1] -= 1.0
                
                # Circle repulsion
                for j in range(N):
                    if i == j:
                        continue
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    # Estimate radii
                    r_i = 0.5 * min(coords[i][0], w - coords[i][0], coords[i][1], h - coords[i][1])
                    r_j = 0.5 * min(coords[j][0], w - coords[j][0], coords[j][1], h - coords[j][1])
                    min_dist_needed = r_i + r_j
                    if dist < min_dist_needed:
                        # Repulsive force
                        force_mag = (min_dist_needed - dist) / dist
                        forces[i][0] += dx * force_mag
                        forces[i][1] += dy * force_mag
                        forces[j][0] -= dx * force_mag
                        forces[j][1] -= dy * force_mag
            
            coords += lr * forces
            coords[:, 0] = np.clip(coords[:, 0], 0, w)
            coords[:, 1] = np.clip(coords[:, 1], 0, h)
        
        return coords
    
    # ---------- Radius growth polish ----------
    def polish(coords, w, h, max_time=5):
        """Iteratively grow radii and resolve overlaps."""
        t_start = time.time()
        radii = compute_radii(coords, w, h)
        coords = coords.copy()
        
        for iteration in range(50):
            if time.time() - t_start > max_time:
                break
            if elapsed() > deadline - 3:
                break
            
            # Try to grow each radius slightly
            for i in range(N):
                # Find current max possible radius
                max_r = 0.5 * min(coords[i][0], w - coords[i][0], coords[i][1], h - coords[i][1])
                for j in range(N):
                    if i == j:
                        continue
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, 0.5 * (dist - radii[j]))
                
                if max_r > radii[i] + 1e-6:
                    # Can grow
                    radii[i] = max_r
            
            # Resolve overlaps by moving centers
            for _ in range(20):
                if time.time() - t_start > max_time:
                    break
                moved = False
                for i in range(N):
                    for j in range(i+1, N):
                        dx = coords[i][0] - coords[j][0]
                        dy = coords[i][1] - coords[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = radii[i] + radii[j]
                        if dist < min_dist - 1e-8:
                            # Push apart
                            if dist < 1e-6:
                                dx = random.uniform(-0.001, 0.001)
                                dy = random.uniform(-0.001, 0.001)
                                dist = np.sqrt(dx*dx + dy*dy)
                            push = (min_dist - dist) / dist * 0.5
                            coords[i][0] += dx * push
                            coords[i][1] += dy * push
                            coords[j][0] -= dx * push
                            coords[j][1] -= dy * push
                            moved = True
                
                coords[:, 0] = np.clip(coords[:, 0], 0, w)
                coords[:, 1] = np.clip(coords[:, 1], 0, h)
                
                if not moved:
                    break
            
            # Recompute radii
            new_radii = compute_radii(coords, w, h)
            if np.sum(new_radii) <= np.sum(radii) - 1e-8:
                # No improvement, revert
                break
            radii = new_radii
        
        return coords, radii
    
    # ---------- Main search ----------
    best_score = 0.0
    best_coords = None
    best_radii = None
    best_w = 1.0
    best_h = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.667, 1.333, 1.75, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if elapsed() > deadline - 10:
            break
        
        # w + h = 2, w/h = ar => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Generate multiple initial configurations
        inits = []
        
        # Grid
        for cols in [4, 5, 6]:
            rows = int(np.ceil(N / cols))
            inits.append(grid_solution(w, h, cols, rows))
        
        # Hexagonal
        inits.append(hex_solution(w, h))
        
        # Random
        for _ in range(3):
            inits.append(random_solution(w, h))
        
        # Optimize each
        for init in inits:
            if elapsed() > deadline - 5:
                break
            
            # First, force relax to get a feasible starting point
            relaxed = force_relax(init, w, h, n_iter=100, lr=0.005)
            
            # Then optimize
            coords_opt, radii_opt, score, opt_time = optimize_config(relaxed, w, h, max_time=8)
            
            # Polish
            if elapsed() < deadline - 3:
                coords_pol, radii_pol = polish(coords_opt, w, h, max_time=3)
                score_pol = np.sum(radii_pol)
                if score_pol > score:
                    coords_opt, radii_opt, score = coords_pol, radii_pol, score_pol
            
            if score > best_score:
                best_score = score
                best_coords = coords_opt
                best_radii = radii_opt
                best_w = w
                best_h = h
    
    # Final polish on best result
    if best_coords is not None and elapsed() < deadline - 2:
        coords_final, radii_final = polish(best_coords, best_w, best_h, max_time=2)
        if np.sum(radii_final) > best_score:
            best_coords = coords_final
            best_radii = radii_final
            best_score = np.sum(radii_final)
    
    # Construct output
    if best_coords is None:
        # Fallback: simple grid
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    result = np.column_stack((best_coords, best_radii))
    return result

# EVOLVE-BLOCK-END
