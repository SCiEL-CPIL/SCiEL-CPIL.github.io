"""Rewritten program for circle packing in perimeter-4 rectangle using force-directed relaxation."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
from collections import defaultdict


def _build_contact_graph(centers, radii, w, h, tol=0.05):
    """
    Build contact graph from current configuration.
    Two circles are 'in contact' if their distance is within tol of r_i + r_j.
    Also add wall contacts.
    Returns adjacency list.
    """
    n = len(centers)
    adj = defaultdict(set)
    
    # Circle-circle contacts
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if abs(dist - min_dist) < tol:
                adj[i].add(j)
                adj[j].add(i)
    
    # Wall contacts (mark which walls each circle touches)
    wall_adj = defaultdict(set)  # circle -> set of wall indices (0=left, 1=right, 2=bottom, 3=top)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if abs(x - r) < tol:
            wall_adj[i].add(0)
        if abs(x + r - w) < tol:
            wall_adj[i].add(1)
        if abs(y - r) < tol:
            wall_adj[i].add(2)
        if abs(y + r - h) < tol:
            wall_adj[i].add(3)
    
    return adj, wall_adj


def _force_directed_with_contacts(centers, radii, w, h, iterations=10000, lr_start=0.02, lr_end=0.001):
    """
    Force-directed relaxation using contact graph to maintain structural integrity.
    Uses spring forces for contact edges and repulsion for non-contact overlaps.
    """
    n = len(centers)
    centers = centers.copy().astype(float)
    radii = radii.copy().astype(float)
    
    # Initial contact graph
    adj, wall_adj = _build_contact_graph(centers, radii, w, h, tol=0.1)
    
    for it in range(iterations):
        # Annealing schedule
        progress = it / iterations
        lr = lr_start * (lr_end / lr_start) ** progress
        tol = 0.1 * (1 - progress) + 0.02 * progress
        
        forces = np.zeros_like(centers)
        
        # 1. Repulsive forces for all overlapping pairs (strong penalty)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-10:
                        fx = (dx / dist) * overlap * 10.0
                        fy = (dy / dist) * overlap * 10.0
                    else:
                        angle = random.uniform(0, 2 * np.pi)
                        fx = np.cos(angle) * overlap * 10.0
                        fy = np.sin(angle) * overlap * 10.0
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # 2. Spring forces for contact edges (maintain structure)
        # Contact edges want to be at exactly r_i + r_j
        for i in range(n):
            for j in adj[i]:
                if j > i:  # avoid double counting
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    target = radii[i] + radii[j]
                    
                    if dist > 1e-10:
                        # Spring force: proportional to deviation from target
                        deviation = dist - target
                        k = 5.0  # spring constant
                        fx = (dx / dist) * deviation * k
                        fy = (dy / dist) * deviation * k
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
        
        # 3. Wall constraint forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 10.0
            # Right wall
            if x + r > w:
                forces[i, 0] -= (x + r - w) * 10.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 10.0
            # Top wall
            if y + r > h:
                forces[i, 1] -= (y + r - h) * 10.0
        
        # Apply forces
        centers += lr * forces
        
        # Clamp to rectangle (hard constraint)
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Update contact graph periodically
        if it % 100 == 0:
            adj, wall_adj = _build_contact_graph(centers, radii, w, h, tol=tol)
        
        # Gradually increase radii to explore larger configurations
        growth_rate = 5e-6 * (1 - progress * 0.5)
        radii *= (1.0 + growth_rate)
        
        # Ensure radii don't exceed physical limits
        max_r = min(w, h) / 2.0
        radii = np.minimum(radii, max_r)
    
    return centers, radii


def _evaluate(centers, w, h, radii):
    """Evaluate validity and objective. Returns (is_valid, objective)."""
    n = len(centers)
    
    # Check bounding box perimeter
    if w + h > 2.0 + 1e-6:
        return False, 0.0
    
    # Check all circles are within the rectangle
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
            return False, 0.0
    
    # Check no overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return False, 0.0
    
    # Objective: sum of radii
    return True, np.sum(radii)


def _refine_with_scipy(centers, radii, w, h, n):
    """
    Refine solution using scipy optimization with proper constraints.
    """
    params0 = np.concatenate([centers.flatten(), radii])
    
    def objective(p):
        pos = p[:2 * n].reshape(n, 2)
        r = p[2 * n:]
        
        # Soft penalties for constraints
        penalty = 0.0
        
        # Wall violations
        for i in range(n):
            x, y = pos[i]
            ri = r[i]
            if x - ri < 0:
                penalty += (x - ri) ** 2
            if x + ri > w:
                penalty += (x + ri - w) ** 2
            if y - ri < 0:
                penalty += (y - ri) ** 2
            if y + ri > h:
                penalty += (y + ri - h) ** 2
        
        # Overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(r) + 50.0 * penalty
    
    # Bounds: positions within [r, w-r] and [r, h-r], radii positive
    bounds = []
    for i in range(n):
        bounds.append((radii[i] * 0.5, w - radii[i] * 0.5))  # x
        bounds.append((radii[i] * 0.5, h - radii[i] * 0.5))  # y
    for i in range(n):
        bounds.append((radii[i] * 0.5, min(w, h) / 2.0))  # radius
    
    try:
        result = minimize(objective, params0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < objective(params0):
            opt_params = result.x
            opt_centers = opt_params[:2 * n].reshape(n, 2)
            opt_radii = opt_params[2 * n:]
            return opt_centers, opt_radii
    except Exception:
        pass
    
    return centers, radii


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    
    Returns: array of shape (n, 3) with [x, y, r] for each circle.
    """
    n = N
    
    best_solution = None
    best_score = -np.inf
    
    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [
        (1.0, 1.0),    # square
        (0.9, 1.1),    # slightly tall
        (1.1, 0.9),    # slightly wide
        (0.8, 1.2),    # tall
        (1.2, 0.8),    # wide
        (0.7, 1.3),    # very tall
        (1.3, 0.7),    # very wide
    ]
    
    for trial in range(30):
        random.seed(trial * 7919 + 42)
        np.random.seed(trial * 7919 + 42)
        
        # Select aspect ratio
        w, h = aspect_ratios[trial % len(aspect_ratios)]
        
        # Generate diverse initial layouts
        layout_type = trial % 3
        
        if layout_type == 0:
            # Grid layout
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            centers = []
            for i in range(rows):
                for j in range(cols):
                    if len(centers) >= n:
                        break
                    x = (j + 0.5) * w / cols
                    y = (i + 0.5) * h / rows
                    centers.append([x, y])
            centers = np.array(centers[:n])
            base_r = min(w / cols, h / rows) * 0.45
            radii = np.full(n, base_r)
            
        elif layout_type == 1:
            # Hexagonal packing approximation
            centers = []
            row_height = 0.866  # sqrt(3)/2
            r_est = 0.15
            for row in range(15):
                y = (row + 0.5) * h / 15
                offset = (row % 2) * 0.5
                for col in range(15):
                    if len(centers) >= n:
                        break
                    x = (col + 0.5 + offset) * w / 15
                    centers.append([x, y])
            centers = np.array(centers[:n])
            base_r = 0.12
            radii = np.full(n, base_r)
            
        else:
            # Random layout
            centers = np.random.uniform(0.2, [w - 0.2, h - 0.2], size=(n, 2))
            base_r = 0.10
            radii = np.full(n, base_r)
        
        # Force-directed relaxation with contact graph
        centers, radii = _force_directed_with_contacts(centers, radii, w, h, 
                                                        iterations=8000, 
                                                        lr_start=0.02, 
                                                        lr_end=0.001)
        
        # Check validity
        is_valid, score = _evaluate(centers, w, h, radii)
        
        if is_valid and score > best_score:
            best_score = score
            best_solution = np.column_stack([centers, radii])
        
        # Scipy refinement
        centers, radii = _refine_with_scipy(centers, radii, w, h, n)
        
        is_valid, score = _evaluate(centers, w, h, radii)
        if is_valid and score > best_score:
            best_score = score
            best_solution = np.column_stack([centers, radii])
    
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
# EVOLVE-BLOCK-END
