"""
Strategy: Physics-based repulsion + Greedy Radius Growth
Instead of generic gradient descent with penalties, we treat the circles as
charged particles that repel each other and the walls. This naturally drives
them into a valid, tight configuration. We then iteratively increase the
minimum radius until no more growth is possible, and finally use a local
refiner to polish the positions for maximum sum of radii.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _physics_step(pos, radii, w, h, dt, iterations=50):
    """
    Perform physics-based relaxation.
    pos: (N, 2) array of centers
    radii: (N,) array of radii
    w, h: dimensions of the rectangle
    dt: time step
    """
    pos = pos.copy()
    n = len(pos)
    
    for _ in range(iterations):
        forces = np.zeros_like(pos)
        
        # Wall repulsion
        for i in range(n):
            r = radii[i]
            # Left
            if pos[i, 0] < r:
                forces[i, 0] += (r - pos[i, 0]) ** 2
            # Right
            if pos[i, 0] > w - r:
                forces[i, 0] -= (pos[i, 0] - (w - r)) ** 2
            # Bottom
            if pos[i, 1] < r:
                forces[i, 1] += (r - pos[i, 1]) ** 2
            # Top
            if pos[i, 1] > h - r:
                forces[i, 1] -= (pos[i, 1] - (h - r)) ** 2
        
        # Circle-Circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-8:
                    # Strong repulsion when overlapping
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    fx = (dx / dist) * force_mag
                    fy = (dy / dist) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist <= 1e-8:
                    # If exactly same position, push apart randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    force_mag = 10.0
                    fx = np.cos(angle) * force_mag
                    fy = np.sin(angle) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions
        pos += dt * forces
        
        # Project back into bounds (soft constraint enforcement)
        for i in range(n):
            r = radii[i]
            if pos[i, 0] < r: pos[i, 0] = r
            if pos[i, 0] > w - r: pos[i, 0] = w - r
            if pos[i, 1] < r: pos[i, 1] = r
            if pos[i, 1] > h - r: pos[i, 1] = h - r
            
    return pos

def _refine_positions(pos, radii, w, h):
    """
    Use scipy to minimize the maximum overlap/penetration to polish the packing.
    Objective: minimize sum of (overlap)^2 + sum of (wall_violation)^2
    """
    def objective(params):
        p = params.reshape(N, 2)
        pen = 0.0
        # Wall penalties
        for i in range(N):
            r = radii[i]
            if p[i, 0] < r: pen += (r - p[i, 0])**2
            if p[i, 0] > w - r: pen += (p[i, 0] - (w - r))**2
            if p[i, 1] < r: pen += (r - p[i, 1])**2
            if p[i, 1] > h - r: pen += (p[i, 1] - (h - r))**2
            
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        return pen

    x0 = pos.flatten()
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    return res.x.reshape(N, 2)

def _is_valid(pos, radii, w, h, tol=1e-5):
    for i in range(N):
        r = radii[i]
        if pos[i, 0] - r < -tol or pos[i, 0] + r > w + tol:
            return False
        if pos[i, 1] - r < -tol or pos[i, 1] + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    # Square: w=1, h=1
    # Rectangle: w=1.5, h=0.5 etc.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial grid placement
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            # Map to [0, w] x [0, h]
            pos[i, 0] = (col + 0.5) * (w / side)
            pos[i, 1] = (row + 0.5) * (h / side)
            
        # Start with small radii
        min_dim = min(w, h)
        base_r = 0.5 * min_dim / side * 0.8
        radii = np.full(N, base_r)
        
        # Phase 1: Physics relaxation to separate circles
        pos = _physics_step(pos, radii, w, h, dt=0.01, iterations=100)
        
        # Phase 2: Greedy radius growth
        # Try to increase all radii uniformly until collision
        growth_rate = 0.001
        max_growth_steps = 500
        
        for step in range(max_growth_steps):
            if time.time() > deadline:
                break
                
            new_radii = radii + growth_rate
            # Check if this growth is feasible with physics relaxation
            test_pos = pos.copy()
            test_pos = _physics_step(test_pos, new_radii, w, h, dt=0.005, iterations=20)
            
            # Check validity
            if _is_valid(test_pos, new_radii, w, h, tol=1e-4):
                pos = test_pos
                radii = new_radii
            else:
                # Try smaller growth
                growth_rate /= 2
                if growth_rate < 1e-6:
                    break
        
        # Phase 3: Refine positions to maximize sum of radii
        # We can try to increase individual radii or just refine positions for current radii
        # Let's refine positions first to ensure no overlaps
        pos = _refine_positions(pos, radii, w, h)
        
        # Check if we can grow further after refinement
        for _ in range(50):
            if time.time() > deadline:
                break
            new_radii = radii + growth_rate
            test_pos = pos.copy()
            test_pos = _physics_step(test_pos, new_radii, w, h, dt=0.005, iterations=20)
            test_pos = _refine_positions(test_pos, new_radii, w, h)
            
            if _is_valid(test_pos, new_radii, w, h, tol=1e-5):
                pos = test_pos
                radii = new_radii
            else:
                growth_rate /= 2
                if growth_rate < 1e-7:
                    break
        
        # Final validation
        if _is_valid(pos, radii, w, h, tol=1e-6):
            value = np.sum(radii)
            if value > best_value:
                best_value = value
                best_result = np.column_stack([pos, radii])
        
        # If invalid, try to shrink slightly to make valid
        else:
            # Simple shrink
            scale = 0.99
            while not _is_valid(pos, radii * scale, w, h, tol=1e-6) and scale > 0.5:
                scale -= 0.01
            if _is_valid(pos, radii * scale, w, h, tol=1e-6):
                value = np.sum(radii * scale)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([pos, radii * scale])

    if best_result is None:
        # Fallback: simple valid grid
        w, h = 1.0, 1.0
        side = 5
        pos = np.zeros((N, 2))
        radii = np.full(N, 0.08)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
        best_result = np.column_stack([pos, radii])
        
    return best_result
