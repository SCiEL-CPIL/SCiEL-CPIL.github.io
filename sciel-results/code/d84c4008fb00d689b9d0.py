"""Circle packing in perimeter-4 rectangle using force-directed optimization."""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_result = None
    best_score = -np.inf
    
    # Initial seed: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    
    # Try multiple initial configurations and aspect ratios
    for trial in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Vary initial aspect ratio
        if trial < 10:
            aspect = 1.0 + 0.1 * (trial % 5)  # 1.0 to 1.4
        elif trial < 20:
            aspect = 0.7 + 0.05 * (trial % 5)  # 0.7 to 0.9
        else:
            aspect = 1.0 + 0.2 * np.random.uniform(-0.5, 0.5)
        
        # Rectangle dimensions with perimeter 4
        # 2*(w + h) = 4, so w + h = 2
        # aspect = w/h, so w = aspect * h
        # aspect * h + h = 2 => h = 2 / (1 + aspect)
        h = 2.0 / (1.0 + aspect)
        w = 2.0 - h
        
        # Initial circle positions: grid + random perturbation
        cols = side
        rows = (N + cols - 1) // cols
        
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            positions[i, 0] = (col + 0.5) * w / cols + np.random.uniform(-0.02, 0.02)
            positions[i, 1] = (row + 0.5) * h / rows + np.random.uniform(-0.02, 0.02)
            radii[i] = 0.5 * min(w / cols, h / rows) * 0.9
        
        # Force-directed optimization
        positions, radii, w, h = optimize_forcedirected(
            positions, radii, w, h, deadline
        )
        
        # Validate and compute score
        if time.time() > deadline - 2.0:
            break
        
        score, valid = validate_and_score(positions, radii, w, h)
        
        if valid and score > best_score:
            best_score = score
            # Format result: (x, y, r) for each circle
            result = np.column_stack([positions, radii])
            best_result = result
    
    # If no valid result found, return a safe grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def optimize_forcedirected(positions, radii, w, h, deadline):
    """
    Force-directed optimization to pack circles in rectangle.
    Simultaneously optimizes positions, radii, and aspect ratio.
    """
    N = len(positions)
    
    # Penalty weights
    overlap_penalty = 100.0
    boundary_penalty = 50.0
    
    # Learning rates
    lr_pos = 0.01
    lr_rad = 0.005
    lr_aspect = 0.01
    
    # Current aspect ratio
    aspect = w / h
    
    for iteration in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        # Compute forces
        pos_grad = np.zeros_like(positions)
        rad_grad = np.zeros_like(radii)
        aspect_grad = 0.0
        
        # Overlap forces between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlap: push apart
                    overlap = min_dist - dist
                    force = overlap_penalty * overlap / dist
                    pos_grad[i, 0] += force * dx
                    pos_grad[i, 1] += force * dy
                    pos_grad[j, 0] -= force * dx
                    pos_grad[j, 1] -= force * dy
                    
                    # Encourage larger radii (gradient of sum of radii)
                    rad_grad[i] += 1.0
                    rad_grad[j] += 1.0
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if positions[i, 0] < radii[i]:
                overlap = radii[i] - positions[i, 0]
                pos_grad[i, 0] += boundary_penalty * overlap
                aspect_grad += boundary_penalty * overlap * h / 4.0
            # Right wall
            if positions[i, 0] > w - radii[i]:
                overlap = positions[i, 0] - (w - radii[i])
                pos_grad[i, 0] -= boundary_penalty * overlap
                aspect_grad -= boundary_penalty * overlap * h / 4.0
            # Bottom wall
            if positions[i, 1] < radii[i]:
                overlap = radii[i] - positions[i, 1]
                pos_grad[i, 1] += boundary_penalty * overlap
                aspect_grad += boundary_penalty * overlap * w / 4.0
            # Top wall
            if positions[i, 1] > h - radii[i]:
                overlap = positions[i, 1] - (h - radii[i])
                pos_grad[i, 1] -= boundary_penalty * overlap
                aspect_grad -= boundary_penalty * overlap * w / 4.0
        
        # Update positions
        positions += lr_pos * pos_grad
        
        # Update radii (encourage growth, but penalize overlaps)
        radii += lr_rad * rad_grad
        radii = np.maximum(radii, 0.01)  # Keep positive
        
        # Update aspect ratio
        aspect += lr_aspect * aspect_grad
        aspect = np.clip(aspect, 0.5, 2.0)
        
        # Update w, h from aspect
        h = 2.0 / (1.0 + aspect)
        w = 2.0 - h
        
        # Keep circles inside bounds (hard constraint)
        for i in range(N):
            positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
    
    return positions, radii, w, h


def validate_and_score(positions, radii, w, h):
    """
    Validate that circles are disjoint and inside rectangle.
    Return (sum of radii, is_valid).
    """
    N = len(positions)
    
    # Check perimeter constraint
    if abs(2 * (w + h) - 4.0) > 1e-6:
        return 0.0, False
    
    # Check boundaries
    for i in range(N):
        if positions[i, 0] < radii[i] - 1e-9 or positions[i, 0] > w - radii[i] + 1e-9:
            return 0.0, False
        if positions[i, 1] < radii[i] - 1e-9 or positions[i, 1] > h - radii[i] + 1e-9:
            return 0.0, False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return 0.0, False
    
    score = np.sum(radii)
    return score, True
