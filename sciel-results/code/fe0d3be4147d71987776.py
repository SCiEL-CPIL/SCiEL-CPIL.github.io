"""
Strategy: Repulsive Force Relaxation with Iterative Hard-Constraint Repair
- Uses a physics-based approach: circles repel each other and walls.
- Uses Langevin dynamics with cooling (velocity + damping + noise).
- Crucially, after every small batch of dynamics steps, it applies a 
  hard-constraint repair: if any circle overlaps or is out of bounds, 
  it shrinks it and pushes it back. This ensures we never stay in an 
  invalid state for long, and the "error_repair" is a core operator.
- The objective is to maximize sum of radii, but we start with fixed 
  large radii and let the positions find space, then allow radii to 
  grow if space permits (via a "growth pressure" term).
"""

import numpy as np
import time

N = 26

def _repair_and_validate(centers, radii):
    """
    Hard constraint repair: 
    1. Ensure circles are inside [0,1]^2.
    2. Resolve overlaps by shrinking the smaller circle in each overlapping pair.
    Returns (valid, sum_radii).
    """
    centers = centers.copy()
    radii = radii.copy()
    
    # 1. Boundary constraint: clamp center and shrink radius if needed
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        
        # Max radius allowed by boundaries
        r_max = min(x, 1.0 - x, y, 1.0 - y)
        
        if r > r_max:
            # Shrink to fit boundary
            radii[i] = max(0.001, r_max)
            # Re-center if possible, otherwise keep center but shrink
            # Actually, if r > r_max, the circle is stuck against a wall.
            # We should adjust center to be as far from wall as possible?
            # No, if we shrink, the center can stay. But to maximize future growth,
            # maybe we should move it towards the center of the available space?
            # For now, just shrink.
            radii[i] = max(0.001, radii[i])
            
        # Ensure center is valid for the new radius
        r = radii[i]
        centers[i, 0] = np.clip(x, r, 1.0 - r)
        centers[i, 1] = np.clip(y, r, 1.0 - r)
        
    # 2. Overlap constraint: iterative shrinking
    # We iterate until no overlaps or max iterations reached
    for _ in range(50):
        modified = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlap detected
                    # Shrink the smaller circle to resolve
                    if radii[i] < radii[j]:
                        # Shrink i
                        # How much? Enough to separate.
                        # New dist should be >= radii_i_new + radii_j
                        # dist = radii_i_new + radii_j => radii_i_new = dist - radii_j
                        # But dist is fixed for now.
                        # Actually, if we shrink i, the overlap reduces.
                        # Let's shrink i by (min_dist - dist) / 2
                        overlap = min_dist - dist
                        radii[i] = max(0.001, radii[i] - overlap / 2)
                        modified = True
                    else:
                        overlap = min_dist - dist
                        radii[j] = max(0.001, radii[j] - overlap / 2)
                        modified = True
                        
        if not modified:
            break
            
    # Final check
    valid = True
    total_r = 0.0
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            valid = False
        total_r += r
            
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                valid = False
                
    return valid, total_r

def _physics_relaxation(centers, radii, time_limit, seed):
    """
    Langevin dynamics with cooling.
    """
    random_seed = seed
    np.random.seed(random_seed)
    
    centers = centers.copy()
    radii = radii.copy()
    velocities = np.zeros_like(centers)
    
    # Initial temperature
    T = 0.05
    T_min = 1e-4
    cooling_rate = 0.9995
    damping = 0.9
    dt = 0.01
    
    start_time = time.time()
    last_repair_time = start_time
    repair_interval = 0.1 # seconds
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = 0.0
    
    # Initial repair
    valid, score = _repair_and_validate(centers, radii)
    if valid and score > best_score:
        best_score = score
        best_centers = centers.copy()
        best_radii = radii.copy()
        
    while time.time() - start_time < time_limit:
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # 1. Repulsive forces between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlapping or close: strong repulsion
                    # Force magnitude proportional to overlap
                    overlap = min_dist - dist
                    # Add a small epsilon to avoid division by zero if dist is tiny
                    force_mag = 10.0 * overlap + 0.1
                    if dist > 1e-12:
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                    else:
                        # Random direction if exactly overlapping
                        fx = np.random.randn()
                        fy = np.random.randn()
                    forces[i] += np.array([fx, fy])
                    forces[j] -= np.array([fx, fy])
                else:
                    # Not overlapping, but maybe close: soft repulsion to keep them separated
                    # This helps find local minima where they are tangent
                    gap = dist - min_dist
                    if gap < 0.1:
                        force_mag = 1.0 * (0.1 - gap)
                        if dist > 1e-12:
                            fx = force_mag * dx / dist
                            fy = force_mag * dy / dist
                        else:
                            fx = 0
                            fy = 0
                        forces[i] += np.array([fx, fy])
                        forces[j] -= np.array([fx, fy])
                        
        # 2. Wall forces (keep circles inside)
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r + 0.05:
                forces[i, 0] += 5.0 * (r + 0.05 - x)
            # Right wall
            if x > 1.0 - r - 0.05:
                forces[i, 0] -= 5.0 * (x - (1.0 - r - 0.05))
            # Bottom wall
            if y < r + 0.05:
                forces[i, 1] += 5.0 * (r + 0.05 - y)
            # Top wall
            if y > 1.0 - r - 0.05:
                forces[i, 1] -= 5.0 * (y - (1.0 - r - 0.05))
                
        # 3. Growth pressure: encourage circles to grow if they have space
        # This is tricky in force-based methods. 
        # Alternative: periodically try to increase radii and repair.
        # For now, we just optimize positions for fixed radii.
        # We will handle radius growth in a separate loop or by starting with large radii.
        
        # Update velocities and positions (Langevin)
        for i in range(N):
            # v = damping * v + force * dt + noise
            noise = np.random.randn(2) * T
            velocities[i] = damping * velocities[i] + forces[i] * dt + noise
            centers[i] += velocities[i] * dt
            
        # Periodic repair
        if time.time() - last_repair_time > repair_interval:
            valid, score = _repair_and_validate(centers, radii)
            if valid and score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_radii = radii.copy()
            last_repair_time = time.time()
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_centers, best_radii, best_score

def _try_grow_radii(centers, radii, time_limit, seed):
    """
    Attempt to grow radii by finding local maxima of free space.
    """
    np.random.seed(seed)
    centers = centers.copy()
    radii = radii.copy()
    
    best_score, _ = _repair_and_validate(centers, radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    start_time = time.time()
    
    # We will try to grow the smallest circles first, or random ones.
    # Strategy: Pick a random circle, try to increase its radius slightly, 
    # repair, and keep if valid and sum increased.
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(0, N)
        
        # Try to grow
        old_r = radii[i]
        new_r = old_r * (1.0 + 0.05) # 5% growth
        
        # Apply growth
        radii[i] = new_r
        
        # Repair
        valid, score = _repair_and_validate(centers, radii)
        
        if valid and score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_radii = radii.copy()
        else:
            # Revert if it didn't help or was invalid
            radii[i] = old_r
            # Re-repair to ensure state is consistent
            _, _ = _repair_and_validate(centers, radii)
            
    return best_centers, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    best_centers = None
    best_radii = None
    best_score = -np.inf
    
    # Multi-start
    for trial in range(10):
        if time.time() > deadline:
            break
            
        seed = trial * 100 + 1
        np.random.seed(seed)
        
        # Initialize
        # Start with a grid of circles with moderate radius
        # 5x5 grid = 25 circles, we need 26.
        # Let's use a 5x5 grid + 1 in center of a cell? Or just random init?
        # Random init with large radii is risky.
        # Let's use a hexagonal-ish packing init with small radii, then grow.
        
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Simple grid init
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                centers[idx, 0] = (i + 0.5) / side
                centers[idx, 1] = (j + 0.5) / side
                radii[idx] = 0.15 # Start small
                idx += 1
        if idx < N:
            # Add remaining
            for k in range(idx, N):
                centers[k, 0] = np.random.rand()
                centers[k, 1] = np.random.rand()
                radii[k] = 0.15
                
        # Phase 1: Physics relaxation to find good positions for current radii
        c1, r1, s1 = _physics_relaxation(centers, radii, time_limit=5.0, seed=seed)
        
        # Phase 2: Grow radii iteratively
        c2, r2, s2 = _try_grow_radii(c1, r1, time_limit=5.0, seed=seed+10)
        
        # Phase 3: Final physics relaxation with grown radii
        c3, r3, s3 = _physics_relaxation(c2, r2, time_limit=5.0, seed=seed+20)
        
        # Phase 4: Final grow attempt
        c4, r4, s4 = _try_grow_radii(c3, r3, time_limit=5.0, seed=seed+30)
        
        # Validate final
        valid, final_score = _repair_and_validate(c4, r4)
        
        if valid and final_score > best_score:
            best_score = final_score
            best_centers = c4.copy()
            best_radii = r4.copy()
            
        # If time allows, do one more long relaxation on the best found
        if best_centers is not None and time.time() < deadline - 10:
            c5, r5, s5 = _physics_relaxation(best_centers, best_radii, time_limit=10.0, seed=seed+100)
            c6, r6, s6 = _try_grow_radii(c5, r5, time_limit=10.0, seed=seed+200)
            
            valid6, final_score6 = _repair_and_validate(c6, r6)
            if valid6 and final_score6 > best_score:
                best_score = final_score6
                best_centers = c6.copy()
                best_radii = r6.copy()

    if best_centers is None:
        # Fallback
        side = 5
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            best_centers[i, 0] = (gx + 0.5) / side
            best_centers[i, 1] = (gy + 0.5) / side
            best_radii[i] = 0.09
            
    # Final repair to ensure validity
    valid, score = _repair_and_validate(best_centers, best_radii)
    
    result = np.column_stack([best_centers, best_radii])
    
    return result
