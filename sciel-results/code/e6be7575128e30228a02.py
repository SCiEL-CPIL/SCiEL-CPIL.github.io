import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Best known radius is ~0.125825
    # We aim to find a packing with radius close to or exceeding this.
    # Start with a conservative radius and increase it.
    r_best = 0.120
    best_pos = None
    
    # We will try to increase r from 0.120 up to 0.126
    # Step size for r
    dr = 0.0001
    
    def energy(pos, r):
        # pos: (N, 2)
        # Calculate pairwise distances
        # d_ij = ||pos_i - pos_j||
        # Overlap penalty: max(0, 2r - d_ij)^2
        # Boundary penalty: max(0, ||pos_i|| + r - 1)^2
        
        # Vectorized pairwise distance
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist = np.sqrt(np.sum(diff**2, axis=2))
        # This creates a (N, N) matrix. N=50 is small, so this is fine.
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal (distance 0)
        np.fill_diagonal(dist, 1e10)
        
        # Overlaps
        overlap_dist = 2.0 * r - dist
        overlap_pen = np.where(overlap_dist > 0, overlap_dist**2, 0.0)
        
        # Boundary
        radii = np.sqrt(np.sum(pos**2, axis=1))
        bound_pen = np.where(radii + r > 1.0, (radii + r - 1.0)**2, 0.0)
        
        return np.sum(overlap_pen) + np.sum(bound_pen)

    def simulate_annealing(r, max_iter=50000, T_start=1.0, T_end=1e-4, restarts=5):
        """
        Attempts to find a valid configuration for circles of radius r.
        Returns (success, positions).
        """
        for _ in range(restarts):
            # Initialize positions randomly in the valid region
            # Radius of container for centers is 1 - r
            R_container = 1.0 - r
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N)) * R_container
            pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            
            T = T_start
            cooling = (T_end / T_start) ** (1.0 / max_iter)
            
            best_E = energy(pos, r)
            best_pos = pos.copy()
            
            for i in range(max_iter):
                # Propose move: pick a random particle and move it
                idx = np.random.randint(N)
                # Move magnitude scales with T
                move_mag = 0.05 * T + 0.001
                dx = np.random.uniform(-move_mag, move_mag)
                dy = np.random.uniform(-move_mag, move_mag)
                
                old_x, old_y = pos[idx]
                pos[idx] = [old_x + dx, old_y + dy]
                
                new_E = energy(pos, r)
                delta_E = new_E - best_E
                
                # Metropolis criterion
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    # Accept move
                    if new_E < best_E:
                        best_E = new_E
                        best_pos = pos.copy()
                else:
                    # Reject move
                    pos[idx] = [old_x, old_y]
                
                T *= cooling
                
                if best_E < 1e-8:
                    return True, best_pos
            
            if best_E < 1e-8:
                return True, best_pos
        
        return False, best_pos

    r_current = 0.120
    
    while r_current < 0.126 and (time.time() - start_time) < time_limit:
        # Try to find a packing for r_current
        # Use fewer iterations per restart if time is tight, but more restarts
        success, pos = simulate_annealing(r_current, max_iter=20000, restarts=3)
        
        if success:
            r_best = r_current
            best_pos = pos
            r_current += dr
        else:
            # If failed, maybe try a slightly smaller r or just stop
            # To be safe, if we fail, we might be stuck in a local minimum for this r.
            # We can try one more time with more iterations if we have time
            if (time.time() - start_time) < time_limit - 20:
                success, pos = simulate_annealing(r_current, max_iter=50000, restarts=1)
                if success:
                    r_best = r_current
                    best_pos = pos
                    r_current += dr
                else:
                    # Failed to pack at this radius. Break to save time or try smaller?
                    # Usually, if we can't pack at r, we can't pack at r+dr.
                    # So we break.
                    break
            else:
                break

    if best_pos is None:
        # Fallback to seed-like solution if SA fails completely
        # Generate a simple hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pos = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98

    return best_pos
