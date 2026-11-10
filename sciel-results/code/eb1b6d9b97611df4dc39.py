"""
Simulated Annealing for packing 300 circles in a unit circle.
Strategy:
1. Initialize with a hexagonal lattice to provide a good starting point.
2. Use Simulated Annealing to explore the configuration space.
   - Moves: Random perturbations of individual points.
   - Acceptance: Metropolis criterion based on score improvement/degradation.
3. Vectorized score calculation to speed up the inner loop.
4. Multiple restarts if time permits, keeping the best solution found.
"""

import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    def compute_score(points):
        """
        Compute the packing radius r.
        r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
        Vectorized for speed.
        """
        n = len(points)
        if n < 2:
            return 0.0
        
        # Pairwise distances
        # Using broadcasting creates an (N, N) matrix. For N=300, this is 90k elements, very fast.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Exclude diagonal (self-distance)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_pair_dist_sq = np.min(dist_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Distance to boundary
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def generate_initial_hex():
        """Generate a hexagonal lattice packing."""
        # Approximate optimal spacing for hex packing in a circle
        # Area argument: N * pi * r^2 <= pi * R^2 * packing_efficiency
        # r ~ sqrt(0.907 / N)
        # spacing = 2*r
        r_est = np.sqrt(0.907 / N)
        spacing = 2.0 * r_est
        
        pts = []
        # Generate points in a large square and filter
        # Max radius 1, so range -1.5 to 1.5 is safe
        k = int(2.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0:
                    pts.append((x, y))
        
        pts = np.array(pts)
        
        # We need exactly N points. If we have more, pick the N closest to center?
        # Or if we have less, we need to adjust spacing.
        # Let's try to fit N points.
        # If len(pts) < N, we need smaller spacing.
        # If len(pts) > N, we need larger spacing? No, we want max radius, so we want them as tight as possible.
        # Actually, for the initial guess, just taking a hex grid that fits N points is fine.
        
        # Let's refine: find spacing such that we get at least N points, then take N closest to center?
        # No, taking closest to center leaves empty space at boundary, which is bad for max radius.
        # Better: find spacing that yields ~N points, then randomize slightly?
        
        # Let's just use a spacing that gives roughly N points in the disk.
        # Area of hex unit cell = spacing * spacing * sqrt(3)/2
        # Total area = pi
        # N ~ pi / (spacing^2 * sqrt(3)/2)
        # spacing ~ sqrt(2 * pi / (N * sqrt(3)))
        spacing_opt = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        pts = []
        k = int(2.0 / spacing_opt) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing_opt
                y = j * spacing_opt * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    pts.append((x, y))
        
        pts = np.array(pts)
        
        if len(pts) < N:
            # Need tighter packing
            # This is rare with the formula above, but handle it
            return None
            
        # If we have more than N, we have to remove some.
        # Removing random ones might create holes.
        # Removing outer ones keeps the center dense but wastes boundary.
        # For an initial guess, let's just take the first N or random N?
        # Actually, a better initial guess is to use a slightly smaller spacing so we have > N points,
        # then use a greedy algorithm to place N points?
        
        # Simpler: Just use the hex lattice points that fit, and if we have > N,
        # we can just pick N of them. The SA will fix the arrangement.
        # To maximize initial score, pick N points that are well-distributed.
        # Sorting by distance to center and taking the first N is a decent heuristic.
        radii = np.linalg.norm(pts, axis=1)
        idx = np.argsort(radii)[:N]
        return pts[idx]

    best_points = None
    best_score = 0.0

    # Multiple restarts to escape local optima
    # Budget: 145s total.
    # Let's do 5 restarts, ~28s each.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Initialize
        points = generate_initial_hex()
        if points is None:
            # Fallback: random uniform in disk
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                    if x*x + y*y <= 1.0:
                        points[i] = [x, y]
                        break
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Simulated Annealing
        T = 0.01 # Initial temperature (relative to score scale)
        # Score is around 0.05. So T=0.01 allows some bad moves.
        T_min = 1e-6
        alpha = 0.9995 # Cooling rate
        
        # Time limit for this restart
        restart_deadline = time.time() + 25.0
        
        step_size = 0.05
        
        while time.time() < restart_deadline and T > T_min:
            # Pick a random point to move
            i = np.random.randint(N)
            
            # Random perturbation
            # Gaussian noise
            noise = np.random.normal(0, step_size, 2)
            new_point = points[i] + noise
            
            # Check boundary constraint
            # If it goes outside, project back or reject?
            # Projecting is better for keeping valid states.
            r_new = np.linalg.norm(new_point)
            if r_new > 1.0:
                new_point = new_point / r_new
                # It's on the boundary. This is valid, but might be suboptimal.
                # Let's allow it, but it reduces the boundary margin for this point.
            
            old_point = points[i]
            points[i] = new_point
            
            new_score = compute_score(points)
            
            delta = new_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Reject move
                points[i] = old_point
                
            T *= alpha
            # Slightly decrease step size as well?
            step_size *= 0.9999

    return best_points

# Keep the interface clean
if __name__ == "__main__":
    pts = construct()
    # Quick check
    # print(pts.shape)
    pass
