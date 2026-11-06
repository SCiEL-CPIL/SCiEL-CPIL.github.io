"""Circle packing for N=300 in unit circle using hexagonal lattice init, KD-tree force relaxation, and multi-restart."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances using broadcasting (only for final validation)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_hexagonal_lattice():
        """Generate initial configuration using hexagonal lattice clipped to unit disk."""
        points = []
        # Hexagonal lattice spacing
        # For N=300 in unit circle, approximate spacing
        # Area per circle ~ pi/N, so spacing ~ sqrt(pi/N) * sqrt(4/sqrt(3)) for hex
        # Let's just try a range of spacings and pick the one that gives ~N points
        best_pts = None
        best_count = 0
        for spacing in np.linspace(0.03, 0.08, 50):
            pts = []
            # Generate hexagonal lattice points
            # a1 = (s, 0), a2 = (s/2, s*sqrt(3)/2)
            a1 = np.array([spacing, 0.0])
            a2 = np.array([spacing / 2.0, spacing * np.sqrt(3.0) / 2.0])
            max_idx = int(1.0 / spacing) + 2
            for i in range(-max_idx, max_idx + 1):
                for j in range(-max_idx, max_idx + 1):
                    pt = i * a1 + j * a2
                    if np.linalg.norm(pt) < 0.98:
                        pts.append(pt)
            if len(pts) > best_count:
                best_count = len(pts)
                best_pts = np.array(pts)
        
        if best_pts is None or len(best_pts) == 0:
            # Fallback to Fibonacci
            return generate_fibonacci_disk()
        
        # If we have more than N, randomly select N
        if len(best_pts) > N:
            idx = np.random.choice(len(best_pts), N, replace=False)
            best_pts = best_pts[idx]
        elif len(best_pts) < N:
            # Add random points to fill
            missing = N - len(best_pts)
            extra = np.random.randn(missing, 2) * 0.5
            best_pts = np.vstack([best_pts, extra])
        
        # Project inside
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        outside = radii > 0.98
        if np.any(outside):
            best_pts[outside] = best_pts[outside] * (0.98 / radii[outside, np.newaxis])
        
        return best_pts

    def generate_fibonacci_disk():
        """Generate initial configuration using Fibonacci disk distribution."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        points = np.zeros((N, 2))
        for i in range(N):
            r = np.sqrt((i + 0.5) / N)
            theta = i * golden_angle
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
        
        max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
        if max_r > 0.95:
            points *= 0.95 / max_r
        
        return points

    def optimize_kdtree(pts, deadline_check, max_iters=50000):
        """
        KD-tree based force relaxation.
        Uses cKDTree for efficient neighbor search.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Parameters
        step_size = 0.005
        decay = 0.99995
        target_margin = 0.002  # Target distance above current best
        
        it = 0
        no_improve = 0
        max_no_improve = 5000
        
        while it < max_iters and time.time() < deadline_check:
            it += 1
            
            # Build KD-tree
            tree = cKDTree(pts)
            
            # Find close pairs (within some radius)
            # We want to push apart pairs that are too close
            # Estimate target radius
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(1.0 - radii)
            
            # Find pairs with distance < 2 * (best_score + margin)
            search_radius = 2.0 * (best_score + target_margin)
            
            # Get pairs within search_radius
            pairs = tree.query_pairs(search_radius)
            
            if len(pairs) == 0:
                # No close pairs, try to expand
                # Push points outward slightly
                forces = np.zeros_like(pts)
                for i in range(N):
                    r = np.linalg.norm(pts[i])
                    if r > 1e-10:
                        forces[i] = pts[i] / r * 0.001
                pts += forces * step_size
                # Project
                new_radii = np.hypot(pts[:, 0], pts[:, 1])
                outside = new_radii > 1.0
                if np.any(outside):
                    pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
                continue
            
            # Compute forces from close pairs
            forces = np.zeros_like(pts)
            for i, j in pairs:
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random direction
                    diff = np.random.randn(2)
                    dist = np.linalg.norm(diff)
                # Force magnitude: push apart if too close
                target_dist = 2.0 * (best_score + target_margin)
                if dist < target_dist:
                    force_mag = (target_dist - dist) / target_dist
                    forces[i] += diff / dist * force_mag
                    forces[j] -= diff / dist * force_mag
            
            # Boundary forces
            boundary_violation = np.maximum(radii - (1.0 - best_score - target_margin), 0.0)
            if np.any(boundary_violation > 0):
                for i in np.where(boundary_violation > 0)[0]:
                    r = np.linalg.norm(pts[i])
                    if r > 1e-10:
                        forces[i] -= pts[i] / r * boundary_violation[i] * 5.0
            
            # Update
            pts += forces * step_size
            
            # Project inside unit circle
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
            
            # Compute new score (approximate using KD-tree for speed)
            # For speed, we can compute min pair distance using KD-tree
            tree = cKDTree(pts)
            min_pair_dist = np.inf
            # Find closest pair
            pairs = tree.query_pairs(10.0)  # Large radius to find all close pairs
            if len(pairs) > 0:
                for i, j in pairs[:100]:  # Check first 100 pairs
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            min_pair = min_pair_dist / 2.0
            new_score = min(min_pair, min_boundary)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Decay step size
            step_size *= decay
            
            # If stuck, perturb
            if no_improve > max_no_improve:
                n_perturb = np.random.randint(3, 10)
                idx = np.random.choice(N, n_perturb, replace=False)
                pts[idx] += np.random.randn(n_perturb, 2) * 0.01
                new_radii = np.hypot(pts[:, 0], pts[:, 1])
                outside = new_radii > 1.0
                if np.any(outside):
                    pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
                no_improve = 0
                step_size = 0.005
        
        return best_pts, best_score

    def optimize_annealing(pts, deadline_check, max_iters=10000):
        """Simulated annealing for final refinement."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        temp = 0.003
        cool_rate = 0.99995
        
        for it in range(max_iters):
            if time.time() > deadline_check:
                break
            
            # Perturb a random point
            idx = np.random.randint(N)
            orig_pt = pts[idx].copy()
            pts[idx] += np.random.randn(2) * temp
            
            # Project inside
            r = np.hypot(pts[idx, 0], pts[idx, 1])
            if r > 1.0:
                pts[idx] = pts[idx] * (1.0 / r)
            
            # Compute score using KD-tree for speed
            tree = cKDTree(pts)
            min_pair_dist = np.inf
            pairs = tree.query_pairs(10.0)
            if len(pairs) > 0:
                for i, j in pairs[:50]:
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            min_pair = min_pair_dist / 2.0
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(1.0 - radii)
            new_score = min(min_pair, min_boundary)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
            
            # Accept
            if new_score >= best_score or np.random.rand() < np.exp((new_score - best_score) / (temp + 1e-10)):
                pass  # Keep the move
            else:
                pts[idx] = orig_pt
            
            temp *= cool_rate
        
        return best_pts, best_score

    # Main execution
    np.random.seed(42)
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        generate_hexagonal_lattice,
        generate_fibonacci_disk,
    ]
    
    # Phase 1: Run KD-tree optimization on each initialization
    time_per_init = (deadline - time.time() - 30) / len(initializations)
    
    for init_func in initializations:
        if time.time() > deadline - 30:
            break
        
        pts = init_func()
        init_score = compute_score(pts)
        
        # Run KD-tree optimization
        strategy_deadline = min(deadline - 20, time.time() + time_per_init)
        pts, score = optimize_kdtree(pts, strategy_deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Phase 2: Refine best with another KD-tree pass
    if best_pts is not None and time.time() < deadline - 15:
        pts, score = optimize_kdtree(best_pts, min(deadline - 10, time.time() + 20))
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Phase 3: Simulated annealing for final tweaks
    if best_pts is not None and time.time() < deadline - 5:
        pts, score = optimize_annealing(best_pts, deadline - 2)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final validation
    if best_pts is None:
        best_pts = generate_fibonacci_disk()
    
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
