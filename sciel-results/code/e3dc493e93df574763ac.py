import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    N = 50
    
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts
    
    def normalize(v):
        norms = np.linalg.norm(v, axis=1)
        norms[norms == 0] = 1.0
        return v / norms[:, np.newaxis]
    
    def covering_radius(points):
        # Approximate covering radius by sampling the sphere
        # Use a dense grid or random samples
        # For speed, use a fixed set of sample points
        # Generate sample points once outside if possible, but for simplicity here:
        # We will use a moderate number of samples for the inner loop optimization
        # and a denser check for the final score if needed.
        # However, the evaluator calculates the exact score. We just need to minimize it.
        
        # Sample points for optimization
        M = 5000
        # Simple sphere sampling
        u = np.random.uniform(-1, 1, M)
        v = np.random.uniform(-1, 1, M)
        w = np.random.uniform(-1, 1, M)
        samples = np.column_stack([u, v, w])
        norms = np.linalg.norm(samples, axis=1)
        norms[norms == 0] = 1.0
        samples = samples / norms[:, np.newaxis]
        
        # Calculate distance from each sample to nearest point
        # dist_sq = ||s - p||^2 = 2 - 2 * s.p
        # We want max over s of min over p of dist(s, p)
        # Equivalent to max over s of min over p of (2 - 2 * s.p)
        # Which is 2 - 2 * max over s of max over p of (s.p) ... wait.
        # min_p dist(s, p) = sqrt(2 - 2 * max_p (s.p))
        # So we want to maximize min_p dist(s, p)
        # Which is maximizing sqrt(2 - 2 * max_p (s.p))
        # Which is minimizing max_p (s.p) for the worst s?
        # No.
        # Let d(s) = min_p ||s - p||.
        # We want max_s d(s).
        # d(s) = sqrt(2 - 2 * max_p (s.p)).
        # So we want to maximize sqrt(2 - 2 * M(s)) where M(s) = max_p (s.p).
        # This is equivalent to minimizing M(s) for the s that gives the max d(s)?
        # No, we want the s that maximizes d(s).
        # So we find s that minimizes M(s) = max_p (s.p)?
        # If M(s) is small, then 2 - 2*M(s) is large, so d(s) is large.
        # Yes. The worst point s* is the one that minimizes the maximum dot product with any point p.
        
        # Calculate dot products: S (M x N)
        dots = samples @ points.T
        # For each sample, find max dot product
        max_dots = np.max(dots, axis=1)
        # We want the sample with the MINIMUM max_dot
        # Because min max_dot -> max distance
        worst_idx = np.argmin(max_dots)
        worst_point = samples[worst_idx]
        
        # Calculate the covering radius (approx)
        # dist = arccos(max_dot)
        # Actually, the distance is arccos(max_dot)
        # The covering radius is the max over all s of arccos(max_p s.p)
        # So we need the max of arccos(max_dots)
        # Since arccos is decreasing, we need the MIN of max_dots
        # So the covering radius is arccos(min(max_dots))
        cr = np.arccos(np.clip(np.min(max_dots), -1, 1))
        return cr, worst_point

    # Initialize
    points = fibonacci_spiral(N)
    best_points = points.copy()
    best_cr, _ = covering_radius(points)
    
    # Optimization loop
    # Use a simple hill-climbing / subgradient method
    # Move the point closest to the worst point towards the worst point
    
    alpha = 0.1
    for i in range(10000):
        if time.time() > deadline:
            break
            
        cr, worst_point = covering_radius(points)
        
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
            
        # Find which point is closest to the worst point
        dots_to_worst = points @ worst_point
        closest_idx = np.argmax(dots_to_worst)
        
        # Move that point towards the worst point
        # New point = normalize(points[closest_idx] + alpha * (worst_point - points[closest_idx]))
        # This is a step towards the worst point.
        # But we might want to move it *to* the worst point?
        # If we move it exactly, the worst point might become a different one.
        # A smaller step is safer.
        
        step = alpha * (worst_point - points[closest_idx])
        points[closest_idx] += step
        points[closest_idx] /= np.linalg.norm(points[closest_idx])
        
        # Also, to prevent points from sticking too close, we can add a small repulsion
        # But for covering radius, we just want to fill the gaps.
        
        # Optional: Jitter to escape local minima
        if i % 100 == 0 and i > 0:
            # Slightly perturb all points
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            points = normalize(points)

    # Final polish with a denser check
    # Re-evaluate with more samples to ensure we have a good estimate
    # And do a few more iterations with the denser sample if time permits
    
    if time.time() < deadline - 10:
        # Do a few more steps with higher resolution
        for i in range(500):
            if time.time() > deadline - 5:
                break
            # Use higher M for accuracy in finding worst point
            M = 20000
            u = np.random.uniform(-1, 1, M)
            v = np.random.uniform(-1, 1, M)
            w = np.random.uniform(-1, 1, M)
            samples = np.column_stack([u, v, w])
            norms = np.linalg.norm(samples, axis=1)
            norms[norms == 0] = 1.0
            samples = samples / norms[:, np.newaxis]
            
            dots = samples @ points.T
            max_dots = np.max(dots, axis=1)
            worst_idx = np.argmin(max_dots)
            worst_point = samples[worst_idx]
            
            cr = np.arccos(np.clip(np.min(max_dots), -1, 1))
            
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
            
            dots_to_worst = points @ worst_point
            closest_idx = np.argmax(dots_to_worst)
            
            # Smaller step for polish
            step = 0.05 * (worst_point - points[closest_idx])
            points[closest_idx] += step
            points[closest_idx] /= np.linalg.norm(points[closest_idx])

    return best_points
