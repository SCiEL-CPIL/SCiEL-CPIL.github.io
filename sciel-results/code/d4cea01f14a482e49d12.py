import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Generate test points for objective evaluation
    # We use a dense set of points to approximate the covering radius
    # using the sum of squared distances to the nearest point.
    rng = np.random.default_rng(42)
    n_test = 2000
    # Random points on sphere
    test_points = rng.normal(size=(n_test, 3))
    test_points /= np.linalg.norm(test_points, axis=1, keepdims=True)
    
    # 3. Objective Function
    def objective(params):
        # params shape (N, 2): theta (azimuth), phi (elevation)
        theta = params[:, 0]
        phi = params[:, 1]
        
        # Convert to Cartesian
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)
        pts = np.column_stack([x, y, z])
        
        # Normalize (just in case of numerical drift, though cos/sin should be fine)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Compute distances to nearest point for all test points
        # dist^2 = 2 - 2 * dot(test, pts)
        # We want to minimize the max distance, but sum of squares is a good proxy.
        # Let's use the max of the min distances for a subset, or sum of squares.
        # Sum of squares is smoother.
        
        # Efficient computation of min distance
        # dots shape (n_test, N)
        dots = test_points @ pts.T
        # Clip to [-1, 1] to avoid numerical issues with arccos
        dots = np.clip(dots, -1.0, 1.0)
        
        # Angular distance
        # d = arccos(dot)
        # We can optimize d^2 or d. d is non-differentiable at 0, but arccos is fine elsewhere.
        # Actually, minimizing sum of d^2 is equivalent to maximizing sum of dot.
        # Wait, max distance corresponds to min dot.
        # So we want to maximize the minimum dot product for each test point.
        # Let min_dot = min(pts @ test.T) for each test point.
        # We want to maximize min(min_dot) or minimize sum of (1 - min_dot).
        
        min_dots = np.min(dots, axis=1)
        
        # Objective: Minimize sum of (1 - min_dots)^2
        # This penalizes points where the nearest neighbor is far away (low dot product).
        diffs = 1.0 - min_dots
        return np.sum(diffs * diffs)
        
    # 4. Initial parameters from Cartesian points
    def cart_to_sph(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        theta = np.arctan2(y, x)
        phi = np.arcsin(np.clip(z, -1.0, 1.0))
        return np.column_stack([theta, phi])
        
    params = cart_to_sph(points)
    
    # 5. Optimization
    # Use Powell or Nelder-Mead? L-BFGS-B needs gradients. We don't have them easily.
    # Let's use a local search with finite differences or just a heuristic local search.
    # Given the time limit, a simple iterative improvement is safer.
    
    # Heuristic Local Search
    best_obj = objective(params)
    
    # Pre-compute some constants
    lr = 0.05  # Learning rate / step size
    
    while time.time() < deadline:
        improved = False
        # Shuffle points to avoid bias
        idx = np.arange(N)
        rng.shuffle(idx)
        
        for i in idx:
            if time.time() > deadline:
                break
                
            original_params = params[i].copy()
            
            # Try small random perturbations
            for _ in range(10):
                # Random perturbation in theta, phi
                delta = rng.normal(0, lr, size=2)
                new_params = params.copy()
                new_params[i] = original_params + delta
                
                # Handle wrapping for theta
                new_params[i, 0] = np.arctan2(np.sin(new_params[i, 0]), np.cos(new_params[i, 0]))
                # Clamp phi
                new_params[i, 1] = np.clip(new_params[i, 1], -np.pi/2, np.pi/2)
                
                new_obj = objective(new_params)
                
                if new_obj < best_obj:
                    params = new_params
                    best_obj = new_obj
                    improved = True
                    lr = max(lr * 0.95, 0.001) # Decay step size slightly
                    break
            if not improved and rng.random() < 0.1:
                lr = max(lr * 0.9, 0.001)

    # Final conversion to Cartesian
    theta = params[:, 0]
    phi = params[:, 1]
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    final_points = np.column_stack([x, y, z])
    
    # Normalize
    final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
    
    return final_points
