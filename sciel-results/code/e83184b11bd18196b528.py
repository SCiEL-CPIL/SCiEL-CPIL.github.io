# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    n = 14
    d = 3

    def objective(x):
        """
        Minimize -log(d_min) + log(d_max) which is equivalent to minimizing log(d_max/d_min)
        This is equivalent to maximizing d_min/d_max.
        We use a soft version for differentiability.
        """
        points = x.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 1e10
        if d_min == 0:
            return 1e10
            
        # We want to maximize (d_min/d_max)^2
        # Minimizing -2 * log(d_min) + 2 * log(d_max) is equivalent
        # Or simply minimizing log(d_max) - log(d_min)
        return np.log(d_max) - np.log(d_min)

    def objective_gradient(x):
        """
        Numerical gradient approximation or analytical if possible.
        For now, let's rely on BFGS/Nelder-Mead numerical gradients or use finite differences.
        Actually, scipy.optimize.minimize with method='Nelder-Mead' doesn't need gradients.
        With 'L-BFGS-B' it uses numerical gradients by default if not provided.
        """
        return None

    best_points = None
    best_score = -np.inf

    # Initial guesses
    # 1. Random points
    # 2. Points on a sphere (Fibonacci sphere)
    # 3. Points from a regular polyhedron (if available, but 14 is tricky)

    # Generate several initial guesses
    guesses = []

    # Fibonacci Sphere
    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    guesses.append(fibonacci_sphere(n))

    # Random points on sphere
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        guesses.append(pts)

    # Random points in cube
    for _ in range(5):
        pts = np.random.rand(n, d) * 2 - 1
        guesses.append(pts)

    # Run optimization from each guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline:
            break

        try:
            # Use Nelder-Mead for robustness with non-differentiable min/max
            res = minimize(
                objective,
                guess.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-8
                }
            )
            
            if res.success:
                points = res.x.reshape(n, d)
                dists = pdist(points)
                d_min = np.min(dists)
                d_max = np.max(dists)
                
                if d_max > 0 and d_min > 0:
                    score = (d_min / d_max) ** 2
                    if score > best_score:
                        best_score = score
                        best_points = points

        except Exception:
            continue

    # If no valid points found, return random
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
