"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _gradient(pts):
    """Compute gradient of log energy w.r.t. point positions (tangent space)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            # d/dx_i log(1/dist) = -d/dist^2 = -d_i / dist^2
            # But we want gradient of sum log(1/d), so:
            # d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j) / |x_i - x_j|^2
            grad[i] += -d / (dist * dist)
    # Project to tangent space: subtract component along x_i
    for i in range(n):
        x_i = pts[i]
        grad[i] -= np.dot(grad[i], x_i) * x_i
    return grad

def _gradient_descent(pts, max_steps=50000, lr_start=0.01, lr_decay=0.9999, tol=1e-12, deadline=None):
    """Run gradient descent with adaptive learning rate and sphere projection."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = energy
    lr = lr_start
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient(pts)
        
        # Adaptive learning rate: reduce if energy increases
        new_pts = _project_to_sphere(pts - lr * grad)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            energy = new_energy
            pts = new_pts
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            # Slightly increase lr if we're making good progress
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
        
        # Check convergence
        if step > 100 and np.linalg.norm(grad) < tol:
            break
    
    return best_pts, best_energy

def _simulated_annealing(pts, max_steps=20000, t_start=1.0, t_end=0.01, deadline=None):
    """Run simulated annealing to escape local minima."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = energy
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to perturb
        i = np.random.randint(n)
        # Generate random tangent direction
        rand_vec = np.random.standard_normal(3)
        tangent = rand_vec - np.dot(rand_vec, pts[i]) * pts[i]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            continue
        tangent /= tangent_norm
        
        # Perturb
        step_size = 0.01 * t
        new_pts = pts.copy()
        new_pts[i] = _project_to_sphere(new_pts[i] + step_size * tangent)
        new_energy = _log_energy(new_pts)
        
        delta = new_energy - energy
        if delta < 0 or np.random.random() < math.exp(-delta / max(t, 1e-10)):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        t *= cooling
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple restarts with different initializations
    best_pts = None
    best_energy = float('inf')
    
    # Start with Fibonacci spiral
    fib_pts = _fibonacci_points(N)
    refined, energy = _gradient_descent(fib_pts, max_steps=30000, deadline=deadline)
    if energy < best_energy:
        best_energy = energy
        best_pts = refined.copy()
    
    # Try random restarts
    rng = np.random.RandomState(42)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        rand_pts = _random_points(N, rng)
        refined, energy = _gradient_descent(rand_pts, max_steps=20000, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = refined.copy()
    
    # If we have time, try simulated annealing on the best result
    if time.time() < deadline - 10.0:
        sa_pts, sa_energy = _simulated_annealing(best_pts, max_steps=10000, t_start=0.5, t_end=0.001, deadline=deadline)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts.copy()
        
        # Final gradient descent polish
        if time.time() < deadline - 5.0:
            final_pts, final_energy = _gradient_descent(best_pts, max_steps=10000, deadline=deadline)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_points(N)
    
    return best_pts
# EVOLVE-BLOCK-END
