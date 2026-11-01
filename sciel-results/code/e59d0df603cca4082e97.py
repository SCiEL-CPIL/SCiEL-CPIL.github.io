"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


def compute_radius(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^3."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.abs(centers)
    min_wall = wall.min()
    # Pairwise distances
    d = pdist(centers)
    min_pair = d.min() / 2.0
    return min(min_wall, min_pair)


def soft_min_radius(centers, beta):
    """
    Smooth surrogate for the min radius objective.
    Uses log-sum-exp style soft-min for both wall and pair constraints.
    Returns a smooth approximation of the min radius.
    """
    centers = np.asarray(centers)
    # Wall distances
    wall = 0.5 - np.abs(centers)
    # Pairwise distances / 2
    d = pdist(centers) / 2.0
    
    # Soft min over wall distances
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # For numerical stability, shift by min
    w_min = wall.min()
    w_soft = w_min - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (wall - w_min))))
    
    # Soft min over pair distances
    p_min = d.min()
    p_soft = p_min - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d - p_min))))
    
    # The radius is min(w_soft, p_soft) -- but we want to maximize this.
    # Use a soft-min of the two:
    m = min(w_soft, p_soft)
    r_soft = m - (1.0 / beta) * np.log(np.exp(-beta * (w_soft - m)) + np.exp(-beta * (p_soft - m)))
    
    return r_soft


def soft_min_radius_grad(centers, beta):
    """
    Gradient of the soft-min radius surrogate.
    """
    centers = np.asarray(centers)
    wall = 0.5 - np.abs(centers)
    d = pdist(centers) / 2.0
    
    # Soft min over wall
    w_min = wall.min()
    exp_w = np.exp(-beta * (wall - w_min))
    sum_w = np.sum(exp_w)
    w_soft = w_min - (1.0 / beta) * np.log(sum_w)
    
    # Soft min over pairs
    p_min = d.min()
    exp_p = np.exp(-beta * (d - p_min))
    sum_p = np.sum(exp_p)
    p_soft = p_min - (1.0 / beta) * np.log(sum_p)
    
    m = min(w_soft, p_soft)
    exp_m_w = np.exp(-beta * (w_soft - m))
    exp_m_p = np.exp(-beta * (p_soft - m))
    sum_m = exp_m_w + exp_m_p
    r_soft = m - (1.0 / beta) * np.log(sum_m)
    
    # Now compute gradient. This is complex; let's use a simpler approach.
    # Actually, for optimization, let's just use the raw soft objective and let scipy handle it,
    # or use a simpler formulation.
    
    # Simpler: just return the soft radius, and use numerical gradient or a simpler gradient.
    # For efficiency, let's use a differentiable but simpler objective:
    # Minimize the "worst" constraint in a smooth way.
    
    # Actually, let's just use the soft-min value and compute gradient numerically if needed,
    # or use a simpler smooth objective.
    
    # Let's use a simpler approach: minimize the sum of squared violations of a target radius.
    # But that requires a target. Instead, let's just return the soft radius and use scipy's
    # gradient estimation.
    
    return r_soft


def make_initial_grid(n):
    """Create a simple cubic grid of n points in [-0.5, 0.5]^3."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with a grid
    centers = make_initial_grid(N)
    best_centers = centers.copy()
    best_radius = compute_radius(centers)
    
    # Simulated annealing with soft-min objective
    np.random.seed(42)
    
    T = 0.01  # Initial temperature
    T_min = 1e-8
    cooling = 0.999
    beta_start = 100.0
    beta_end = 5000.0
    
    # Number of steps based on time budget
    # Estimate: each step involves pdist (O(N^2)) and some ops
    # N=100, pdist is ~4950 distances, fast
    # Let's do as many steps as we can
    
    step = 0
    current_radius = best_radius
    
    while time.time() < deadline:
        step += 1
        
        # Update beta
        progress = min(1.0, step / 10000.0)
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Generate a perturbation
        # Random move: perturb one or a few centers
        n_perturb = np.random.choice([1, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.1])
        idx = np.random.choice(N, size=n_perturb, replace=False)
        
        old_pos = centers[idx].copy()
        
        # Perturbation size scales with temperature
        sigma = T * 0.05
        centers[idx] += np.random.randn(n_perturb, 3) * sigma
        
        # Keep centers in the cube
        centers[idx] = np.clip(centers[idx], -0.499, 0.499)
        
        # Evaluate
        new_radius = compute_radius(centers)
        
        if new_radius > current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta = new_radius - current_radius
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_radius = new_radius
                else:
                    centers[idx] = old_pos
            
        # Cool
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Every 500 steps, do a local polish with L-BFGS-B on the best solution
        if step % 500 == 0 and time.time() < deadline - 10.0:
            # Local optimization on a copy
            local_centers = best_centers.copy()
            
            def objective(x):
                c = x.reshape(N, 3)
                # Penalize going outside the cube
                wall = 0.5 - np.abs(c)
                if np.any(wall < 0):
                    return -1e6
                d = pdist(c) / 2.0
                return -min(wall.min(), d.min())
            
            try:
                result = minimize(
                    objective,
                    local_centers.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                local_radius = compute_radius(result.x.reshape(N, 3))
                if local_radius > best_radius:
                    best_radius = local_radius
                    best_centers = result.x.reshape(N, 3).copy()
                    centers = best_centers.copy()
                    current_radius = best_radius
            except:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        final_centers = best_centers.copy()
        
        def objective_final(x):
            c = x.reshape(N, 3)
            wall = 0.5 - np.abs(c)
            if np.any(wall < 0):
                return -1e6
            d = pdist(c) / 2.0
            return -min(wall.min(), d.min())
        
        try:
            result = minimize(
                objective_final,
                final_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            final_radius = compute_radius(result.x.reshape(N, 3))
            if final_radius > best_radius:
                best_radius = final_radius
                best_centers = result.x.reshape(N, 3).copy()
        except:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
