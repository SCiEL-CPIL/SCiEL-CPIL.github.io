import numpy as np
from scipy.optimize import differential_evolution, L_BFGS_B
from autoresearch_suite.chem_clusters import morse_energy_gradient, relax

def construct(instance):
    """
    Hybrid Global-Local Optimization for Morse Clusters.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # 1. Define Bounds
    # Empirical radius for compact clusters: R ~ 1.5 * n^(1/3)
    # We use a box that comfortably contains the cluster
    box_radius = 2.5 * (size ** (1.0 / 3.0))
    bounds = [(-box_radius, box_radius)] * (3 * size)
    
    # 2. Objective Function for Global Search (with penalty for overlaps)
    def objective_penalized(x):
        r = x.reshape((size, 3))
        # Vectorized distance calculation
        diff = r[:, np.newaxis, :] - r[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Penalty for r < 0.5
        triu_idx = np.triu_indices(size, k=1)
        pair_dists = dists[triu_idx]
        
        # Morse Energy
        arg = rho * (1.0 - pair_dists)
        # Prevent overflow: if r is very small, arg is large positive.
        # exp(arg) can overflow. Clip arg to max ~ 700.
        arg = np.clip(arg, -700, 700)
        e = np.exp(arg)
        energy = np.sum(e * (e - 2.0))
        
        # Distance Penalty
        violations = np.maximum(0, 0.5 - pair_dists)
        penalty = 1000.0 * np.sum(violations**2)
        
        return energy + penalty

    # 3. Gradient Function for Local Search
    def gradient(x):
        r = x.reshape((size, 3))
        _, grad = morse_energy_gradient(r, rho)
        return grad.flatten()

    # 4. Raw Objective for Local Search
    def objective_raw(x):
        r = x.reshape((size, 3))
        e, _ = morse_energy_gradient(r, rho)
        return e

    # 5. Global Search: Differential Evolution
    # Tuned for speed and reasonable exploration
    de_result = differential_evolution(
        objective_penalized,
        bounds,
        maxiter=80,        # Increased iterations
        popsize=12,        # Moderate population
        tol=0.001,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        workers=4,         # Parallelize
        polish=False        # We handle polishing
    )
    
    best_x = de_result.x
    best_pos = best_x.reshape((size, 3))
    
    # 6. Local Refinement: L-BFGS-B
    # This will converge to the local minimum of the raw Morse potential
    # starting from the best DE solution.
    try:
        res = L_BFGS_B(
            objective_raw,
            best_x,
            grad=gradient,
            maxiter=200,
            ftol=1e-10,
            gtol=1e-8
        )
        best_pos = res.x.reshape((size, 3))
    except Exception:
        # Fallback to the provided relax function if L-BFGS-B fails
        _, best_pos = relax(best_pos, potential, rho)

    # 7. Final Check and Return
    return best_pos
