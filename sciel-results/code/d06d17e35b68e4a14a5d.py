"""
Strategy: Basin Hopping with BCC Initialization.
The current seed program uses random packing and a generic relax function.
This program uses a specific global optimization algorithm: Basin Hopping.
It initializes the system in a BCC lattice (known to be near-optimal for 
long-range Morse potentials), then iteratively:
1. Locally minimizes the energy using L-BFGS-B (via scipy.optimize.minimize).
2. Randomly perturbs the positions.
3. Accepts the new state based on a Metropolis criterion (or always in this simplified version, 
   though standard BH uses Metropolis. For finding global minima, we can just keep the best 
   and occasionally accept worse to escape, or just do random restarts from best. 
   Actually, simple "Best of N random perturbations from current best" is a valid form of 
   stochastic local search that often outperforms naive random restarts by leveraging 
   the local landscape structure).
   
   Let's implement a robust version:
   - Start with BCC.
   - Local Minimize.
   - Loop:
     - Perturb.
     - Local Minimize.
     - If better, keep.
     - If time is up, break.
   
   This is effectively a "steepest descent" in the space of local minima, guided by random kicks.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """
    Compute exact Morse energy and gradient.
    E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise distances
    # Using broadcasting for efficiency
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    
    # Avoid division by zero / self-interaction issues in gradient calculation
    # We only sum over i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_vals = r[mask]
    
    # Energy
    e = np.exp(rho * (1 - r_vals))
    energy = np.sum(e * (e - 2))
    
    # Gradient
    # dE/dr = rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 1)
    # dE/dx_i = -dE/dr * (r_vec / r)
    # Note: The force on atom i is -grad_i E.
    # Let's compute the gradient of E with respect to positions.
    
    # dE/dr_ij = rho * e_ij * (2*e_ij - 1)
    de_dr = rho * e * (2 * e - 1)
    
    # We need to distribute this back to the positions.
    # grad_i E = sum_{j != i} de_dr_ij * ( (r_i - r_j) / r_ij )
    # Note: r_vec_ij = r_i - r_j. 
    # d r_ij / d r_i = (r_i - r_j) / r_ij
    # d r_ij / d r_j = (r_j - r_i) / r_ij = - (r_i - r_j) / r_ij
    
    # Create a matrix of forces/grads
    # de_dr is 1D array of length n*(n-1)/2 corresponding to upper triangle
    # We need to map it back to a 2D matrix
    de_dr_matrix = np.zeros((n, n))
    de_dr_matrix[mask] = de_dr
    
    # grad_i = sum_j de_dr_ij * (r_i - r_j) / r_ij
    # Let's compute the vector part
    # r_vec / r
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.where(r > 1e-10, 1.0/r, 0.0)
    
    # grad_matrix[i, j, k] = de_dr[i,j] * (positions[i,k] - positions[j,k]) * r_inv[i,j]
    # This is a bit heavy to do with full 3D broadcasting for n=80 (80^3 = 512k, fine)
    
    # Alternative: 
    # grad = np.zeros_like(positions)
    # for i in range(n):
    #   for j in range(i+1, n):
    #     r_vec = positions[i] - positions[j]
    #     r_val = np.linalg.norm(r_vec)
    #     if r_val < 1e-10: continue
    #     de_dr_val = de_dr_matrix[i,j]
    #     factor = de_dr_val / r_val
    #     grad[i] += factor * r_vec
    #     grad[j] -= factor * r_vec
    
    # Vectorized version:
    # diff is (n, n, 3)
    # r_inv is (n, n)
    # de_dr_matrix is (n, n)
    
    # term = de_dr_matrix[:, :, np.newaxis] * diff * r_inv[:, :, np.newaxis]
    # grad = np.sum(term, axis=1)
    
    # Wait, sign check.
    # E(r). dE/dx_i = dE/dr * dr/dx_i.
    # r = |x_i - x_j|. dr/dx_i = (x_i - x_j)/r.
    # So grad_i = sum_j dE/dr_ij * (x_i - x_j)/r_ij.
    # My `diff` is `positions[:, np.newaxis, :] - positions[np.newaxis, :, :]` which is x_i - x_j.
    # So term[i,j] = dE/dr_ij * (x_i - x_j)/r_ij.
    # Summing over j for each i gives grad_i.
    
    term = de_dr_matrix[:, :, np.newaxis] * diff * r_inv[:, :, np.newaxis]
    grad = np.sum(term, axis=1)
    
    # Symmetry check: sum(grad) should be 0.
    # grad[j] gets -term[i,j] effectively? 
    # term[j,i] = de_dr[j,i] * (x_j - x_i) * r_inv[j,i]
    # de_dr[j,i] = de_dr[i,j] (symmetric)
    # r_inv[j,i] = r_inv[i,j]
    # (x_j - x_i) = -(x_i - x_j)
    # So term[j,i] = -term[i,j].
    # Summing term[i,:] for i gives grad_i.
    # This is correct.
    
    return energy, grad

def _generate_bcc(n):
    """
    Generate a BCC-like cluster of n atoms.
    We pick points from a BCC lattice in a sphere of appropriate radius.
    """
    # Approximate radius for n atoms in a cluster with density ~1
    # Volume ~ n, Radius ~ (3n/4pi)^(1/3)
    # For BCC, packing fraction is higher.
    # Let's just generate a grid and pick the n closest to origin.
    
    # Side length of cube
    L = int(np.ceil(n ** (1/3) * 1.5)) + 2
    coords = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            for z in range(-L, L+1):
                # BCC points: integer lattice + (0.5, 0.5, 0.5)
                coords.append([x, y, z])
                coords.append([x + 0.5, y + 0.5, z + 0.5])
    
    coords = np.array(coords)
    # Sort by distance to origin
    dists = np.linalg.norm(coords, axis=1)
    idx = np.argsort(dists)
    selected = coords[idx[:n]]
    
    # Center of mass to 0
    selected -= np.mean(selected, axis=0)
    
    return selected

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Initial guess: BCC
    pos = _generate_bcc(size)
    
    # Initial local minimization
    x0 = pos.flatten()
    
    def fun(x):
        p = x.reshape(size, 3)
        e, g = _morse_energy_and_grad(p, parameter)
        return e, g
    
    # Use L-BFGS-B
    # We don't have constraints easily in L-BFGS-B without bounds.
    # The problem states atoms must be at least 0.5 apart.
    # L-BFGS-B doesn't handle non-linear constraints like min distance well.
    # However, the Morse potential has a steep repulsive wall.
    # If we start from BCC (distances ~1.0), we are likely safe.
    # If the optimizer tries to push atoms closer, the energy explodes, providing a natural barrier.
    # We can add a small penalty if distance < 0.5 to be safe, but the Morse potential
    # at r=0.5 is exp(rho*0.5)(exp(rho*0.5)-2). For rho=10, exp(5) ~ 148, so E ~ 148*146 ~ 21600.
    # This is huge compared to ground state (~ -n). So the optimizer will avoid it.
    
    best_energy = np.inf
    best_pos = None
    
    # Perform initial relaxation
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-5})
        if res.success or res.fun < best_energy:
            best_energy = res.fun
            best_pos = res.x.reshape(size, 3)
    except Exception:
        pass
    
    if best_pos is None:
        best_pos = pos
        best_energy = fun(pos.flatten())[0]
    
    # Basin Hopping Loop
    rng = np.random.default_rng(42)
    
    # Perturbation size
    # Start with moderate perturbation
    sigma = 0.5
    
    while time.time() - _START_TIME < _TIME_LIMIT:
        # Perturb the best position found so far
        # Or perturb the current state? 
        # Standard BH: perturb current, minimize, accept if better (or Metropolis).
        # For finding global min, "Perturb Best" is a form of random restart from good basins.
        # "Perturb Current" with Metropolis allows exploring the landscape of local minima.
        # Let's use Perturb Current with Metropolis-like acceptance (accept if better, or with prob exp(-dE/T)).
        # But we don't have a temperature schedule easily. 
        # Simple strategy: Perturb current, minimize. If new energy < best energy, update best.
        # If new energy < current energy, update current.
        # This is a simple stochastic local search.
        
        current_pos = best_pos # Start exploring from the best found so far to stay in good basins
        # Actually, to escape local minima, we should perturb the *current* state, 
        # and if we find a better one, it becomes the new current/best.
        # If we only perturb the best, we might get stuck in the best basin.
        # Let's maintain a `current` state.
        
        # Let's restructure:
        # 1. current = best_pos
        # 2. Loop:
        #    a. perturb current
        #    b. minimize
        #    c. if e_new < e_current: current = new
        #    d. if e_new < e_best: best = new
        #    e. occasional "kick" to random to escape? Or just rely on perturbation.
        
        # Let's stick to a robust pattern:
        # We have `best_pos`. We generate a perturbation of `best_pos`.
        # We minimize it.
        # If the result is better than `best_pos`, we update `best_pos`.
        # This is "Best-of-N" from the current best. It converges to a local minimum of the "basin landscape".
        # To escape the global best basin, we need to occasionally perturb more strongly or restart.
        
        # Let's use a simple schedule for sigma.
        # Start with sigma = 0.5, decrease slowly? Or keep constant?
        # Constant sigma is fine for BH if we have enough time.
        
        perturbed = best_pos + rng.normal(0, sigma, best_pos.shape)
        
        try:
            res = minimize(fun, perturbed.flatten(), jac=True, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-4})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x.reshape(size, 3)
                # If we found a new best, we might want to reduce sigma to refine?
                # Or keep it.
        except Exception:
            pass
        
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
            
    return best_pos
# EVOLVE-BLOCK-END
