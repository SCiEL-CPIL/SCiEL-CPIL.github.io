import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient of E = sum log(1/d) = - sum log(d)
    # d/dx_i log(1/|x_i - x_j|) = - 1/|x_i - x_j|^2 * (x_i - x_j)
    # Sum over j
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: The formula in the previous code was:
    # grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    # Let's verify:
    # weighted_sum_pts = sum_{j != i} inv_d2[i,j] * x_j
    # sum_inv_d2 = sum_{j != i} inv_d2[i,j]
    # grad_i = - x_i * sum_{j} inv_d2[i,j] + sum_{j} inv_d2[i,j] * x_j = sum_{j} inv_d2[i,j] (x_j - x_i)
    # This matches - sum (x_i - x_j) / d^2.
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_cg(pts, time_limit=10.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    prev_grad = grad.copy()
    
    # Initial search direction
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Backtracking line search with Armijo condition
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Sufficient decrease (Armijo)
            if new_energy <= energy + c1 * alpha * np.sum(grad * d):
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
                break
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            alpha *= 0.5
            
            if alpha < 1e-12:
                break
                
        if best_alpha == 0.0:
            break
            
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Polak-Ribiere beta
        d_new = -grad
        beta = 0.0
        if it > 0:
            num = np.sum(grad * (grad - prev_grad))
            den = np.sum(prev_grad * prev_grad)
            if den > 1e-30:
                beta = max(0.0, num / den)
        else:
            beta = 0.0
            
        d = d_new + beta * d
        prev_grad = grad.copy()
        
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left_init = time_limit - (time.time() - start) - 3
    if time_left_init > 3:
         current, energy = _riemannian_cg(current, time_limit=min(12, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.92
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2:
            break
            
        # Perturb with varying scale based on temperature
        rng = np.random.default_rng()
        perturbations = rng.standard_normal((N, 3))
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        # Scale by temperature - use sqrt for better exploration
        perturb_scale = 0.15 * math.sqrt(temp)
        new_pts = _exponential_map(current, perturbations * perturb_scale)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        time_left = time_limit - (time.time() - start) - 1
        if time_left < 4:
             break
             
        new_opt_pts, new_energy = _riemannian_cg(new_pts, time_limit=min(8, time_left))
        new_energy = _log_energy_fast(new_opt_pts)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or (temp > 0.01 and np.random.rand() < math.exp(-delta_e / temp)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
            
        temp *= decay
        
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        # Map to sphere
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _bfgs_sphere(pts, time_limit=10.0, max_iter=200):
    """BFGS quasi-Newton on the sphere with projection line search."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initialize inverse Hessian approximation as identity
    n = len(pts)
    H = np.eye(3 * n)
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction
        p = -H @ grad.flatten()
        p = p.reshape(n, 3)
        
        # Project to tangent space
        dot_prod = np.sum(p * current, axis=1, keepdims=True)
        p -= dot_prod * current
        
        # Line search with sphere projection
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        for ls in range(40):
            # Apply step and project to sphere
            new_pts = _exponential_map(current, alpha * p)
            new_pts = _normalize(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy > energy + c1 * alpha * np.sum(grad * p):
                break
            
            alpha *= 0.6
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            break
        
        # Update inverse Hessian
        s = (best_pts - current).flatten()
        y = (best_grad - grad).flatten()
        sy = np.dot(s, y)
        
        if sy > 1e-10:
            Hs = H @ s
            Hy = H @ y
            rho = 1.0 / sy
            # BFGS update: H_new = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
            I = np.eye(3 * n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
    
    return current, energy

def _evolve_population(pop, time_limit, relax_time=2.0):
    """
    Evolutionary Strategy for global optimization.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    pop_size = len(pop)
    
    # Evaluate initial population
    energies = []
    for i in range(pop_size):
        # Relax each individual to a local minimum
        pts, e = _riemannian_cg(pop[i], time_limit=relax_time)
        e = _log_energy_fast(pts)
        pop[i] = pts
        energies.append(e)
    
    best_idx = np.argmin(energies)
    best_pts = pop[best_idx].copy()
    best_energy = energies[best_idx]
    
    generation = 0
    while time.time() < deadline - 10: # Keep 10s for final polish
        generation += 1
        # Selection: Keep top 20% (elitism)
        k = max(2, pop_size // 5)
        sorted_indices = np.argsort(energies)
        elites = [pop[i].copy() for i in sorted_indices[:k]]
        
        # Create new population
        new_pop = []
        new_energies = []
        
        # Add elites
        for i in range(k):
            new_pop.append(elites[i])
            new_energies.append(energies[sorted_indices[i]])
        
        # Create offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 15:
                break
                
            # Crossover: Uniform crossover from two random parents
            i1 = np.random.randint(0, pop_size)
            i2 = np.random.randint(0, pop_size)
            
            # Pick points from either parent
            mask = np.random.rand(N) > 0.5
            child = np.where(mask[:, np.newaxis], pop[i1], pop[i2])
            
            # Mutation: Add random noise
            noise = np.random.standard_normal((N, 3))
            # Project noise to tangent space of child
            dot_prods = np.sum(noise * child, axis=1, keepdims=True)
            noise -= dot_prods * child
            
            # Scale mutation
            scale = 0.1
            child = _exponential_map(child, noise * scale)
            child = _normalize(child)
            
            # Relax the child
            remaining = deadline - time.time() - 15
            if remaining < 1.0:
                break
                
            child_relaxed, e_child = _riemannian_cg(child, time_limit=min(1.5, remaining / 10))
            e_child = _log_energy_fast(child_relaxed)
            
            new_pop.append(child_relaxed)
            new_energies.append(e_child)
            
            # Update global best
            if e_child < best_energy:
                best_energy = e_child
                best_pts = child_relaxed.copy()
        
        # Replace population
        # Ensure we have enough members
        while len(new_pop) < pop_size:
            # Fill with mutated elites
            e_idx = np.random.randint(0, len(elites))
            child = elites[e_idx].copy()
            noise = np.random.standard_normal((N, 3))
            dot_prods = np.sum(noise * child, axis=1, keepdims=True)
            noise -= dot_prods * child
            child = _exponential_map(child, noise * 0.05)
            child = _normalize(child)
            child_relaxed, e_child = _riemannian_cg(child, time_limit=1.0)
            e_child = _log_energy_fast(child_relaxed)
            new_pop.append(child_relaxed)
            new_energies.append(e_child)
            
            if e_child < best_energy:
                best_energy = e_child
                best_pts = child_relaxed.copy()
                
        pop = new_pop
        energies = new_energies
        
    return best_pts, best_energy

def _local_optimize(pts, time_limit):
    """Local optimization using Riemannian CG with tight tolerance."""
    if time_limit < 0.5:
        return pts, _log_energy_fast(pts)
    pts, energy = _riemannian_cg(pts, time_limit=time_limit, max_iter=500)
    energy = _log_energy_fast(pts)
    return pts, energy

def _basin_hopping_local(pts, time_limit):
    """Basin hopping with local optimization, focusing on finding better minima."""
    start = time.time()
    current = pts.copy()
    current, energy = _local_optimize(current, min(10, time_limit - 2))
    best_pts = current.copy()
    best_energy = energy
    
    temp = 0.3
    decay = 0.95
    max_restarts = 20
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 1:
            break
        
        # Perturb with temperature-dependent scale
        rng = np.random.default_rng()
        perturbations = rng.standard_normal((N, 3))
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        # Scale by temperature
        perturb_scale = 0.1 * math.sqrt(temp)
        new_pts = _exponential_map(current, perturbations * perturb_scale)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        time_left = time_limit - (time.time() - start) - 0.5
        if time_left < 2:
            break
            
        new_opt_pts, new_energy = _local_optimize(new_pts, min(8, time_left))
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or (temp > 0.01 and np.random.rand() < math.exp(-delta_e / temp)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
            
        temp *= decay
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations - diverse
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N, base_start=2))
    inits.append(_halton_sphere(N, base_start=3))
    
    # Random inits with different seeds
    for s in range(8):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci with different scales
    for s, scale in [(100, 0.05), (101, 0.1), (102, 0.2), (103, 0.3)]:
        rng = np.random.default_rng(s)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
    
    # Run basin hopping on each initialization
    best_pts = inits[0]
    best_energy = float('inf')
    
    # Allocate time based on number of inits
    n_inits = len(inits)
    time_per_init = (deadline - start_time - 10) / n_inits
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        time_left = deadline - time.time() - 10
        bh_time = min(time_per_init, time_left)
        
        if bh_time < 3:
            continue
            
        try:
            bh_pts, bh_energy = _basin_hopping_local(init_pts.copy(), bh_time)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pts = bh_pts
        except Exception:
            continue
    
    # Final Polish with more time
    if time.time() < deadline - 2:
        polish_time = deadline - time.time() - 2
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=1000)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
