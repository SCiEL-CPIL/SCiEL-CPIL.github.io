import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

N = 100
DEADLINE = 145.0  # Safety margin from 150s limit

def _compute_smooth_obj_grad(c, alpha):
    """
    Computes the smooth log-sum-exp objective and gradient.
    Objective: f(c) = (1/alpha) * log( sum_{s} exp(-alpha * s) )
    Minimizing f(c) maximizes the minimum distance s.
    """
    # Pairwise distances
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    # Wall distances
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Numerical stability
    m = np.min(all_dists)
    shifted = all_dists - m
    exp_terms = np.exp(-alpha * shifted)
    Z = np.sum(exp_terms)
    
    if Z == 0:
        return 0.0, np.zeros_like(c)
        
    obj_val = (1.0 / alpha) * (np.log(Z) - alpha * m)
    
    # Gradient
    weights = exp_terms / Z  # Sum to 1
    
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    pair_dists_raw = d[i_idx, j_idx]
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj  # (4950, 3)
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms  # Unit vector j -> i
    
    w_pair = weights[:len(pair_dists)]
    w_pair = w_pair[:, np.newaxis]
    
    contrib = (0.5 * w_pair * u)
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall part
    w_wall = weights[len(pair_dists):]
    ww = w_wall.reshape(6, N)
    
    grad_c[:, 0] += (ww[0] - ww[1])
    grad_c[:, 1] += (ww[2] - ww[3])
    grad_c[:, 2] += (ww[4] - ww[5])
    
    return obj_val, grad_c

def _score(c):
    """Compute the exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(c > 0.5) or np.any(c < -0.5):
        return 0.0
    
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))

def _initial_jittered_grid(seed):
    """Create a structured grid with small jitter, tuned for better initial packing."""
    rng = np.random.RandomState(seed)
    # Use a slightly tighter grid to start closer to optimal spacing
    # 5x5x4 = 100 spheres
    xs = np.linspace(-0.40, 0.40, 5)
    ys = np.linspace(-0.40, 0.40, 5)
    zs = np.linspace(-0.36, 0.36, 4)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                jitter = rng.uniform(-0.015, 0.015, 3)
                centres.append([x, y, z] + jitter)
    
    return np.array(centres[:N])

def _initial_random(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.42, 0.42, (N, 3))

def _lj_energy_grad(c, r_target=0.111):
    """
    Lennard-Jones energy and gradient with a penalty for going outside the cube.
    We use a soft repulsion that mimics LJ but with a cutoff at 2^(1/6)*r_target
    where LJ is zero, and a penalty for walls.
    """
    r_c6 = r_target ** 6
    r_c12 = r_target ** 12
    
    # Pairwise LJ energy: sum_{i<j} [ r_c12/rij^12 - 2*r_c6/rij^6 ]
    # This is minimized when rij = 2^(1/6) * r_target
    # We actually want to maximize minimum distance, so we want to push pairs apart
    # beyond the equilibrium. A better model for packing is pure repulsion:
    # E = sum_{i<j} 1/rij^p  or  E = sum (rij - d_min)^2 * I(rij < d_min)
    
    # Let's use a smooth potential that is purely repulsive for rij < 2*r_target
    # E_pair = max(0, 2*r_target - rij)^4  (soft sphere)
    # This is smooth and pushes apart if overlapping or close.
    
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx]
    
    # Soft sphere potential
    # V = (max(0, 2r - d))^4
    # dV/dd = -4*(2r-d)^3 * I(d < 2r)
    # Gradient wrt ci: dV/dd * (ci - cj)/d = -4*(2r-d)^3 * (ci-cj)/d * I(d<2r)
    
    energy = 0.0
    grad_c = np.zeros_like(c)
    
    mask = pair_dists < 2 * r_target
    if np.any(mask):
        dm = pair_dists[mask]
        # Avoid division by zero
        dm_safe = np.where(dm < 1e-10, 1e-10, dm)
        
        # V = (2r - dm)^4
        term = (2 * r_target - dm_safe) ** 4
        energy = np.sum(term)
        
        # dV/d(ci) = -4*(2r - dm)^3 * (ci - cj) / dm
        ci = c[i_idx[mask]]
        cj = c[j_idx[mask]]
        diff = ci - cj
        factor = -4 * (2 * r_target - dm_safe) ** 3 / dm_safe[:, np.newaxis]
        g_ci = factor * diff
        
        i_masked = i_idx[mask]
        j_masked = j_idx[mask]
        
        np.add.at(grad_c, i_masked, g_ci)
        np.add.at(grad_c, j_masked, -g_ci)
    
    # Wall penalty: P = sum (max(0, |cik| - 0.5 + epsilon))^2
    # Or simpler: push towards center if close to wall
    # Let's use a quadratic penalty for exceeding bounds
    # P = sum_k max(0, |c_ik| - (0.5 - r_target))^2
    # This creates a "soft wall" at distance r_target from boundary
    
    wall_margin = 0.5 - r_target
    abs_c = np.abs(c)
    excess = np.maximum(0, abs_c - wall_margin)
    wall_energy = np.sum(excess ** 2)
    energy += wall_energy
    
    # Gradient of wall penalty
    # dP/d(ci) = 2 * excess * sign(ci) * I(|ci| > wall_margin)
    grad_wall = 2 * excess * np.sign(c)
    grad_c += grad_wall
    
    return energy, grad_c

def _optimize_lj(c_init, time_limit, start_time):
    """
    Optimize using Lennard-Jones / Soft-Sphere energy.
    """
    best_c = c_init.copy()
    best_score = _score(c_init)
    current_c = c_init.copy()
    
    # We can try to tune the target radius during optimization
    # Start with a slightly smaller target to avoid large initial gradients
    r_targets = [0.08, 0.09, 0.10, 0.105, 0.108, 0.110, 0.111, 0.1113]
    
    for r_t in r_targets:
        if time.time() - start_time > time_limit:
            break
            
        def obj(x):
            c_flat = x.reshape(N, 3)
            val, _ = _lj_energy_grad(c_flat, r_t)
            return val
            
        def grad(x):
            c_flat = x.reshape(N, 3)
            _, g = _lj_energy_grad(c_flat, r_t)
            return g.flatten()
            
        try:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 2:
                break
                
            maxiter = int(min(1000, max(50, remaining_time * 10)))
            
            result = minimize(
                obj,
                current_c.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (3 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
                current_c = c_new.copy()
                
        except Exception:
            pass
            
    return best_c

def _optimize_with_annealing(c_init, time_limit, start_time, alphas=None):
    """
    Run the annealing optimization from a given start.
    """
    if alphas is None:
        alphas = [5, 10, 20, 40, 80, 150, 300, 600, 1200, 2500, 5000]
        
    best_c = c_init.copy()
    best_score = _score(c_init)
    current_c = c_init.copy()
    stall_count = 0
    stall_threshold = 3
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
            
        def obj(x):
            c_flat = x.reshape(N, 3)
            val, _ = _compute_smooth_obj_grad(c_flat, alpha)
            return val
            
        def grad(x):
            c_flat = x.reshape(N, 3)
            _, g = _compute_smooth_obj_grad(c_flat, alpha)
            return g.flatten()
            
        x0 = current_c.flatten()
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                break
                
            if alpha < 100:
                maxiter = int(min(1000, max(100, remaining_time * 15)))
            else:
                maxiter = int(min(500, max(50, remaining_time * 8)))
            
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
                current_c = c_new.copy()
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= stall_threshold:
                    noise_scale = 0.01 + 0.01 * stall_count
                    noise = np.random.normal(0, noise_scale, (N, 3))
                    c_kicked = best_c + noise
                    c_kicked = np.clip(c_kicked, -0.5, 0.5)
                    
                    kick_alpha = max(5, alpha // 2)
                    
                    def obj_kick(x):
                        c_flat = x.reshape(N, 3)
                        val, _ = _compute_smooth_obj_grad(c_flat, kick_alpha)
                        return val
                        
                    def grad_kick(x):
                        c_flat = x.reshape(N, 3)
                        _, g = _compute_smooth_obj_grad(c_flat, kick_alpha)
                        return g.flatten()
                        
                    if time.time() - start_time < time_limit - 10:
                        res_kick = minimize(
                            obj_kick,
                            c_kicked.flatten(),
                            jac=grad_kick,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        c_kicked_opt = res_kick.x.reshape(N, 3)
                        c_kicked_opt = np.clip(c_kicked_opt, -0.5, 0.5)
                        score_kicked = _score(c_kicked_opt)
                        
                        if score_kicked > best_score:
                            best_score = score_kicked
                            best_c = c_kicked_opt.copy()
                            current_c = c_kicked_opt.copy()
                            stall_count = 0
                        else:
                            current_c = best_c.copy()
                else:
                    if result.fun < obj(x0) - 1e-12:
                        current_c = c_new.copy()
                    else:
                        current_c = best_c.copy()
                        
        except Exception:
            pass
            
    return best_c

def _epigraph_polish(c_init, time_limit, start_time):
    """
    Final high-precision polish using an epigraph formulation with auxiliary variable t.
    """
    best_c = c_init.copy()
    best_score = _score(c_init)
    current_c = c_init.copy()
    
    alphas_polish = [3000, 8000, 20000, 50000]
    
    for alpha in alphas_polish:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            c_flat = x.reshape(N, 3)
            val, _ = _compute_smooth_obj_grad(c_flat, alpha)
            return val
            
        def grad(x):
            c_flat = x.reshape(N, 3)
            _, g = _compute_smooth_obj_grad(c_flat, alpha)
            return g.flatten()
            
        try:
            remaining_time = time_limit - time.time()
            if remaining_time < 2:
                break
                
            maxiter = int(min(2000, max(200, remaining_time * 20)))
            
            result = minimize(
                obj,
                current_c.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (3 * N),
                options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
                current_c = c_new.copy()
            else:
                current_c = best_c.copy()
                
        except Exception:
            pass
            
    return best_c

def construct():
    start_time = time.time()
    
    best_c = None
    best_score = 0.0
    
    # Strategy:
    # 1. Try Lennard-Jones / Soft-Sphere approach from random starts.
    # 2. Try the smooth annealing approach from grid starts.
    # 3. Polish the best result.
    
    # Start 1: LJ from Random
    if time.time() < DEADLINE - 60:
        c_rand = _initial_random(123)
        t_lj = start_time + (DEADLINE - start_time) * 0.35
        c_lj = _optimize_lj(c_rand, t_lj, start_time)
        s_lj = _score(c_lj)
        if s_lj > best_score:
            best_score = s_lj
            best_c = c_lj

    # Start 2: Annealing from Grid
    if time.time() < DEADLINE - 50:
        c_grid = _initial_jittered_grid(1)
        t1_limit = start_time + (DEADLINE - start_time) * 0.55
        c1 = _optimize_with_annealing(c_grid, t1_limit, start_time)
        s1 = _score(c1)
        if s1 > best_score:
            best_score = s1
            best_c = c1

    # Start 3: LJ from a different random seed (if time permits)
    if time.time() < DEADLINE - 40 and best_c is not None:
        # Use best_c as a good starting point for another LJ run with higher target?
        # Or try another random?
        # Let's try another random start with LJ, but with less time.
        c_rand2 = _initial_random(456)
        t_lj2 = start_time + (DEADLINE - start_time) * 0.75
        c_lj2 = _optimize_lj(c_rand2, t_lj2, start_time)
        s_lj2 = _score(c_lj2)
        if s_lj2 > best_score:
            best_score = s_lj2
            best_c = c_lj2

    # Start 4: Annealing from Random
    if time.time() < DEADLINE - 20:
        c_rand3 = _initial_random(789)
        t2_limit = start_time + (DEADLINE - start_time) * 0.90
        c2 = _optimize_with_annealing(c_rand3, t2_limit, start_time)
        s2 = _score(c2)
        if s2 > best_score:
            best_score = s2
            best_c = c2

    # Final Polish
    if time.time() < DEADLINE - 5 and best_c is not None:
        c_polish = _epigraph_polish(best_c, DEADLINE - 1, start_time)
        s_polish = _score(c_polish)
        if s_polish > best_score:
            best_score = s_polish
            best_c = c_polish

    if best_c is None:
        best_c = _initial_jittered_grid(1)

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
