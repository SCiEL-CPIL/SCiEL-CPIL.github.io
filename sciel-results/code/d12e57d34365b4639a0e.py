"""
Strategy: Simulated Annealing with L-BFGS-B local minimization.
Rationale: Dzugutov potential clusters have rugged energy landscapes with many local minima.
Simple geometric templates + single relaxation (current approach) often get stuck.
SA allows escape from local minima via thermal fluctuations.
L-BFGS-B is a robust gradient-based optimizer suitable for the smooth regions between minima.
We perform a sequence of SA runs with different random seeds and temperature schedules.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0 # Leave 10s buffer for overhead

# Dzugutov Potential Parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _energy_and_grad(pos):
    """
    Compute energy and gradient for Dzugutov potential.
    pos: (n, 3) array
    Returns: energy (float), grad (n, 3) array
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Compute pairwise distances
    # Using broadcasting for efficiency
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Mask for valid pairs (i < j) to avoid double counting and self-interaction
    # We need to be careful with the gradient calculation.
    # It is often safer to loop over pairs for gradient if n is small (< 100)
    # n is 38-60, so O(n^2) loop is ~1800 pairs, which is fast.
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Precompute masks for efficiency in loop
    # V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    # Note: The potential definition in the prompt has a discontinuity or specific ranges.
    # r < a: V1
    # a <= r < b: V2? 
    # Prompt says: V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    # This implies:
    # if r < a: V = A(r^-m - B)exp(c/(r - a)) + B exp(d/(r - b))
    # if a <= r < b: V = B exp(d/(r - b))
    # if r >= b: V = 0
    
    # Let's verify the derivative.
    # For r < a:
    # V1 = A * r^-m * exp(c/(r-a)) - A*B * exp(c/(r-a))
    # dV1/dr = A * (-m*r^(-m-1) * exp(c/(r-a)) + r^-m * exp(c/(r-a)) * (-c/(r-a)^2)) - A*B * exp(c/(r-a)) * (-c/(r-a)^2)
    #        = A * exp(c/(r-a)) * [ -m*r^(-m-1) - c*r^(-m)/(r-a)^2 + B*c/(r-a)^2 ]
    
    # For a <= r < b:
    # V2 = B * exp(d/(r-b))
    # dV2/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
    
    # Gradient force on atom i from atom j is -dV/dr * (r_ij / r)
    # Total gradient on i is sum over j != i of force_ij
    
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < 1e-6: # Avoid division by zero
                # Randomly perturb if too close
                grad[i] += np.random.normal(0, 1, 3) * 100
                grad[j] -= np.random.normal(0, 1, 3) * 100
                continue
            
            if r < a:
                exp_c = np.exp(c / (r - a))
                exp_d = np.exp(d / (r - b))
                
                # Energy
                e1 = A * (r ** (-m) - B) * exp_c
                e2 = B * exp_d
                energy += e1 + e2
                
                # Derivative
                # dV1/dr
                term1 = -m * r ** (-m - 1)
                term2 = -c * r ** (-m) / (r - a) ** 2
                term3 = B * c / (r - a) ** 2
                dV1 = A * exp_c * (term1 + term2 + term3)
                
                # dV2/dr
                dV2 = -B * d * exp_d / (r - b) ** 2
                
                dVdr = dV1 + dV2
                
            elif r < b:
                exp_d = np.exp(d / (r - b))
                energy += B * exp_d
                dVdr = -B * d * exp_d / (r - b) ** 2
            else:
                continue
            
            # Force on i due to j
            # F_ij = -dVdr * (r_ij / r)
            # r_ij = pos[i] - pos[j]
            r_vec = pos[i] - pos[j]
            force = -dVdr * (r_vec / r)
            
            grad[i] += force
            grad[j] -= force
            
    return energy, grad

def _lbfgs_minimize(pos):
    """
    Perform local minimization using L-BFGS-B.
    Returns: (energy, positions)
    """
    def func_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _energy_and_grad(pos)
        return e, g.flatten()
    
    try:
        res = minimize(
            func_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True, # func_and_grad returns (f, g)
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success:
            return res.fun, res.x.reshape(-1, 3)
        else:
            # Even if not successful, use the best point found
            pos_best = res.x.reshape(-1, 3)
            e_best, _ = _energy_and_grad(pos_best)
            return e_best, pos_best
    except Exception:
        return np.inf, pos

def _simulated_annealing(n, seed, t_max=5.0, t_min=0.1, cooling=0.99, steps_per_temp=20):
    """
    Run Simulated Annealing.
    """
    rng = np.random.default_rng(seed)
    
    # Initial guess: Random points in a compact sphere
    # Radius roughly estimated from ideal density
    # Dzugutov clusters are compact.
    # Volume ~ n * (1.1)^3 ?
    radius = 0.8 * n ** (1.0/3.0)
    pos = rng.uniform(-radius, radius, (n, 3))
    
    # Ensure minimum distance constraint is roughly respected by relaxation
    # If two atoms are too close, push them apart
    for _ in range(100):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist > 0.5:
            break
        # Push apart
        i, j = min_pair
        mid = (pos[i] + pos[j]) / 2
        diff = pos[i] - pos[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            diff = rng.normal(0, 1, 3)
            norm = np.linalg.norm(diff)
        diff = diff / norm
        pos[i] = mid + diff * 0.3
        pos[j] = mid - diff * 0.3
        
    # Relax initial guess
    e_cur, pos_cur = _lbfgs_minimize(pos)
    
    t = t_max
    best_e = e_cur
    best_pos = pos_cur.copy()
    
    while t > t_min:
        for _ in range(steps_per_temp):
            # Propose a move: random displacement
            # Displacement size should scale with temperature
            disp_mag = 0.1 * t
            move = rng.normal(0, disp_mag, (n, 3))
            pos_new = pos_cur + move
            
            # Relax the new position
            e_new, pos_new_relaxed = _lbfgs_minimize(pos_new)
            
            # Acceptance criterion
            delta_e = e_new - e_cur
            if delta_e < 0 or rng.random() < np.exp(-delta_e / t):
                pos_cur = pos_new_relaxed
                e_cur = e_new
                if e_cur < best_e:
                    best_e = e_cur
                    best_pos = pos_cur.copy()
        
        t *= cooling
        
    return best_e, best_pos

def construct(instance):
    """
    Find the lowest-energy structure for a Dzugutov cluster of size n.
    Uses multiple SA runs with different seeds and time management.
    """
    n = instance['n']
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    # We will run SA with different seeds.
    # Each SA run involves many L-BFGS calls.
    # We need to estimate how many full SA runs we can afford.
    # A single SA run with t_max=5.0, t_min=0.1, cooling=0.99:
    # Number of temps = log(0.1/5.0) / log(0.99) ~ 300
    # Steps per temp = 20 -> 6000 L-BFGS calls.
    # This is likely too slow for 300s.
    # We need a faster SA or fewer steps.
    
    # Let's optimize the SA parameters for speed.
    # Reduce steps_per_temp and use a faster cooling.
    # Or, use a "basin hopping" approach where we only do 1 L-BFGS per move?
    # That's what we are doing.
    
    # Let's estimate cost of _lbfgs_minimize.
    # n=50, 100 iterations max.
    # Each iteration: O(n^2) energy/grad calc.
    # 50^2 = 2500 pairs.
    # 100 iters * 2500 pairs = 250k pair calcs.
    # In Python, this might take ~1-5 ms per L-BFGS call?
    # If 1 SA run takes 6000 calls * 2ms = 12 seconds.
    # We have ~290 seconds. So we can afford ~20 SA runs.
    
    # However, L-BFGS might converge faster or slower.
    # Let's start with a few runs and check time.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # Adjust SA parameters based on remaining time?
        # For now, fixed parameters.
        # To be safe, let's make the SA shorter initially and see.
        # t_max=2.0, t_min=0.5, cooling=0.95, steps=10
        # Temps: log(0.25)/log(0.95) ~ 14 temps.
        # Steps: 14 * 10 = 140 L-BFGS calls.
        # This is much faster. ~0.5s per run.
        # We can run 500 runs.
        
        # Let's try a middle ground.
        # t_max=1.0, t_min=0.1, cooling=0.98, steps=5
        # Temps: log(0.1)/log(0.98) ~ 114 temps.
        # Steps: 114 * 5 = 570 calls.
        # ~1-2s per run.
        # ~150 runs possible.
        
        try:
            e, pos = _simulated_annealing(
                n, 
                seed, 
                t_max=1.5, 
                t_min=0.1, 
                cooling=0.97, 
                steps_per_temp=5
            )
            
            if e < best_energy:
                best_energy = e
                best_positions = pos
                
        except Exception as e:
            # If SA fails, try a simple random relaxation
            rng = np.random.default_rng(seed)
            radius = 0.8 * n ** (1.0/3.0)
            pos_rand = rng.uniform(-radius, radius, (n, 3))
            try:
                e_rand, pos_relaxed = _lbfgs_minimize(pos_rand)
                if e_rand < best_energy:
                    best_energy = e_rand
                    best_positions = pos_relaxed
            except:
                pass

    # If we still don't have a result (shouldn't happen), fallback
    if best_positions is None:
        rng = np.random.default_rng(0)
        radius = 0.8 * n ** (1.0/3.0)
        pos_rand = rng.uniform(-radius, radius, (n, 3))
        best_energy, best_positions = _lbfgs_minimize(pos_rand)
        
    return best_positions
# EVOLVE-BLOCK-END
