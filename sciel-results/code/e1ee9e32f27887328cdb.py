import numpy as np
from scipy.optimize import minimize
import time
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

START_TIME = time.time()
TOTAL_TIME_LIMIT = 290.0

def get_icosahedron_seed(n):
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    scale = 1.0 / np.sqrt(1 + phi**2)
    v = v * scale
    core = np.vstack([np.array([0, 0, 0]), v])
    
    if n <= 13:
        return core[:n]
    
    rng = np.random.default_rng(0)
    extra_atoms = []
    for _ in range(n - 13):
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        p = np.array([x, y, z]) * 1.1
        extra_atoms.append(p)
    extra_atoms = np.array(extra_atoms)
    return np.vstack([core, extra_atoms])

def local_minimize(positions):
    def fun_and_grad(x):
        e, g = dzugutov_energy_gradient(x)
        return e, g
        
    res = minimize(
        fun_and_grad,
        positions,
        jac=True, # Since fun_and_grad returns both
        method='L-BFGS-B'
    )
    return res.fun, res.x

def construct(instance):
    n = instance['n']
    
    elapsed = time.time() - START_TIME
    remaining = TOTAL_TIME_LIMIT - elapsed
    if remaining < 5:
        return get_icosahedron_seed(n)
    
    time_budget = min(45.0, remaining * 0.5)
    
    seed = get_icosahedron_seed(n)
    
    # Initial relaxation
    try:
        best_energy, best_pos = local_minimize(seed)
    except:
        best_energy, best_pos = float('inf'), seed
    
    start_bh = time.time()
    rng = np.random.default_rng(123)
    
    while time.time() - start_bh < time_budget:
        kick = rng.normal(0, 0.3, best_pos.shape)
        perturbed = best_pos + kick
        
        try:
            e_new, pos_new = local_minimize(perturbed)
        except:
            continue
            
        if e_new < best_energy:
            best_energy = e_new
            best_pos = pos_new
    
    return best_pos
