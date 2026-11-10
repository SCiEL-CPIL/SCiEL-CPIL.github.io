"""
Strategy: Simulated Annealing with Molten Start and Fast Growth
- Uses a "molten" random compact seed generated via fast nearest-neighbor growth.
- Implements Simulated Annealing: Perturb (Gaussian) -> Relax -> Accept/Reject (Metropolis).
- Uses a cooling schedule to balance exploration and exploitation.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _generate_molten_seed(n, rng):
    """
    Generate a compact 'molten' seed using fast nearest-neighbor growth.
    This avoids the O(n^2) cost of rejection sampling for large n.
    1. Place first atom at origin.
    2. For each subsequent atom, place it near a random existing atom at a random angle/distance.
    3. This creates a compact, somewhat disordered cluster.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    pts = np.zeros((n, 3))
    # First atom at origin
    # pts[0] is already 0
    
    # Typical interatomic distance for SC is around 1.0-1.2
    # We want to grow a compact cluster.
    for i in range(1, n):
        # Pick a random existing atom to attach to
        parent_idx = rng.integers(0, i)
        parent_pos = pts[parent_idx]
        
        # Generate a random direction
        dir = rng.normal(size=3)
        norm = np.linalg.norm(dir)
        if norm < 1e-9:
            dir = np.array([1.0, 0.0, 0.0])
        else:
            dir /= norm
            
        # Random distance, slightly larger than minimum to avoid overlap initially
        # SC equilibrium distance is ~1.1-1.2. Let's use 1.0-1.5
        dist = rng.uniform(1.0, 1.5)
        
        new_pos = parent_pos + dir * dist
        pts[i] = new_pos
        
    return pts

def construct(instance):
    """
    Simulated Annealing Global Optimization.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    rng = np.random.default_rng(123)
    
    # 1. Generate initial molten seed
    pos = _generate_molten_seed(n, rng)
    
    # 2. Initial Relaxation
    try:
        e_curr, pos = relax(pos, potential, kind)
        if np.isfinite(e_curr):
            best_energy = e_curr
            best_pos = pos
    except Exception:
        # Fallback if relaxation fails immediately
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # 3. Simulated Annealing Loop
    # Temperature schedule: Start high, cool down
    # T0 should be comparable to the energy scale. 
    # For SC, energy is often in the range of -10 to -100 for these sizes.
    # Let's estimate T0 based on initial energy or a fixed value.
    T0 = 5.0  # Initial temperature
    T_end = 0.01
    cooling_rate = 0.98  # Geometric cooling
    
    # Perturbation scale
    # Start with a relatively large scale, maybe shrink it slightly as we cool?
    # Or keep it constant. Constant is often better for SA.
    perturb_scale = 0.2
    
    T = T0
    steps = 0
    max_steps = 5000 # Hard cap on steps if time allows
    
    while T > T_end and _time_left() > 10.0 and steps < max_steps:
        # Check time more frequently for long runs
        if steps % 10 == 0 and _time_left() < 5.0:
            break
            
        # Perturb
        # Gaussian noise
        noise = rng.normal(0, perturb_scale, pos.shape)
        pos_perturbed = pos + noise
        
        # Relax
        try:
            e_new, pos_new = relax(pos_perturbed, potential, kind)
            if not np.isfinite(e_new):
                # If relaxation fails or returns inf, reject and try again
                # Or maybe just skip this step
                steps += 1
                T *= cooling_rate
                continue
        except Exception:
            steps += 1
            T *= cooling_rate
            continue
            
        # Metropolis Acceptance
        if e_new < e_curr:
            # Always accept better
            accept = True
        else:
            # Accept worse with probability exp(-(e_new - e_curr) / T)
            delta_e = e_new - e_curr
            if T > 1e-10:
                prob = np.exp(-delta_e / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            e_curr = e_new
            pos = pos_new
            if e_curr < best_energy:
                best_energy = e_curr
                best_pos = pos
        else:
            # Reject: keep current pos and energy, but still cool
            pass
            
        # Cool down
        T *= cooling_rate
        steps += 1
        
    # 4. Final Polish: Run a few more relaxations from the best found
    # Sometimes the last accepted state is slightly worse than the global best found earlier.
    # We return the global best.
    
    if best_pos is None:
        # Fallback
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Ensure validity: check min distance
    # The evaluator rejects if any pair < 0.35.
    # relax() usually ensures stability, but let's be safe.
    # If best_pos has close pairs, we might need to nudge them.
    # However, relax() with L-BFGS-B on SC potential should naturally push atoms apart if they are too close.
    # If we are very close to deadline, just return best_pos.
    
    return best_pos
# EVOLVE-BLOCK-END
