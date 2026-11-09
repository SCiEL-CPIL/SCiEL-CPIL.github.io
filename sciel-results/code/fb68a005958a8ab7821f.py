"""Dzugutov cluster global minima via polytetrahedral geometric initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_DEADLINE = time.time() + 290.0  # safety margin before 300s kill

def _get_seed(n):
    """
    Generate a seed structure for n atoms.
    Uses a 13-atom icosahedron as the base.
    """
    # 13-atom icosahedron: 1 center, 12 vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Scale so that nearest neighbor distance is ~1.8-2.0 (optimal for Dzugutov)
    # Distance between [0,1,phi] and [1,phi,0] is sqrt(1 + (phi-1)^2) = sqrt(1+phi-2+phi^2) = sqrt(phi^2+phi-1)
    # phi^2 = phi+1, so sqrt(2phi) ~ 1.902
    # This is a good starting distance.
    center = np.zeros(3)
    
    if n == 13:
        return np.vstack([center, verts])
    
    # If n > 13, we start with the 13-atom core and add atoms later in the solver
    # For now, just return the 13-atom core. The solver will handle expansion.
    # However, for n < 13, we need to trim.
    if n < 13:
        # Take center and first n-1 vertices
        # This is a crude approximation for small n, but n>=38 in task
        return np.vstack([center, verts[:n-1]])
        
    return np.vstack([center, verts])

def _solve_instance_with_deadline(instance, deadline):
    """
    Solve a single instance with a specific deadline.
    """
    n = instance['n']
    potential = instance['potential']
    
    # 1. Generate Seed
    seed = _get_seed(13)
    
    # 2. Relax Seed
    try:
        # Try 2-arg version first as suggested by prompt
        energy_seed, core_pos = relax(seed, potential)
    except Exception:
        try:
            # Fallback to 3-arg version if 2-arg fails
            parameter = instance.get("rho", instance.get("kind"))
            energy_seed, core_pos = relax(seed, potential, parameter)
        except Exception:
            core_pos = seed
    
    best_energy = np.inf
    best_pos = None
    
    # Estimate time per restart
    # Let's try 5 restarts.
    num_restarts = 5
    
    rng = np.random.default_rng(n)
    
    for k in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        pos = core_pos.copy()
        current_radius = np.max(np.linalg.norm(pos, axis=1))
        atoms_to_add = n - len(pos)
        
        new_atoms = []
        attempts = 0
        while len(new_atoms) < atoms_to_add and attempts < 500:
            r = rng.uniform(current_radius * 0.8, current_radius + 2.5)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(-np.pi/2, np.pi/2)
            
            p = np.array([
                r * np.cos(phi) * np.cos(theta),
                r * np.cos(phi) * np.sin(theta),
                r * np.sin(phi)
            ])
            
            if np.all(np.linalg.norm(pos - p, axis=1) > 0.5):
                if not new_atoms or np.all(np.linalg.norm(np.array(new_atoms) - p, axis=1) > 0.5):
                    new_atoms.append(p)
            attempts += 1
            
        if len(new_atoms) < atoms_to_add:
            for i in range(len(new_atoms), atoms_to_add):
                p = np.array([current_radius + 2 + i, 0, 0])
                new_atoms.append(p)
                
        pos = np.vstack([pos, np.array(new_atoms)])
        
        # 4. Relax
        try:
            try:
                e, opt_pos = relax(pos, potential)
            except Exception:
                parameter = instance.get("rho", instance.get("kind"))
                e, opt_pos = relax(pos, potential, parameter)
            
            if e < best_energy:
                best_energy = e
                best_pos = opt_pos
                
        except Exception:
            pass
            
    if best_pos is None:
        best_pos = pos
        
    return best_pos, best_energy

def construct(instance):
    """
    Construct a low-energy Dzugutov cluster.
    """
    n = instance['n']
    potential = instance['potential']
    
    # Local deadline for this call
    # 300s / 6 instances = 50s. Let's use 45s.
    local_deadline = time.time() + 45.0
    
    pos, energy = _solve_instance_with_deadline(instance, local_deadline)
    
    return pos

# EVOLVE-BLOCK-END
