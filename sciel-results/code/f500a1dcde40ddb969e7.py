"""
Strategy: Rigid Body Assembly for Morse Clusters.
1. Generate small rigid "super-atoms" (10-12 atoms) from crystal templates.
2. Assemble these super-atoms into a cluster of size N.
3. Optimize the positions and orientations of the super-atoms (low dimensionality).
4. Perform full atomic relaxation on the best candidate.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START = None
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill


def _get_remaining_time():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return _DEADLINE_SECONDS - (time.time() - _GLOBAL_START)


def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Morse potential: exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0))
    return energy


def _morse_energy_and_grad(pos, rho):
    """Compute energy and gradient for L-BFGS-B."""
    energy, grad = morse_energy_gradient(pos, rho)
    return energy, grad


def _quat_to_rot(q):
    """Convert quaternion to rotation matrix."""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R


def _generate_superatom(n_atoms, spacing=1.0, seed=0):
    """Generate a small rigid cluster of n_atoms."""
    rng = np.random.default_rng(seed)
    
    # Generate a small FCC chunk
    a = spacing
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    side = int(math.ceil(n_atoms ** (1.0 / 3.0))) + 1
    lattice_points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * a
                    lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    com = np.mean(lattice_points, axis=0)
    lattice_points -= com
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    lattice_points = lattice_points[order]
    
    if len(lattice_points) >= n_atoms:
        return lattice_points[:n_atoms].copy()
    else:
        # Pad with random
        remaining = n_atoms - len(lattice_points)
        extra = rng.randn(remaining, 3) * 0.5
        return np.vstack([lattice_points, extra])


def _assemble_cluster(n_total, n_superatoms, superatom_size, seed=0):
    """Assemble a cluster from rigid superatoms."""
    rng = np.random.default_rng(seed)
    
    # Determine how many superatoms of size superatom_size and one remainder
    num_full = n_total // superatom_size
    remainder = n_total % superatom_size
    
    superatoms = []
    # Generate distinct superatoms
    for i in range(num_full):
        sa = _generate_superatom(superatom_size, seed=seed + i)
        superatoms.append(sa)
    
    if remainder > 0:
        sa_rem = _generate_superatom(remainder, seed=seed + num_full)
        superatoms.append(sa_rem)
    elif num_full == 0:
        # If n_total < superatom_size, just use one superatom
        sa_rem = _generate_superatom(n_total, seed=seed)
        superatoms.append(sa_rem)
    
    K = len(superatoms)
    
    # Initial positions for superatoms: roughly spherical distribution
    radius = n_total ** (1/3) * 0.8
    positions = rng.uniform(-radius, radius, (K, 3))
    
    # Initial quaternions: identity
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (K, 1))
    
    # Flatten parameters: [pos1 (3), quat1 (4), pos2 (3), quat2 (4), ...]
    params = np.concatenate([positions.flatten(), quats.flatten()])
    
    return superatoms, params


def _expand_superatoms(superatoms, params):
    """Expand rigid superatom parameters to full atomic positions."""
    K = len(superatoms)
    positions = np.zeros((len(params) // 7 * 0, 3)) # Placeholder
    
    all_positions = []
    for i, sa in enumerate(superatoms):
        pos = params[i*7 : i*7+3]
        quat = params[i*7+3 : i*7+7]
        R = _quat_to_rot(quat)
        # Rotate superatom atoms and translate
        rotated_sa = sa @ R.T + pos
        all_positions.append(rotated_sa)
    
    return np.vstack(all_positions)


def _rigid_energy_and_grad(params, superatoms, rho):
    """Energy and gradient for rigid superatom assembly."""
    pos = _expand_superatoms(superatoms, params)
    energy, grad = morse_energy_gradient(pos, rho)
    
    # Map gradient back to superatom parameters
    # This requires chain rule: dE/dparams = sum_atoms (dE/dpos_atom) * (dpos_atom/dparams)
    # dpos_atom/dpos_superatom = I
    # dpos_atom/dquat_superatom = -cross(R * sa_atom, ...) -- complex
    # For simplicity and speed, we will use finite differences for the rigid phase
    # or just use scipy's minimize without jac for the rigid phase, 
    # but that's slow. 
    # Better: Use analytic gradient for positions, and approximate for rotations?
    # Actually, let's just use a simple numerical gradient for the rigid phase 
    # since K is small (e.g., 10 superatoms -> 70 params). 
    # 70 params * 2 evaluations * cost of energy. 
    # Cost of energy for N=80 is ~ 80^2/2 = 3200 ops. 
    # 70 * 2 * 3200 = 448,000 ops per gradient step. 
    # This is feasible.
    
    # However, implementing analytic gradient for quaternions is error-prone.
    # Let's use a trick: minimize only positions first, then orientations?
    # Or just use finite difference gradient for the whole rigid param vector.
    
    # Let's compute finite difference gradient
    h = 1e-5
    grad_params = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += h
        pos_plus = _expand_superatoms(superatoms, params_plus)
        e_plus, _ = morse_energy_gradient(pos_plus, rho)
        
        params_minus = params.copy()
        params_minus[i] -= h
        pos_minus = _expand_superatoms(superatoms, params_minus)
        e_minus, _ = morse_energy_gradient(pos_minus, rho)
        
        grad_params[i] = (e_plus - e_minus) / (2 * h)
    
    return energy, grad_params


def construct(instance):
    """Find low-energy arrangements of atoms bound by Morse potentials."""
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy: Try different superatom sizes and numbers
    # For N=38, maybe 3 superatoms of 12 + 2? Or 2 of 19?
    # For N=80, maybe 6 superatoms of 13 + 2?
    
    superatom_configs = [
        (12, 5), # Size, Seed offset
        (10, 15),
        (8, 25),
        (15, 35),
    ]
    
    # Also try a direct atomic start for comparison/fallback
    direct_inits = []
    # Generate a few direct atomic starts
    for seed in [100, 200]:
        # Simple random compact
        rng = np.random.default_rng(seed)
        radius = size ** (1/3) * 0.8
        pos = rng.uniform(-radius, radius, (size, 3))
        direct_inits.append(pos)
    
    for sa_size, sa_seed_offset in superatom_configs:
        remaining = _get_remaining_time()
        if remaining < 10.0:
            break
            
        # Assemble cluster
        superatoms, params = _assemble_cluster(size, 0, sa_size, seed=sa_seed_offset)
        
        # Optimize rigid parameters
        try:
            # Use Nelder-Mead or L-BFGS-B? 
            # L-BFGS-B with finite diff grad might be slow if K is large.
            # Let's use Powell or Nelder-Mead for the rigid phase?
            # Nelder-Mead doesn't need gradients.
            # But we have a gradient function (finite diff).
            # Let's try L-BFGS-B with the finite diff gradient.
            
            # Limit iterations to ensure we don't hang
            maxiter = 200
            
            result = minimize(
                _rigid_energy_and_grad,
                params,
                args=(superatoms, rho),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'ftol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = _expand_superatoms(superatoms, result.x).copy()
                
        except Exception:
            pass
            
    # Also run direct atomic optimization on the best rigid result and direct inits
    candidates = []
    if best_positions is not None:
        candidates.append(best_positions)
    candidates.extend(direct_inits)
    
    for init in candidates:
        remaining = _get_remaining_time()
        if remaining < 5.0:
            break
            
        # Ensure validity
        n = len(init)
        for _ in range(5):
            min_dist = np.inf
            for a in range(n):
                for b in range(a + 1, n):
                    d = np.linalg.norm(init[a] - init[b])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5:
                break
            # Jitter closest pair
            for a in range(n):
                for b in range(a + 1, n):
                    d = np.linalg.norm(init[a] - init[b])
                    if d < 0.5:
                        direction = init[a] - init[b]
                        if np.linalg.norm(direction) < 1e-6:
                            direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        init[a] += direction * (0.5 - d + 0.01)
                        init[b] -= direction * (0.5 - d + 0.01)
                        
        try:
            result = minimize(
                _morse_energy_and_grad,
                init,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.copy()
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback
        init = direct_inits[0]
        energy, best_positions = relax(init, potential, rho)
        
    return best_positions
# EVOLVE-BLOCK-END
