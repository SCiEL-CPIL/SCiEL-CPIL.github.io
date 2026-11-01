"""Lennard-Jones 55-atom cluster: multi-start search on Mackay icosahedron."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def get_mackay_directions():
    """Get the 55 unit direction vectors for a Mackay icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Return directions and shell assignments
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = get_mackay_directions()
    
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    
    return pos

def relax_positions(pos, maxiter=300, gtol=1e-6):
    """Relax atomic positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-8})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Get Mackay geometry with reasonable initial radii
    # Nearest neighbor ~ 2^(1/6) ≈ 1.122
    # For 55-atom Mackay: shell1 ~ 1.0, shell2 ~ 1.8, shell3 ~ 2.6
    initial_radii = np.array([1.0, 1.8, 2.6])
    base_pos = make_mackay_from_radii(initial_radii)
    
    # First, relax the base Mackay to get a good reference
    base_pos_relaxed, base_e = relax_positions(base_pos, maxiter=300, gtol=1e-8)
    best_pos = base_pos_relaxed.copy()
    best_e = base_e
    
    # Multi-start: perturb the relaxed Mackay with Gaussian noise
    # Use multiple noise scales to explore different basins
    rng = np.random.RandomState(42)
    
    # Define perturbation scales - small perturbations first (most likely to stay in global basin)
    noise_scales = [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7]
    
    # Try perturbations from the relaxed Mackay
    for scale in noise_scales:
        if time.time() > deadline - 15:
            break
        
        # Try 3-5 perturbations per scale
        n_perturb = 3 if scale < 0.1 else 2
        for trial in range(n_perturb):
            if time.time() > deadline - 15:
                break
            
            # Generate perturbation
            noise = rng.randn(N, 3) * scale
            # Keep center atom fixed (atom 0)
            noise[0] = 0.0
            perturbed = base_pos_relaxed + noise
            
            # Quick energy check - skip if clearly worse
            e_perturbed = lj_energy(perturbed)
            if e_perturbed > best_e + 50:  # Too far, skip
                continue
            
            # Relax
            pos_relaxed, e_relaxed = relax_positions(perturbed, maxiter=200, gtol=1e-6)
            
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # Also try perturbing from the unrelaxed Mackay (different basin)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        if time.time() > deadline - 10:
            break
        
        for trial in range(2):
            if time.time() > deadline - 10:
                break
            
            noise = rng.randn(N, 3) * scale
            noise[0] = 0.0
            perturbed = base_pos + noise
            
            e_perturbed = lj_energy(perturbed)
            if e_perturbed > best_e + 50:
                continue
            
            pos_relaxed, e_relaxed = relax_positions(perturbed, maxiter=200, gtol=1e-6)
            
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # Final polish: tight relaxation on the best candidate
    if time.time() < deadline - 3:
        pos_final, e_final = relax_positions(best_pos, maxiter=500, gtol=1e-8)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END
