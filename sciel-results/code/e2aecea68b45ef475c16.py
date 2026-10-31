import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    min_dist2 = min_dist * min_dist
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < min_dist2:
                return False
    return True

def make_fcc_38():
    """
    Generate the 38 closest atoms in an FCC lattice, scaled for LJ equilibrium.
    FCC lattice: (i, j, k) where i+j+k is even.
    Nearest neighbor distance in lattice units: sqrt(2).
    Target LJ nearest neighbor distance: 2^(1/6).
    Scale factor s: s * sqrt(2) = 2^(1/6) => s = 2^(1/6) / 2^(1/2) = 2^(-1/3).
    """
    scale = 2.0 ** (-1.0 / 3.0)
    max_coord = 10
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Select the first 38 points
    selected = [p[:3] for p in points[:N]]
    coords = np.array(selected, dtype=float)
    
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) / 2 (because double counted in matrix)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient:
    # F_i = sum_j (-24 r^-14 + 12 r^-8) * (r_i - r_j)
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def optimize_scale(positions, max_iter=100):
    """
    Optimize the global scale factor s to minimize energy E(s * positions).
    This helps if the cluster is slightly compressed or expanded.
    """
    def energy_at_scale(s):
        scaled = positions * s
        return lj_energy_vectorized(scaled)
    
    # Use bounded scalar minimization. Scale is likely close to 1.0
    # LJ equilibrium is 2^(1/6) ~ 1.12. If we started at 1.0, s might be ~1.12.
    # If we started at FCC scaled by 2^(-1/3), s should be very close to 1.
    res = minimize_scalar(energy_at_scale, bounds=(0.5, 2.0), method='bounded', 
                          options={'maxiter': max_iter})
    
    if res.success:
        return positions * res.x
    return positions

def _detect_icosahedral_motifs(positions, cutoff=1.35):
    """
    Detect atoms with icosahedral-like coordination (12 neighbors at ~1.12-1.25 distance
    with angular distribution characteristic of icosahedral symmetry).
    Returns list of indices with icosahedral coordination.
    """
    n = len(positions)
    ico_indices = []
    
    for i in range(n):
        # Find neighbors within cutoff
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < cutoff:
                neighbors.append((j, r))
        
        # Icosahedral coordination: ~12 neighbors
        if len(neighbors) < 10 or len(neighbors) > 14:
            continue
        
        # Check angular distribution: icosahedron has 5-fold axis
        # Compute angles between neighbor vectors
        if len(neighbors) < 2:
            continue
        
        neighbor_vecs = []
        for j, r in neighbors:
            v = positions[j] - positions[i]
            v = v / np.linalg.norm(v)
            neighbor_vecs.append(v)
        
        # Check for 5-fold symmetry: look for a direction where 5 neighbors are roughly equally spaced
        # Simplified: check if there's a pair of neighbors with angle ~72 degrees (360/5)
        # and another pair with ~144 degrees
        angles = []
        for a in range(len(neighbor_vecs)):
            for b in range(a+1, len(neighbor_vecs)):
                cos_angle = np.dot(neighbor_vecs[a], neighbor_vecs[b])
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        if len(angles) == 0:
            continue
        
        # 5-fold axis: look for angles near 72° (1.2566 rad) and 144° (2.5133 rad)
        near_72 = sum(1 for a in angles if abs(a - 1.2566) < 0.15)
        near_144 = sum(1 for a in angles if abs(a - 2.5133) < 0.15)
        
        if near_72 >= 3 and near_144 >= 2:
            ico_indices.append(i)
    
    return ico_indices


def _topology_swap_displacement(positions, rng):
    """
    Apply a targeted symmetry-adapted displacement to break icosahedral motifs.
    For each detected icosahedral center, apply a displacement perpendicular to
    the local 5-fold axis to favor FCC-like rearrangement.
    """
    n = len(positions)
    displacement = np.zeros_like(positions)
    
    ico_indices = _detect_icosahedral_motifs(positions)
    
    if not ico_indices:
        # No icosahedral motifs detected; apply generic perturbation
        return rng.normal(0, 0.3, positions.shape)
    
    for i in ico_indices:
        # Find the 5-fold axis: average of 5 most symmetrically placed neighbors
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if 1.0 < r < 1.3:
                neighbors.append((j, r, rij / r))
        
        if len(neighbors) < 5:
            # Fallback: random displacement
            displacement[i] += rng.normal(0, 0.25, 3)
            continue
        
        # Estimate 5-fold axis: find direction that maximizes symmetry
        # Simplified: use the direction that is most orthogonal to the plane of neighbors
        # Or just pick a random direction perpendicular to the average neighbor direction
        
        avg_dir = np.mean([v for _, _, v in neighbors], axis=0)
        avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-10)
        
        # Create a vector perpendicular to avg_dir
        if np.linalg.norm(avg_dir) < 1e-10:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            # Pick an arbitrary vector not parallel to avg_dir
            if abs(avg_dir[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            perp = perp - np.dot(perp, avg_dir) * avg_dir
            perp = perp / (np.linalg.norm(perp) + 1e-10)
        
        # Apply displacement: push atom away from icosahedral center along perp direction
        # with some randomization
        magnitude = 0.2 + rng.random() * 0.15
        displacement[i] += magnitude * perp * rng.choice([-1, 1])
        
        # Also perturb nearby atoms to help them rearrange
        for j, r, v in neighbors:
            if r < 1.25:
                displacement[j] += 0.1 * rng.normal(0, 1, 3)
    
    return displacement


def basin_hopping(positions, n_iter=20, step_size=0.2, maxiter=100, temperature=0.05):
    """Perform basin hopping optimization with analytic gradient and topology-swap."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Decide perturbation strategy: topology-swap vs generic
        use_topology_swap = (i % 3 == 0) or (i >= n_iter // 2)
        
        if use_topology_swap:
            # Topology-swap: detect icosahedral motifs and apply targeted displacement
            disp = _topology_swap_displacement(x0.reshape(N, 3), rng)
            perturbed = x0 + disp.flatten()
        else:
            # Generic random perturbation
            noise = rng.normal(0, step_size, x0.shape)
            perturbed = x0 + noise
        
        # Relax
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
            
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Construct FCC 38
    try:
        fcc = make_fcc_38()
        if check_validity(fcc):
            # Relax
            relaxed = relax(fcc, maxiter=200, gtol=1e-7)
            if check_validity(relaxed):
                # Optimize Scale
                scaled = optimize_scale(relaxed)
                if check_validity(scaled):
                    # Relax again after scaling
                    relaxed2 = relax(scaled, maxiter=200, gtol=1e-7)
                    if check_validity(relaxed2):
                        energy = lj_energy_vectorized(relaxed2)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = relaxed2
    except Exception:
        pass

    # 2. If we have time, do Basin Hopping from the best result with topology-swap
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            # First round: topology-swap focused BH
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=25, step_size=0.25, maxiter=80, temperature=0.04)
            if check_validity(bh_result):
                # Optimize scale and relax again
                scaled_bh = optimize_scale(bh_result)
                if check_validity(scaled_bh):
                    final_bh = relax(scaled_bh, maxiter=200, gtol=1e-8)
                    if check_validity(final_bh):
                        energy = lj_energy_vectorized(final_bh)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = final_bh
        except Exception:
            pass
        
        # Second round: if still time, do another BH from the best
        if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
            try:
                bh_result2, bh_energy2 = basin_hopping(best_positions, n_iter=15, step_size=0.35, maxiter=60, temperature=0.06)
                if check_validity(bh_result2):
                    scaled_bh2 = optimize_scale(bh_result2)
                    if check_validity(scaled_bh2):
                        final_bh2 = relax(scaled_bh2, maxiter=200, gtol=1e-8)
                        if check_validity(final_bh2):
                            energy = lj_energy_vectorized(final_bh2)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = final_bh2
            except Exception:
                pass

    # 3. Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 8:
        try:
            # One more scale optimization
            scaled_final = optimize_scale(best_positions)
            if check_validity(scaled_final):
                polished = relax(scaled_final, maxiter=500, gtol=1e-10)
                if check_validity(polished):
                    energy = lj_energy_vectorized(polished)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = polished
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        # Try a simple cubic lattice fallback if everything fails
        side = int(np.ceil(N ** (1/3)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pts = np.array(pts[:N], dtype=float)
        # Center it
        pts -= np.mean(pts, axis=0)
        # Scale to LJ equilibrium
        pts *= 1.12
        if check_validity(pts):
            best_positions = pts
        else:
            # Last resort
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
