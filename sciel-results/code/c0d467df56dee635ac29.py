import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 143.0
START_TIME = time.time()

def lj_energy_and_grad(positions):
    """Compute LJ energy and gradient together (vectorized)."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6)
    np.fill_diagonal(coeff, 0.0)
    grad = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

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
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def lj_gradient(positions):
    """Compute gradient of LJ energy with respect to positions (vectorized)."""
    _, grad = lj_energy_and_grad(positions)
    return grad

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist

def make_fcc_cluster(n_atoms=38):
    """Generate an FCC truncated octahedron cluster for LJ38."""
    a = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    points.sort(key=lambda p: p[3])
    selected = [p[:3] for p in points[:n_atoms]]
    return np.array(selected, dtype=float)

def make_bcc_cluster(n_atoms=38):
    """Generate a BCC cluster by selecting closest lattice points to the origin."""
    a = 2.0 / math.sqrt(3.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append((i*a, j*a, k*a))
                points.append(((i+0.5)*a, (j+0.5)*a, (k+0.5)*a))
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    return np.array(points[:n_atoms], dtype=float)

def make_icosahedral_cluster(n_atoms=38):
    """Generate a nested icosahedral cluster."""
    phi = (1 + math.sqrt(5)) / 2.0
    r_ico = 1.0 / math.sqrt(1 + phi * phi)
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * r_ico
    points = [np.array([0.0, 0.0, 0.0])]
    points.extend([v * 1.12 for v in ico_verts])
    rng = np.random.default_rng(42)
    for _ in range(n_atoms - len(points)):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * math.pi * u
        z = 2 * v - 1
        r = math.sqrt(1 - z*z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append(np.array([x, y, z]) * 1.6)
    return np.array(points, dtype=float)

def relax(positions, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with vectorized gradient."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, _ = lj_energy_and_grad(pos)
        return e
    def grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        _, g = lj_energy_and_grad(pos)
        return g.flatten()
    x0 = positions.flatten()
    try:
        result = minimize(energy_func, x0, jac=grad_func, method='L-BFGS-B', 
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions):
    """High-precision polish pass with tighter tolerance and more iterations."""
    if time.time() - START_TIME > DEADLINE - 15:
        return positions
    try:
        polished = relax(positions, maxiter=2000, gtol=1e-14)
        if check_validity(polished):
            e1 = lj_energy_vectorized(positions)
            e2 = lj_energy_vectorized(polished)
            if e2 < e1:
                return polished
    except Exception:
        pass
    return positions

def optimize_scale(positions):
    """Optimize the overall scale factor of the cluster."""
    def energy_with_scale(scale):
        scaled = positions * scale[0]
        return lj_energy_vectorized(scaled)
    try:
        result = minimize(energy_with_scale, [1.0], method='Nelder-Mead',
                         options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500})
        if result.success:
            optimal_scale = result.x[0]
            if 0.5 < optimal_scale < 2.0:
                return positions * optimal_scale
    except Exception:
        pass
    return positions

def basin_hopping(positions, n_restarts=10, noise_scale=0.1):
    """Perform basin hopping: perturb, relax, keep best."""
    best_energy = lj_energy_vectorized(positions)
    best_pos = positions.copy()
    rng = np.random.default_rng(42)
    for i in range(n_restarts):
        if time.time() - START_TIME > DEADLINE - 30:
            break
        noise = rng.normal(0, noise_scale, positions.shape)
        perturbed = positions + noise
        if not check_validity(perturbed):
            noise = rng.normal(0, noise_scale * 0.5, positions.shape)
            perturbed = positions + noise
            if not check_validity(perturbed):
                continue
        relaxed = relax(perturbed)
        if not check_validity(relaxed):
            continue
        energy = lj_energy_vectorized(relaxed)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    return best_pos

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    candidates = []
    
    # 1. FCC Cluster (Global Min Candidate)
    try:
        fcc = make_fcc_cluster(N)
        if check_validity(fcc):
            candidates.append(fcc)
    except Exception:
        pass
        
    # 2. Perturbed FCC (small noise)
    try:
        fcc_pert = make_fcc_cluster(N)
        rng = np.random.default_rng(123)
        fcc_pert += rng.normal(0, 0.03, fcc_pert.shape)
        if check_validity(fcc_pert):
            candidates.append(fcc_pert)
    except Exception:
        pass
        
    # 3. BCC Cluster
    try:
        bcc = make_bcc_cluster(N)
        if check_validity(bcc):
            candidates.append(bcc)
    except Exception:
        pass
        
    # 4. Icosahedral Cluster
    try:
        ico = make_icosahedral_cluster(N)
        if check_validity(ico):
            candidates.append(ico)
    except Exception:
        pass
    
    # Relax all candidates
    for i, cand in enumerate(candidates):
        if time.time() - START_TIME > DEADLINE - 40:
            break
        try:
            relaxed = relax(cand)
            if not check_validity(relaxed):
                continue
            scaled = optimize_scale(relaxed)
            if check_validity(scaled):
                e_scaled = lj_energy_vectorized(scaled)
                if e_scaled < best_energy:
                    best_energy = e_scaled
                    best_positions = scaled
            else:
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # Basin hopping from the best candidate
    if best_positions is not None:
        if time.time() - START_TIME < DEADLINE - 30:
            try:
                bh_result = basin_hopping(best_positions, n_restarts=8, noise_scale=0.08)
                e_bh = lj_energy_vectorized(bh_result)
                if e_bh < best_energy:
                    best_energy = e_bh
                    best_positions = bh_result
            except Exception:
                pass
    
    # Final high-precision polish on the best result
    if best_positions is not None:
        if time.time() - START_TIME < DEADLINE - 10:
            try:
                polished = polish(best_positions)
                if check_validity(polished):
                    scaled_polished = optimize_scale(polished)
                    if check_validity(scaled_polished):
                        e_scaled = lj_energy_vectorized(scaled_polished)
                        if e_scaled < best_energy:
                            best_energy = e_scaled
                            best_positions = scaled_polished
                    else:
                        e_polished = lj_energy_vectorized(polished)
                        if e_polished < best_energy:
                            best_energy = e_polished
                            best_positions = polished
            except Exception:
                pass
    
    if best_positions is None:
        try:
            best_positions = make_fcc_cluster(N)
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
