"""LJ38 cluster global minimum search.

Strategy:
  * Directly construct the known FCC truncated-octahedron geometry (the
    accepted global minimum for LJ38, E ~ -173.928427) by taking the 38
    nearest FCC lattice points to the origin and rescaling to the LJ
    equilibrium nearest-neighbour distance 2^(1/6).
  * Polish with L-BFGS using an analytic gradient (vectorised), plus a few
    alternative starts (icosahedral funnel, random-perturbed FCC) as a safety
    net in case the FCC seed is imperfect.
  * Explicit deadline guard: return the best configuration found before the
    120 s evaluation window (killed at 150 s), with a generous safety margin.
"""
import numpy as np
import scipy.optimize
import time

N = 38


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero

    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Matrix includes both (i,j) and (j,i), so divide by 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0

    # Gradient
    # dE/dr2 = -24 * r2^-7 + 12 * r2^-4
    r8 = r2 ** 4
    r14 = r2 ** 7
    M = -24.0 / r14 + 12.0 / r8
    grad = (M @ diff).reshape(-1)

    return energy, grad


def _enforce_min_spacing(pos, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist."""
    pos = pos.copy()
    for _ in range(200):  # iterations
        changed = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    # Push apart
                    if d < 1e-10:
                        pos[j] += np.array([min_dist, 0, 0])
                    else:
                        direction = (pos[j] - pos[i]) / d
                        push = (min_dist - d) / 2.0
                        pos[i] -= push * direction
                        pos[j] += push * direction
                    changed = True
        if not changed:
            break
    return pos


def _scale_to_target_nn(pos, target_nn=2.0 ** (1.0 / 6.0)):
    """Rescale positions so that the median nearest-neighbour distance equals target_nn."""
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each atom, find nearest neighbour distance
    nn_dists = np.min(dists, axis=1)
    median_nn = np.median(nn_dists)
    
    if median_nn > 1e-10:
        scale = target_nn / median_nn
        pos = pos * scale
    
    return pos


def _make_fcc_truncated_octahedron(n_atoms=N):
    """
    Generate an FCC lattice fragment of n_atoms atoms centered at the origin.
    This approximates the known global minimum topology for LJ38.
    """
    # FCC lattice points: (i, j, k) where i+j+k is even
    # Scaled so that nearest-neighbour distance is 2^(1/6)
    a = 2.0 ** (2.0 / 3.0)  # lattice constant
    
    points = []
    max_range = 8
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                if (i + j + k) % 2 == 0:
                    x = i * a
                    y = j * a
                    z = k * a
                    r2 = x * x + y * y + z * z
                    points.append((r2, np.array([x, y, z])))
    
    points.sort(key=lambda x: x[0])
    selected = [p[1] for p in points[:n_atoms]]
    
    coords = np.array(selected)
    center = np.mean(coords, axis=0)
    coords = coords - center
    
    # Rescale to target nearest-neighbour distance
    coords = _scale_to_target_nn(coords)
    
    return coords


def _make_icosahedral(n_atoms=N):
    """
    Generate an icosahedral cluster (13 + 12 + 13 = 38 atoms).
    Shell 0: 1 atom at center
    Shell 1: 12 atoms (icosahedron)
    Shell 2: 25 atoms (approximate second shell)
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_vertices /= np.linalg.norm(ico_vertices[0])  # normalize to unit sphere
    
    # Center atom
    atoms = [np.zeros(3)]
    
    # First shell: 12 icosahedron vertices at radius r1
    r1 = 2.0 ** (1.0 / 6.0)  # target NN distance
    for v in ico_vertices:
        atoms.append(r1 * v)
    
    # Second shell: 25 atoms
    # Approximate by placing atoms at face centers and edge midpoints
    # For simplicity, use a denser spherical shell
    # Generate points on a sphere using Fibonacci lattice
    n_second = n_atoms - 1 - 12  # 25 atoms
    r2 = 2 * r1 * 0.95  # slightly less than 2*r1 to avoid overlap
    
    # Fibonacci sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n_second):
        z = 1 - 2 * i / (n_second - 1)
        radius = np.sqrt(1 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        atoms.append(r2 * np.array([x, y, z]))
    
    coords = np.array(atoms)
    center = np.mean(coords, axis=0)
    coords = coords - center
    
    return coords


def _relax_lbfgs(pos, maxiter=2000, ftol=1e-15, gtol=1e-15):
    """Relax positions using L-BFGS-B with analytic gradient."""
    def objective(x):
        return _lj_energy_and_grad(x.reshape(-1, 3))[0]
    
    def gradient(x):
        return _lj_energy_and_grad(x.reshape(-1, 3))[1]
    
    try:
        res = scipy.optimize.minimize(
            objective,
            pos.flatten(),
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return pos


def _perturb(pos, scale=0.1, seed=None):
    """Add random perturbation to positions."""
    rng = np.random.default_rng(seed)
    return pos + scale * rng.standard_normal(pos.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: FCC truncated octahedron
    pos_fcc = _make_fcc_truncated_octahedron(N)
    pos_fcc = _enforce_min_spacing(pos_fcc)
    pos_relaxed = _relax_lbfgs(pos_fcc)
    energy, _ = _lj_energy_and_grad(pos_relaxed)
    if energy < best_energy:
        best_energy = energy
        best_pos = pos_relaxed
    
    if time.time() > deadline:
        return best_pos
    
    # Strategy 2: Icosahedral
    pos_ico = _make_icosahedral(N)
    pos_ico = _enforce_min_spacing(pos_ico)
    pos_relaxed = _relax_lbfgs(pos_ico)
    energy, _ = _lj_energy_and_grad(pos_relaxed)
    if energy < best_energy:
        best_energy = energy
        best_pos = pos_relaxed
    
    if time.time() > deadline:
        return best_pos
    
    # Strategy 3: Perturbed FCC with multiple restarts
    for seed in [42, 123, 456, 789, 111, 222, 333, 444]:
        if time.time() > deadline:
            break
        pos_pert = _perturb(pos_fcc, scale=0.05, seed=seed)
        pos_pert = _enforce_min_spacing(pos_pert)
        pos_relaxed = _relax_lbfgs(pos_pert)
        energy, _ = _lj_energy_and_grad(pos_relaxed)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos_relaxed
    
    if time.time() > deadline:
        return best_pos
    
    # Strategy 4: Larger perturbations to escape local minima
    for seed in [555, 666, 777, 888]:
        if time.time() > deadline:
            break
        pos_pert = _perturb(pos_fcc, scale=0.15, seed=seed)
        pos_pert = _enforce_min_spacing(pos_pert)
        pos_relaxed = _relax_lbfgs(pos_pert)
        energy, _ = _lj_energy_and_grad(pos_relaxed)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos_relaxed
    
    if time.time() > deadline:
        return best_pos
    
    # High-precision polish
    if best_pos is not None:
        pos_polished = _relax_lbfgs(best_pos, maxiter=5000, ftol=1e-16, gtol=1e-16)
        energy, _ = _lj_energy_and_grad(pos_polished)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos_polished
    
    return best_pos
