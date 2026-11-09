"""Morse cluster global minima search with basin hopping and structural prototypes."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import morse_energy_gradient
from scipy.optimize import minimize

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 290.0


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Compute Morse potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    triu = np.triu_indices(n, 1)
    r = dists[triu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    return float(np.sum(e * (e - 2.0)))


def _morse_gradient(positions, rho):
    """Compute Morse potential gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    triu = np.triu_indices(n, 1)
    r = dists[triu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    # dE/dr = 2*rho*exp(rho*(1-r))*(exp(rho*(1-r))-1)
    de_dr = 2.0 * rho * e * (e - 1.0)
    # grad_r = (r_i - r_j)/r_ij
    grad = np.zeros_like(positions)
    for i, j in zip(triu[0], triu[1]):
        r_ij = r[list(triu[0]).index(i)] if False else dists[i, j]
        if r_ij < 1e-10:
            continue
        factor = de_dr[list(de_dr).index(de_dr[np.where(triu[0]==i)[0][0]])] if False else 0.0
        # Recompute properly
        x_ij = rho * (1.0 - r_ij)
        e_ij = np.exp(x_ij)
        de_dr_ij = 2.0 * rho * e_ij * (e_ij - 1.0)
        vec = (positions[i] - positions[j]) / r_ij
        grad[i] += de_dr_ij * vec
        grad[j] -= de_dr_ij * vec
    return grad


def _check_validity(positions, min_dist=0.5):
    """Check if positions are valid (no overlaps)."""
    if len(positions) < 2:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist


def _local_minimize(positions, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e = _morse_energy(pos, rho)
        g = _morse_gradient(pos, rho).flatten()
        return e, g
    
    x0 = positions.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun


def _generate_fcc_cluster(n, rng):
    """Generate FCC-like cluster by truncating an FCC lattice."""
    # FCC lattice vectors
    a = 1.0  # equilibrium distance
    # Generate points in FCC lattice
    points = []
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    # Generate grid of lattice points
    extent = int(np.ceil(n ** (1/3))) + 2
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                origin = np.array([i*a, j*a, k*a])
                for b in basis:
                    p = origin + b * a
                    points.append(p)
    
    points = np.array(points)
    if len(points) <= n:
        return points
    
    # Center and select nearest n points
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp_cluster(n, rng):
    """Generate HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)  # HCP c/a ratio
    points = []
    extent = int(np.ceil(n ** (1/3))) + 2
    for layer in range(-extent, extent+1):
        z = layer * c
        offset = (0.5, 0.0, 0.0) if layer % 2 == 0 else (0.0, 0.0, 0.0)
        for i in range(-extent, extent+1):
            for j in range(-extent, extent+1):
                x = i * a + offset[0] * a
                y = j * a * np.sqrt(3.0)/2.0 + offset[1] * a
                points.append([x, y, z])
    
    points = np.array(points)
    if len(points) <= n:
        return points
    
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_icosahedral_cluster(n, rng):
    """Generate icosahedral shell cluster."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    # Layer 1: center
    points.append(np.array([0.0, 0.0, 0.0]))
    # Layer 2: icosahedron
    r2 = 1.0
    for v in verts:
        points.append(v * r2)
    # Layer 3: second shell (20 points - octahedral symmetry)
    r3 = np.sqrt(2.0)
    oct_verts = np.array([
        [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
        [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
        [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]
    ], dtype=float)
    oct_verts /= np.linalg.norm(oct_verts, axis=1, keepdims=True)
    for v in oct_verts:
        points.append(v * r3)
    # Layer 4: more points on icosahedral directions
    if n > 1 + 12 + 12:
        r4 = np.sqrt(3.0)
        for v in verts:
            points.append(v * r4)
        # Add face centers of icosahedron
        faces = [
            [0,1,8],[1,2,10],[2,3,11],[3,0,9],
            [4,8,5],[5,9,6],[6,10,7],[7,11,4],
            [0,4,8],[1,5,9],[2,6,10],[3,7,11],
            [0,1,8],[1,2,10],[2,3,11],[3,0,9]
        ]
        # Just add more along radial directions
        for v in verts:
            points.append(v * (r4 + 0.3))
    
    points = np.array(points)
    if len(points) <= n:
        return points
    
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_shell_cluster(n, rng):
    """Generate multi-shell cluster."""
    points = [np.array([0.0, 0.0, 0.0])]
    shells = [1.0, 1.5, 2.0, 2.5, 3.0]
    for r in shells:
        if len(points) >= n:
            break
        # Fibonacci sphere for this shell
        m = min(n - len(points), int(4 * r**2))
        if m <= 0:
            continue
        phi = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(m) + 0.5
        cos_theta = 1.0 - (i / m)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        theta = i * phi
        x = sin_theta * np.cos(theta)
        y = sin_theta * np.sin(theta)
        z = cos_theta
        shell_pts = np.column_stack((x, y, z)) * r
        points.extend(shell_pts)
    
    points = np.array(points)
    if len(points) <= n:
        return points
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_random_sphere(n, radius, rng):
    """Generate random points on a sphere."""
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms * radius
    return points


def _basin_hop(positions, rho, rng, step_size=0.5, max_iter=20):
    """Basin hopping: perturb, relax, accept/reject."""
    best_pos = positions.copy()
    best_energy = _morse_energy(best_pos, rho)
    
    for _ in range(max_iter):
        if _time_remaining() < 2.0:
            break
        
        # Perturb
        perturbed = best_pos + rng.normal(0, step_size, size=best_pos.shape)
        
        # Relax
        try:
            energy, relaxed = relax(perturbed, 'morse', rho)
            if relaxed is None or len(relaxed) != len(best_pos):
                continue
            if not _check_validity(relaxed):
                continue
            
            # Metropolis accept
            delta_e = energy - best_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
                best_pos = relaxed
                best_energy = energy
        except:
            continue
    
    return best_pos, best_energy


def _fibonacci_sphere(n, radius, rng):
    """
    Generate n points on a sphere of given radius using Fibonacci lattice.
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n) + 0.5
    cos_theta = 1.0 - (i / n)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    theta = i * phi
    x = sin_theta * np.cos(theta)
    y = sin_theta * np.sin(theta)
    z = cos_theta
    points = np.column_stack((x, y, z))
    
    # Random rotation
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, xq, yq, zq = q
    R = np.array([
        [1 - 2*yq*yq - 2*zq*zq, 2*xq*yq - 2*w*zq, 2*xq*zq + 2*w*yq],
        [2*xq*yq + 2*w*zq, 1 - 2*xq*xq - 2*zq*zq, 2*yq*zq - 2*w*xq],
        [2*xq*zq - 2*w*yq, 2*yq*zq + 2*w*xq, 1 - 2*xq*xq - 2*yq*yq]
    ])
    points = points @ R.T
    points *= radius
    return points


def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses basin hopping with structural prototypes (FCC, HCP, icosahedral, shells).
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate structural prototypes
    prototypes = []
    
    try:
        fcc = _generate_fcc_cluster(size, rng)
        if len(fcc) == size and _check_validity(fcc):
            prototypes.append(fcc)
    except:
        pass
    
    try:
        hcp = _generate_hcp_cluster(size, rng)
        if len(hcp) == size and _check_validity(hcp):
            prototypes.append(hcp)
    except:
        pass
    
    try:
        ico = _generate_icosahedral_cluster(size, rng)
        if len(ico) == size and _check_validity(ico):
            prototypes.append(ico)
    except:
        pass
    
    try:
        shell = _generate_shell_cluster(size, rng)
        if len(shell) == size and _check_validity(shell):
            prototypes.append(shell)
    except:
        pass
    
    # Add Fibonacci sphere and random sphere starts
    radius = 1.0 * size ** (1.0 / 3.0) * 0.9
    for _ in range(3):
        fib = _fibonacci_sphere(size, radius, rng)
        fib += rng.normal(0, 0.05, size=(size, 3))
        if _check_validity(fib):
            prototypes.append(fib)
    
    for _ in range(3):
        rand = _generate_random_sphere(size, radius, rng)
        rand += rng.normal(0, 0.05, size=(size, 3))
        if _check_validity(rand):
            prototypes.append(rand)
    
    # Basin hop from each prototype
    for proto in prototypes:
        if _time_remaining() < 5.0:
            break
        
        try:
            # First relax the prototype
            energy, relaxed = relax(proto, potential, parameter)
            if relaxed is None or len(relaxed) != size:
                continue
            if not _check_validity(relaxed):
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
            
            # Basin hop from this relaxed state
            if _time_remaining() > 10.0:
                bh_pos, bh_energy = _basin_hop(relaxed, parameter, rng, step_size=0.4, max_iter=15)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos.copy()
                    
        except:
            continue
    
    # Final local minimization with gradient
    if best_positions is not None and _time_remaining() > 3.0:
        try:
            final_pos, final_energy = _local_minimize(best_positions, parameter, max_iter=200)
            if _check_validity(final_pos) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except:
            pass
    
    # Fallback if nothing found
    if best_positions is None:
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.6
        if _time_remaining() > 2.0:
            try:
                energy, relaxed = relax(best_positions, potential, parameter)
                if _check_validity(relaxed):
                    best_positions = relaxed
            except:
                pass
    
    return best_positions
