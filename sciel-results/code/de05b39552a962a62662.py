"""Morse cluster global minima: basin hopping with crystalline starts and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    exp_x = np.exp(x)
    e = exp_x * (exp_x - 2.0)
    energy = float(np.sum(e))
    
    # Gradient: dE/dr_ij = -rho * exp_x * (2*exp_x - 1)
    de_dr = -rho * exp_x * (2.0 * exp_x - 1.0)
    
    # Build gradient vector
    grad = np.zeros_like(pos)
    for idx in range(len(r_ij)):
        i, j = iu[0][idx], iu[1][idx]
        rij = r[i, j]
        if rij < 1e-10:
            continue
        diff_ij = pos[i] - pos[j]
        factor = de_dr[idx] / rij
        grad[i] += factor * diff_ij
        grad[j] -= factor * diff_ij
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist


def _generate_fcc(n):
    """Generate an FCC-like cluster of n atoms."""
    a = 1.0
    positions = []
    # Generate FCC lattice points
    lattice = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i*a + offset[0]*a, j*a + offset[1]*a, k*a + offset[2]*a])
                    lattice.append(p)
    lattice = np.array(lattice)
    # Center
    center = np.mean(lattice, axis=0)
    lattice -= center
    # Sort by distance from center
    dists = np.linalg.norm(lattice, axis=1)
    order = np.argsort(dists)
    lattice = lattice[order]
    # Take n closest
    return lattice[:n].copy()


def _generate_hcp(n):
    """Generate an HCP-like cluster of n atoms."""
    a = 1.0
    c = a * np.sqrt(2.0 / 3.0)
    positions = []
    # HCP layers
    for layer in range(-2, 3):
        z = layer * c
        for i in range(-2, 3):
            for j in range(-2, 3):
                if layer % 2 == 0:
                    x = i * a
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 4
                p = np.array([x, y, z])
                positions.append(p)
    positions = np.array(positions)
    center = np.mean(positions, axis=0)
    positions -= center
    dists = np.linalg.norm(positions, axis=1)
    order = np.argsort(dists)
    positions = positions[order]
    return positions[:n].copy()


def _generate_icosahedral(n):
    """Generate an icosahedral cluster (Mackay icosahedron)."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    # Normalize to unit distance
    verts /= np.linalg.norm(verts[0])
    
    positions = [np.zeros(3)]  # center
    # First shell: 12 vertices
    for v in verts:
        positions.append(v.copy())
    
    # Second shell: 30 vertices (midpoints of edges scaled)
    if n > 13:
        # Generate edge midpoints
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 1.0 < d < 1.5:  # adjacent vertices
                    mid = (verts[i] + verts[j]) / 2
                    mid /= np.linalg.norm(mid)
                    edges.append(mid)
        # Remove duplicates
        unique_edges = []
        for e in edges:
            if not any(np.allclose(e, ue, atol=1e-6) for ue in unique_edges):
                unique_edges.append(e)
        for e in unique_edges[:30]:
            if len(positions) >= n:
                break
            positions.append(e.copy())
    
    # Third shell: 56 vertices
    if n > 43:
        # Add more points on the surface
        for i in range(12):
            for j in range(12):
                if i != j:
                    d = np.linalg.norm(verts[i] - verts[j])
                    if 1.0 < d < 1.5:
                        # Point at 2/3 along edge
                        p = verts[i] + (2/3) * (verts[j] - verts[i])
                        p /= np.linalg.norm(p)
                        if not any(np.allclose(p, pos, atol=1e-6) for pos in positions):
                            if len(positions) >= n:
                                break
                            positions.append(p.copy())
            if len(positions) >= n:
                break
    
    positions = np.array(positions[:n])
    return positions


def _basin_hopping(pos, rho, max_iter, deadline):
    """Perform basin hopping optimization."""
    best_pos = pos.copy()
    best_energy, _ = _morse_energy_and_grad(pos, rho)
    
    # Local relaxation
    energy, relaxed_pos = relax(pos, 'morse', rho)
    if energy < best_energy:
        best_energy = energy
        best_pos = relaxed_pos.copy()
    
    current_pos = relaxed_pos.copy()
    current_energy = energy
    
    step_size = 0.5
    accepted = 0
    total = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturb
        noise = np.random.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'morse', rho)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            accept = True
        else:
            delta_e = new_energy - current_energy
            if delta_e < 1e-10:
                accept = True
            else:
                # Temperature annealing
                T = max(0.1, 1.0 * np.exp(-it / 50.0))
                accept = np.random.random() < np.exp(-delta_e / T)
        
        if accept:
            current_pos = new_pos
            current_energy = new_energy
            accepted += 1
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        total += 1
        
        # Adapt step size
        if accepted / max(total, 1) < 0.3:
            step_size *= 0.9
        elif accepted / max(total, 1) > 0.7:
            step_size *= 1.1
        step_size = np.clip(step_size, 0.1, 2.0)
    
    return best_pos, best_energy


def construct(instance):
    """Find global minimum of Morse cluster using basin hopping with diverse starts."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Generate diverse starting configurations
    starts = []
    
    # FCC
    try:
        fcc = _generate_fcc(n)
        if _is_valid(fcc):
            starts.append(fcc)
    except:
        pass
    
    # HCP
    try:
        hcp = _generate_hcp(n)
        if _is_valid(hcp):
            starts.append(hcp)
    except:
        pass
    
    # Icosahedral
    try:
        ico = _generate_icosahedral(n)
        if _is_valid(ico):
            starts.append(ico)
    except:
        pass
    
    # Random compact
    rng = np.random.default_rng(n + 42)
    radius = n ** (1.0/3.0) * 0.7
    for _ in range(3):
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        if points:
            arr = np.array(points)
            if _is_valid(arr):
                starts.append(arr)
    
    if not starts:
        # Fallback: random
        pos = rng.uniform(-2, 2, size=(n, 3))
        starts.append(pos)
    
    # Run basin hopping from each start
    best_pos = None
    best_energy = np.inf
    
    # Budget allocation
    n_starts = len(starts)
    per_start_time = 250.0 / n_starts  # ~250s total budget
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:  # Keep 30s safety margin
            break
        
        # Local relaxation first
        energy, relaxed = relax(start, 'morse', rho)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        # Basin hopping with time limit
        remaining = deadline - 30 - time.time()
        if remaining < 5:
            break
        
        # Estimate iterations based on time
        # Each BH iteration takes ~0.1-0.5s depending on n
        est_time_per_iter = 0.2 if n < 50 else 0.4
        max_iter = int(remaining / est_time_per_iter)
        max_iter = max(max_iter, 10)
        
        bh_pos, bh_energy = _basin_hopping(relaxed, rho, max_iter, deadline - 30)
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    if best_pos is None:
        # Should not happen, but fallback
        best_pos = starts[0]
    
    return best_pos
# EVOLVE-BLOCK-END
