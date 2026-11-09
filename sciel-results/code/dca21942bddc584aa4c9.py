"""Dzugutov cluster global minima search.
Uses physics-driven self-assembly (sequential atom addition with Langevin dynamics)
combined with basin hopping for final refinement."""

import numpy as np
import time
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16

# Global deadline guard
TOTAL_TIME_LIMIT = 305.0
_start_time = None
_best_results = {}


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass

    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-8:
                continue

            if r < A_CUT:
                V = (A_POT * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
                dVdr = A_POT * (-M_EXP * r ** (-M_EXP - 1) - C_EXP / (r - A_CUT) ** 2 * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
            elif r < B_CUT:
                V = B_POT * np.exp(D_EXP / (r - B_CUT))
                dVdr = B_POT * np.exp(D_EXP / (r - B_CUT)) * (-D_EXP / (r - B_CUT) ** 2)
            else:
                continue

            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec

    return energy, grad


def _safe_relax(positions):
    """Safely call the relax API."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        return energy, pos
    except Exception:
        return None, None


def _langevin_step(positions, velocities, dt, temperature, rng):
    """Perform one Langevin dynamics step."""
    n = len(positions)
    mass = 1.0
    gamma = 0.1  # friction coefficient
    
    # Compute forces
    _, forces = _dzugutov_energy_gradient(positions)
    forces = -forces  # F = -dE/dr
    
    # Langevin update: v_new = (v + F/m*dt + sqrt(2*gamma*kT/m)*random) * exp(-gamma*dt)
    # Position update: x_new = x + v_new*dt
    noise = rng.normal(0, np.sqrt(2 * gamma * temperature / mass), (n, 3))
    velocities = (velocities + forces / mass * dt + noise) * np.exp(-gamma * dt)
    positions = positions + velocities * dt
    
    return positions, velocities


def _self_assemble(n, rng, time_budget):
    """
    Build a cluster by sequentially adding atoms and relaxing.
    Start with a small seed and add one atom at a time.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Start with 2 atoms
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    
    current_n = 2
    start_time = time.time()
    
    # Temperature schedule: start high, cool down
    max_temp = 0.5
    min_temp = 0.01
    
    while current_n < n:
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Add a new atom at a random position near the existing cluster
        if current_n == 2:
            # For the third atom, place it to form a triangle
            angle = rng.uniform(0, 2 * np.pi)
            r = 1.0
            new_pos = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        else:
            # Place near a random existing atom
            idx = rng.integers(0, current_n)
            center = positions[idx]
            # Random direction
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            new_pos = center + dir_vec * 1.0
        
        # Append new atom
        positions = np.vstack([positions, new_pos])
        velocities = np.vstack([velocities, np.zeros(3)])
        current_n += 1
        
        # Relax the system with Langevin dynamics
        # Number of steps proportional to cluster size
        n_steps = 50 + 10 * current_n
        
        # Temperature for this stage
        progress = current_n / n
        temp = max_temp * (1 - progress) + min_temp * progress
        
        for step in range(n_steps):
            if time.time() - start_time > time_budget:
                break
            positions, velocities = _langevin_step(positions, velocities, dt=0.001, temperature=temp, rng=rng)
            
            # Enforce minimum distance
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 0.5:
                        diff = positions[i] - positions[j]
                        if np.linalg.norm(diff) < 1e-8:
                            diff = rng.normal(0, 1, 3)
                        diff /= np.linalg.norm(diff)
                        push = (0.5 - d) / 2 + 0.01
                        positions[i] += diff * push
                        positions[j] -= diff * push
        
        # Dampen velocities between additions
        velocities *= 0.5
    
    # Final gradient-based relaxation
    if time.time() - start_time < time_budget:
        try:
            result = minimize(
                lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[0],
                positions.flatten(),
                jac=lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < _dzugutov_energy_gradient(positions)[0]:
                positions = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    return positions


def _make_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedron cluster with n atoms."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 1. Center
    shells = [np.array([[0.0, 0.0, 0.0]])]
    
    # 2. Vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    R1 = 1.0
    shells.append(verts * R1)
    
    # 3. Edge Midpoints (30)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.25:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
                
    unique_edges = []
    for e in edges:
        if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
            unique_edges.append(e)
            
    if unique_edges:
        R2 = np.linalg.norm(unique_edges[0])
    else:
        R2 = 1.7
    shells.append(np.array(unique_edges) * R2)
    
    # 4. Face Centers (20)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                if d_ij < 1.25 and d_jk < 1.25 and d_ik < 1.25:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    faces.append(fc)
                    
    unique_faces = []
    for f in faces:
        if all(np.linalg.norm(f - uf) > 0.1 for uf in unique_faces):
            unique_faces.append(f)
            
    if unique_faces:
        R3 = np.linalg.norm(unique_faces[0])
    else:
        R3 = 2.2
    shells.append(np.array(unique_faces) * R3)
    
    all_atoms = np.vstack(shells)
    
    # Sort by radius to trim from outside
    radii = np.linalg.norm(all_atoms, axis=1)
    order = np.argsort(radii)
    all_atoms = all_atoms[order]
    
    return all_atoms[:n]


def _basin_hopping(positions, rng, time_budget, max_iter=20):
    """Perform basin hopping to find a lower energy structure."""
    best_pos = positions.copy()
    best_energy, _ = _dzugutov_energy_gradient(best_pos)
    
    start_time = time.time()
    step_size = 0.3
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, positions.shape)
        perturbed = best_pos + noise
        
        # Relax
        energy, relaxed = _safe_relax(perturbed)
        if energy is None:
            continue
            
        # Accept if lower energy (or with Metropolis criterion for escaping local minima)
        if energy < best_energy:
            best_pos = relaxed
            best_energy = energy
            step_size = max(0.1, step_size * 0.95)
        else:
            # Metropolis acceptance
            delta_E = energy - best_energy
            T = 0.1 * (1 - it / max_iter) + 0.01
            if rng.random() < np.exp(-delta_E / T):
                best_pos = relaxed
                best_energy = energy
            step_size = min(0.5, step_size * 1.05)
    
    return best_pos, best_energy


def _enforce_min_distance(positions, rng, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(20):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = rng.normal(0, 1, 3)
                    diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]
    
    # Check if we've already computed this
    if n in _best_results:
        return _best_results[n]

    # Calculate time budget
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 10.0)

    rng = np.random.default_rng(42 + n)

    best_energy = np.inf
    best_positions = None

    # Strategy 1: Self-assembly (physics-driven)
    if _remaining() > 5:
        sa_budget = min(time_budget * 0.4, 30.0)
        sa_pos = _self_assemble(n, rng, sa_budget)
        sa_pos = _enforce_min_distance(sa_pos, rng)
        
        # Relax and evaluate
        energy, relaxed = _safe_relax(sa_pos)
        if energy is not None:
            best_energy = energy
            best_positions = relaxed
        else:
            energy, _ = _dzugutov_energy_gradient(sa_pos)
            best_energy = energy
            best_positions = sa_pos

    # Strategy 2: Basin hopping from self-assembled structure
    if best_positions is not None and _remaining() > 5:
        bh_budget = min(time_budget * 0.3, 20.0)
        bh_pos, bh_energy = _basin_hopping(best_positions, rng, bh_budget, max_iter=15)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos

    # Strategy 3: Mackay icosahedron seed + basin hopping
    if _remaining() > 5:
        mackay_pos = _make_mackay_icosahedron(n, rng)
        mackay_pos = _enforce_min_distance(mackay_pos, rng)
        
        # Relax
        energy, relaxed = _safe_relax(mackay_pos)
        if energy is not None:
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        else:
            energy, _ = _dzugutov_energy_gradient(mackay_pos)
            if energy < best_energy:
                best_energy = energy
                best_positions = mackay_pos
        
        # Basin hopping from Mackay
        if best_positions is not None and _remaining() > 5:
            bh_budget = min(time_budget * 0.2, 15.0)
            bh_pos, bh_energy = _basin_hopping(best_positions, rng, bh_budget, max_iter=10)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos

    # Strategy 4: Random compact + basin hopping
    if _remaining() > 3:
        # Generate random compact structure
        radius = 1.1 * n ** (1.0 / 3.0) * 0.9
        min_dist = 0.6
        positions = []
        for _ in range(5000):
            if len(positions) >= n:
                break
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                    positions.append(p)
        
        while len(positions) < n:
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
        
        rand_pos = np.array(positions[:n])
        rand_pos = _enforce_min_distance(rand_pos, rng)
        
        # Relax
        energy, relaxed = _safe_relax(rand_pos)
        if energy is not None:
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        else:
            energy, _ = _dzugutov_energy_gradient(rand_pos)
            if energy < best_energy:
                best_energy = energy
                best_positions = rand_pos

    # Final fallback
    if best_positions is None:
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i] = rng.uniform(-3, 3, 3)
        best_positions = _enforce_min_distance(best_positions, rng)
        energy, relaxed = _safe_relax(best_positions)
        if relaxed is not None:
            best_positions = relaxed

    # Ensure minimum distance
    final_pos = _enforce_min_distance(best_positions, rng)

    _best_results[n] = final_pos
    return final_pos
