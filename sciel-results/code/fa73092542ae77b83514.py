"""Dzugutov cluster global minima search using basin hopping with polytetrahedral seeding."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def _V(r):
    """Dzugutov potential for scalar distance r."""
    if r < 1e-10:
        return 0.0
    if r < a_POT:
        return A_POT * (r**(-m_POT) - B_POT) * np.exp(c_POT / (r - a_POT))
    elif r < b_POT:
        return B_POT * np.exp(d_POT / (r - b_POT))
    else:
        return 0.0


def _energy_and_grad(positions):
    """Compute energy and gradient for a cluster using the Dzugutov potential."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    
    # Only consider pairs i < j
    triu_indices = np.triu_indices(n, k=1)
    r_ij = r[triu_indices]
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i, j in zip(*triu_indices):
        ri = r_ij[i - j + (j > i) * 0]  # This approach is complex; let's use a simpler method
        pass
    
    # Simpler approach: iterate over pairs
    for i in range(n):
        for j in range(i + 1, n):
            rij_vec = positions[i] - positions[j]
            rij = np.linalg.norm(rij_vec)
            if rij < 1e-10:
                continue
            if rij < a_POT:
                V = A_POT * (rij**(-m_POT) - B_POT) * np.exp(c_POT / (rij - a_POT))
                # dV/dr
                dVdr = A_POT * (-m_POT * rij**(-m_POT - 1)) * np.exp(c_POT / (rij - a_POT)) + \
                       A_POT * (rij**(-m_POT) - B_POT) * np.exp(c_POT / (rij - a_POT)) * (-c_POT / (rij - a_POT)**2)
            elif rij < b_POT:
                V = B_POT * np.exp(d_POT / (rij - b_POT))
                dVdr = B_POT * np.exp(d_POT / (rij - b_POT)) * (-d_POT / (rij - b_POT)**2)
            else:
                V = 0.0
                dVdr = 0.0
            
            energy += V
            # Gradient contribution: dV/dri = dV/dr * (ri - rj)/r
            if rij > 1e-10:
                grad[i] += dVdr * rij_vec / rij
                grad[j] -= dVdr * rij_vec / rij
    
    return energy, grad


def _icosahedron(center, radius=1.1):
    """Generate an icosahedron of 12 atoms around a center atom."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return center + verts


def _build_polytetrahedral_seed(n):
    """Build a polytetrahedral seed by growing icosahedral units."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    atoms = [np.array([0.0, 0.0, 0.0])]
    
    # Add icosahedral shells
    radius1 = 1.1
    radius2 = 1.8
    radius3 = 2.5
    
    # First shell: 12 atoms
    if n > 1:
        shell1 = _icosahedron(np.array([0.0, 0.0, 0.0]), radius1)
        for atom in shell1:
            if len(atoms) < n:
                atoms.append(atom)
    
    # Second shell: 30 atoms (approximate positions)
    if n > 13:
        phi = (1 + np.sqrt(5)) / 2
        # Generate approximate second shell positions
        second_shell = []
        # Edge midpoints of icosahedron scaled
        phi2 = (1 + np.sqrt(5)) / 2
        base_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(base_verts, axis=1, keepdims=True)
        base_verts = base_verts / norms
        
        # Generate 30 positions for second shell
        for i in range(12):
            for j in range(i + 1, 12):
                if len(second_shell) < 30:
                    mid = (base_verts[i] + base_verts[j]) / 2
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-10:
                        mid = mid / mid_norm * radius2
                        # Check if this position is sufficiently far from existing atoms
                        too_close = False
                        for existing in atoms:
                            if np.linalg.norm(mid - existing) < 0.5:
                                too_close = True
                                break
                        if not too_close:
                            second_shell.append(mid)
        
        for atom in second_shell:
            if len(atoms) < n:
                atoms.append(atom)
    
    # Third shell: 92 atoms (approximate)
    if n > 43:
        # Generate approximate third shell positions on a sphere
        rng = np.random.default_rng(42)
        third_shell = []
        attempts = 0
        while len(third_shell) < 92 and attempts < 5000:
            # Random point on sphere
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            point = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius3
            attempts += 1
            # Check distance from existing atoms
            too_close = False
            for existing in atoms:
                if np.linalg.norm(point - existing) < 0.5:
                    too_close = True
                    break
            if not too_close:
                third_shell.append(point)
        
        for atom in third_shell:
            if len(atoms) < n:
                atoms.append(atom)
    
    # If we still don't have enough atoms, add more randomly
    if len(atoms) < n:
        rng = np.random.default_rng(123)
        while len(atoms) < n:
            p = rng.uniform(-3.0, 3.0, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in atoms):
                atoms.append(p)
    
    return np.array(atoms[:n])


def _basin_hopping_step(positions, step_size=0.3):
    """Perform a single basin hopping step: perturb then relax."""
    n = len(positions)
    # Random perturbation
    rng = np.random.default_rng()
    perturbation = rng.normal(0, step_size, (n, 3))
    new_positions = positions + perturbation
    
    # Relax
    energy, relaxed = relax(new_positions, 'dzugutov', None)
    return energy, relaxed


def construct(instance):
    """Find low-energy Dzugutov cluster using basin hopping with polytetrahedral seeding."""
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Build initial polytetrahedral seed
    best_energy = np.inf
    best_positions = _build_polytetrahedral_seed(size)
    
    # Initial relaxation
    energy, relaxed = relax(best_positions, 'dzugutov', None)
    if energy < best_energy:
        best_energy = energy
        best_positions = relaxed
    
    # Basin hopping
    step_size = 0.3
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        energy, new_positions = _basin_hopping_step(best_positions, step_size)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = new_positions
            # Accept, maybe increase step size slightly
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Reject, decrease step size
            step_size = max(step_size * 0.95, 0.1)
    
    return best_positions
# EVOLVE-BLOCK-END
