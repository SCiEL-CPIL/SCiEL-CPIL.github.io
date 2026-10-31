"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for a set of positions."""
    dists = pdist(pos)
    # Ensure no zero distances
    dists = np.maximum(dists, 1e-12)
    r6 = dists ** 6
    return 4.0 * np.sum(1.0 / (r6 * r6) - 1.0 / r6)


def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    n = len(pos)
    # Compute all pairwise distances
    dists = pdist(pos)
    idx = np.triu_indices(n, k=1)
    diff = pos[idx[0]] - pos[idx[1]]
    r = dists
    r6 = r ** 6
    r12 = r6 * r6
    # Force on each pair: 4*(12/r^13 - 6/r^7) * (pi - pj)/r
    # Gradient contribution to i: -4*(12/r^13 - 6/r^7) * diff / r
    # Actually, let's compute more carefully.
    # E = sum 4*(r^-12 - r^-6)
    # dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dpos_i = dE/dr * dr/dpos_i = dE/dr * (pos_i - pos_j)/r
    # dE/dr = 4*(-12/r^13 + 6/r^7) = 4*(6/r^7 - 12/r^13)
    # So dE/dpos_i = 4*(6/r^7 - 12/r^13) * (pos_i - pos_j)/r
    # = 4*(6/r^8 - 12/r^14) * (pos_i - pos_j)
    # Let's just use finite differences or a simpler approach.
    
    # Simpler: use the fact that for each pair (i,j):
    # dE/dpos_i += 4*(12/r^13 - 6/r^7) * (pos_i - pos_j)/r
    # dE/dpos_j += 4*(12/r^13 - 6/r^7) * (pos_j - pos_i)/r
    
    grad = np.zeros_like(pos)
    coeff = 4.0 * (12.0 / (r ** 13) - 6.0 / (r ** 7))
    # coeff has shape (n_pairs,)
    # diff has shape (n_pairs, 3)
    # For each pair, add coeff * diff to grad[i] and -coeff * diff to grad[j]
    # Use np.add.at
    np.add.at(grad, idx[0], coeff[:, np.newaxis] * diff)
    np.add.at(grad, idx[1], -coeff[:, np.newaxis] * diff)
    return grad


def _make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate a truncated octahedron structure for 38 atoms.
    A truncated octahedron has 24 vertices, 8 hexagonal faces, 6 square faces.
    For LJ38, the global minimum is a truncated octahedron with specific atom placements.
    
    The standard truncated octahedron for 38 atoms consists of:
    - 1 atom at center
    - 12 atoms at edge midpoints (or equivalent positions)
    - 24 atoms at vertices
    But that's 37. Let me think more carefully.
    
    Actually, the known global minimum for LJ38 is a truncated octahedron.
    Let me construct it properly.
    
    A regular truncated octahedron can be described by taking a cube and cutting off corners.
    For LJ clusters, the structure is typically:
    - Layer 1 (z=0): 1 atom at center
    - Layer 2 (z=h): 12 atoms in a ring
    - Layer 3 (z=2h): 12 atoms in a ring  
    - Layer 4 (z=3h): 12 atoms in a ring
    - Layer 5 (z=4h): 1 atom at center
    Total: 1+12+12+12+1 = 38
    
    Actually, let me use a more standard construction. The truncated octahedron
    for 38 atoms in LJ is well-known. Let me place atoms at positions that
    approximate this structure.
    
    Alternative: use a known set of coordinates or generate from symmetry.
    
    Let me try a different approach: generate a rough truncated octahedron
    and then relax it.
    """
    # Generate approximate truncated octahedron
    # Use icosahedral-like arrangement but with cubic symmetry
    
    # Actually, let's just create a reasonable initial guess and relax it.
    # Start with a spherical shell arrangement
    
    positions = []
    
    # Center atom
    positions.append([0, 0, 0])
    
    # First shell: 12 atoms (cubic symmetry - vertices of a cube + face centers)
    a = 1.0
    # Cube vertices (8)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                positions.append([sx * a, sy * a, sz * a])
    # Face centers (6)
    for coord in range(3):
        for sign in [-1, 1]:
            p = [0, 0, 0]
            p[coord] = sign * a * np.sqrt(3)  # Adjusted for proper spacing
            positions.append(p)
    
    # Second shell: 24 atoms
    b = 1.8
    # Edge midpoints of a larger cube
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                # Skip corners (already have 8)
                # Add edge midpoints
                pass
    
    # This is getting complicated. Let me just use a simple approach:
    # Generate points on a sphere with cubic symmetry and relax.
    
    return np.array(positions[:n_atoms], dtype=float)


def construct():
    """
    Construct initial positions for 38 LJ atoms.
    Uses a multi-start strategy with local relaxation.
    """
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: Start from a rough truncated octahedron-like structure
    # and relax using L-BFGS
    
    # Generate initial guess: spherical shell with cubic symmetry
    np.random.seed(42)
    
    # Method: Place atoms in shells with some cubic symmetry
    # Shell 1: 1 atom at center
    # Shell 2: 12 atoms (octahedral positions)
    # Shell 3: 24 atoms (cuboctahedral positions)
    
    positions = []
    
    # Center
    positions.append(np.array([0.0, 0.0, 0.0]))
    
    # 12 atoms at octahedral + tetrahedral positions (first coordination shell)
    r1 = 1.0
    # Octahedral: 6 positions
    octa = [
        [r1, 0, 0], [-r1, 0, 0],
        [0, r1, 0], [0, -r1, 0],
        [0, 0, r1], [0, 0, -r1]
    ]
    # Tetrahedral: 4 positions  
    t = r1 / np.sqrt(3)
    tetra = [
        [t, t, t], [t, t, -t], [t, -t, t], [t, -t, -t],
        [-t, t, t], [-t, t, -t], [-t, -t, t], [-t, -t, -t]
    ]
    # That's 14, too many. Let me use 12: 6 octa + 6 from tetra
    
    # Actually for LJ, the first shell is typically 12 atoms.
    # Let's use icosahedral positions for 12 atoms
    phi = (1 + np.sqrt(5)) / 2
    r_ico = 1.0
    ico_positions = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    ico_positions = np.array(ico_positions) / np.linalg.norm(ico_positions[0]) * r_ico
    positions.extend(ico_positions.tolist())
    
    # Now 13 atoms. Need 25 more.
    # Second shell: 25 atoms
    r2 = 1.8
    # Add more positions on a sphere
    # Use Fibonacci sphere for remaining 25
    n_remaining = 38 - len(positions)
    if n_remaining > 0:
        # Fibonacci sphere
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n_remaining):
            z = 1 - 2 * (i + 0.5) / n_remaining
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / golden
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append([x * r2, y * r2, z * r2])
    
    pos = np.array(positions, dtype=float)
    
    # Relax using L-BFGS
    try:
        result = minimize(
            _lj_energy,
            pos,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x
    except Exception:
        pass
    
    # Strategy 2: Try a different initial guess - more cubic symmetry
    np.random.seed(123)
    positions2 = []
    
    # Cubic shell structure
    # 1 center
    positions2.append([0, 0, 0])
    
    # 6 face centers
    a1 = 1.0
    for coord in range(3):
        for sign in [-1, 1]:
            p = [0, 0, 0]
            p[coord] = sign * a1
            positions2.append(p)
    
    # 12 edge centers
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                # Edge centers: two coords are +/-a, one is 0
                pass
    
    # Let me add 12 more at edge-like positions
    a2 = 1.4
    edge_positions = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            edge_positions.append([sx * a2, sy * a2, 0])
            edge_positions.append([sx * a2, 0, sy * a2])
            edge_positions.append([0, sx * a2, sy * a2])
    # That's 24, take 12
    edge_positions = edge_positions[:12]
    positions2.extend(edge_positions)
    
    # Need 38 - 13 = 25 more
    n_remaining2 = 38 - len(positions2)
    if n_remaining2 > 0:
        r3 = 2.2
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n_remaining2):
            z = 1 - 2 * (i + 0.5) / n_remaining2
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / golden
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions2.append([x * r3, y * r3, z * r3])
    
    pos2 = np.array(positions2, dtype=float)
    
    try:
        result2 = minimize(
            _lj_energy,
            pos2,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result2.success or result2.fun < best_energy:
            best_energy = result2.fun
            best_pos = result2.x
    except Exception:
        pass
    
    # Strategy 3: Random starts with relaxation
    for seed in [456, 789, 101, 202, 303]:
        np.random.seed(seed)
        # Random positions in a sphere
        pos3 = np.random.randn(38, 3) * 2.0
        # Ensure minimum distance by scaling if needed
        try:
            result3 = minimize(
                _lj_energy,
                pos3,
                jac=_lj_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result3.success or result3.fun < best_energy:
                best_energy = result3.fun
                best_pos = result3.x
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback: return the first relaxed structure or a simple lattice
        return np.array(positions[:N], dtype=float)
    
    # Ensure minimum distance of 0.5
    # If any pair is too close, nudge them apart
    pos_final = best_pos.copy()
    for _ in range(100):
        dists = pdist(pos_final)
        if np.min(dists) >= 0.5:
            break
        # Find the closest pair
        min_idx = np.argmin(dists)
        # Convert to pair indices
        n = len(pos_final)
        i = int((np.sqrt(1 + 8 * 2 * min_idx) - 1) / 4)
        j = int(min_idx - i * (2 * n - i - 3) / 2 - 1)
        # Nudge them apart
        diff = pos_final[i] - pos_final[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            diff = np.random.randn(3)
            norm = np.linalg.norm(diff)
        diff = diff / norm * 0.1
        pos_final[i] += diff
        pos_final[j] -= diff
    
    return pos_final
# EVOLVE-BLOCK-END
