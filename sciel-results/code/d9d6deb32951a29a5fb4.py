"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct the exact 38-atom FCC truncated octahedron using topology-snap strategy.
    
    The FCC truncated octahedron for N=38 is generated from FCC lattice points
    with integer coordinates (i,j,k) where i+j+k is even, scaled appropriately.
    
    The 38 points are:
    - 1 center: (0,0,0)
    - 12 nearest neighbors: permutations of (±1,±1,0) in units of a/2
    - 18 next-nearest: permutations of (±2,0,0) and (±1,±1,±1) with appropriate signs
    - 7 more to complete the truncated octahedron
    
    We generate all FCC lattice points within radius, sort by distance, and take
    the 38 closest to form the truncated octahedron.
    """
    
    # FCC lattice constant for LJ equilibrium
    # Nearest neighbor distance = 2^(1/6)
    # In FCC, nearest neighbor = a/sqrt(2)
    # So a = 2^(1/6) * sqrt(2) = 2^(2/3)
    a = 2.0 ** (2.0 / 3.0)
    half_a = a / 2.0
    
    # Generate FCC lattice points: (i,j,k) * half_a where i+j+k is even
    # Search range to ensure we capture all 38 atoms of the truncated octahedron
    max_coord = 3  # This should be sufficient for 38 atoms
    
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pos = np.array([i, j, k], dtype=float) * half_a
                    r = np.linalg.norm(pos)
                    candidates.append((r, pos))
    
    # Sort by distance from origin and take the 38 closest
    candidates.sort(key=lambda x: x[0])
    
    # Take exactly N=38 points
    points = np.array([c[1] for c in candidates[:N]])
    
    # Center the cluster at origin
    center = points.mean(axis=0)
    points = points - center
    
    # Apply minimal L-BFGS relaxation to remove any floating-point drift
    def lj_energy_and_grad(coords):
        n = len(coords)
        energy = 0.0
        grad = np.zeros_like(coords)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = coords[i] - coords[j]
                r2 = np.dot(r_vec, r_vec)
                if r2 < 1e-12:
                    continue
                r6 = r2 ** 3
                r12 = r6 ** 2
                r14 = r2 ** 7
                r8 = r2 ** 4
                coeff = 4.0 * (12.0 / r14 - 6.0 / r8)
                grad[i] += coeff * r_vec
                grad[j] -= coeff * r_vec
                energy += 4.0 * (1.0 / r12 - 1.0 / r6)
        return energy, grad
    
    # Single tight relaxation pass
    result = minimize(
        lambda x: lj_energy_and_grad(x.reshape(-1, 3))[0],
        points.flatten(),
        jac=lambda x: lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 10, 'gtol': 1e-12, 'ftol': 1e-14}
    )
    
    final_positions = result.x.reshape(-1, 3)
    
    # Verify minimum separation constraint
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(final_positions[i] - final_positions[j])
            if dist < 0.5:
                # Should not happen, but enforce if needed
                final_positions[j] += 0.01 * (final_positions[j] - final_positions[i]) / dist
    
    return final_positions
# EVOLVE-BLOCK-END
