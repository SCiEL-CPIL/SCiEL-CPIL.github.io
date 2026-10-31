"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for a set of positions."""
    n = pos.shape[0]
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=2)
    # Avoid division by zero for i==j
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix, we get 2x the off-diagonal sum, so divide by 2
    e_matrix = 4.0 * (r12 - r6)
    e_matrix[np.diag_indices_from(e_matrix)] = 0.0
    energy = 0.5 * np.sum(e_matrix)
    
    # Gradient: dE/dr_i = sum_{j!=i} 4*(-12*r^-14 + 6*r^-8) * (r_i - r_j)/r
    # In vector form: grad_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (r_i - r_j) / r
    # = sum_j 4*(-12*r^-14 + 6*r^-8) * rij[i,j] / r
    # Since rij[i,j] = r_i - r_j, and r = |rij|
    # Let's compute coefficient c_ij = 4*(-12*r^-14 + 6*r^-8) / r = 4*(-12*r^-13 + 6*r^-7)
    # Actually, dE/d(r_i) = sum_{j!=i} 4 * d(r_ij^-12 - r_ij^-6)/d(r_i)
    # d(r^-12)/d(r_i) = -12*r^-13 * (r_i - r_j)/r = -12*r^-14 * (r_i - r_j)
    # Wait, r = |r_i - r_j|, dr/dr_i = (r_i - r_j)/r
    # d(r^-12)/dr_i = -12*r^-13 * (r_i - r_j)/r = -12*r^-14 * (r_i - r_j)
    # d(r^-6)/dr_i = -6*r^-7 * (r_i - r_j)/r = -6*r^-8 * (r_i - r_j)
    # So d(r^-12 - r^-6)/dr_i = (-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # Multiply by 4: 4*(-12*r^-14 + 6*r^-8) * (r_i - r_j)
    
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    # coefficient for each pair (i,j): 4*(-12*r^-14 + 6*r^-8)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    coeff[np.diag_indices_from(coeff)] = 0.0
    # grad[i] = sum_j coeff[i,j] * rij[i,j]
    grad = np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return energy, grad


def _fcc_truncated_octahedron():
    """
    Construct the 38-atom FCC truncated octahedron.
    
    The structure consists of:
    - 1 atom at the center (0,0,0)
    - 12 atoms at the vertices of a cube: (±a, ±a, ±a)
    - 24 atoms at the centers of the faces of a larger cube... 
    
    Actually, let's use the standard FCC construction.
    An FCC truncated octahedron with 38 atoms can be described as:
    Take an FCC lattice with lattice parameter a. The nearest neighbor distance is a/sqrt(2).
    For LJ, we want the nearest neighbor distance to be 2^(1/6) ≈ 1.12246.
    So a/sqrt(2) = 2^(1/6) => a = sqrt(2) * 2^(1/6) = 2^(2/3) ≈ 1.5874.
    
    The 38-atom truncated octahedron in FCC corresponds to:
    - 1 atom at origin
    - 12 atoms at FCC nearest neighbors: permutations of (a/2, a/2, 0) with signs
      Actually, FCC neighbors are at (a/2, a/2, 0) and permutations. There are 12 such.
    - 24 atoms at FCC next-nearest neighbors: permutations of (a, 0, 0) and (a/2, a/2, a) ... 
      
    Let me use a more direct approach. The 38-atom cluster is a truncated octahedron
    with 8 hexagonal faces and 6 square faces. 
    
    Standard coordinates (from literature):
    Let a be the cube edge length. The vertices are at:
    - 6 square faces: (±a, 0, 0), (0, ±a, 0), (0, 0, ±a) -> 6 atoms? No.
    
    Let me use the known construction. The 38-atom FCC truncated octahedron has:
    - 1 center atom
    - 12 atoms forming a cuboctahedron (the "inner" shell)
    - 24 atoms forming the outer shell
    
    Actually, a simpler way: use the fact that it's an FCC cluster.
    Take all FCC lattice points within a certain radius.
    
    FCC lattice points: all integer combinations of basis vectors
    b1 = (a/2, a/2, 0), b2 = (a/2, 0, a/2), b3 = (0, a/2, a/2)
    
    Or equivalently, all points (i*a/2, j*a/2, k*a/2) where i,j,k are all even or all odd... 
    No, FCC points are (i*a, j*a, k*a) + FCC basis.
    
    Let me just use the known energy and construct via a known recipe.
    The 38-atom truncated octahedron can be built as:
    - 1 atom at (0,0,0)
    - 12 atoms at the 12 edges of a cube of side a: permutations of (±a/2, ±a/2, 0)
    - 24 atoms at... 
    
    Let me use a different approach: generate FCC points and select the 38 closest to origin.
    """
    a = 2.0 ** (2.0 / 3.0)  # lattice parameter for NN distance = 2^(1/6)
    
    # Generate FCC lattice points
    # FCC: all points (i, j, k) * a where i,j,k integers AND (i+j+k) even? No.
    # FCC primitive cell: points at (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0) + (na, ma, ka)
    # So all points: (na + dx, ma + dy, ka + dz) * a where (dx,dy,dz) in {(0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0)}
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append((x, y, z))
    
    points = np.array(points)
    # Remove duplicates
    points = np.unique(np.round(points, 10), axis=0)
    
    # Sort by distance from origin
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:38]]
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    return points


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with the FCC truncated octahedron
    pos = _fcc_truncated_octahedron()
    
    # Compute initial energy
    e0, g0 = _lj_energy_and_grad(pos)
    
    # Tight L-BFGS relaxation to polish
    try:
        result = minimize(
            fun=lambda x: _lj_energy_and_grad(x)[0],
            x0=pos.flatten(),
            jac=lambda x: _lj_energy_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        if result.success or result.fun < e0:
            pos_relaxed = result.x.reshape(N, 3)
            e1, _ = _lj_energy_and_grad(pos_relaxed)
            if e1 < e0:
                pos = pos_relaxed
                e0 = e1
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5.0:
        return pos
    
    # Try a second, even tighter polish
    try:
        result2 = minimize(
            fun=lambda x: _lj_energy_and_grad(x)[0],
            x0=pos.flatten(),
            jac=lambda x: _lj_energy_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-14, 'ftol': 1e-16}
        )
        if result2.success or result2.fun < e0:
            pos2 = result2.x.reshape(N, 3)
            e2, _ = _lj_energy_and_grad(pos2)
            if e2 < e0:
                pos = pos2
                e0 = e2
    except Exception:
        pass
    
    return pos

# EVOLVE-BLOCK-END
