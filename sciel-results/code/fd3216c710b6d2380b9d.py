"""LJ38 global minimum search: FCC truncated octahedron construction + local relaxation."""
import numpy as np
import time
import sys

N = 38

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for given positions."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    # Exclude self-interactions
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3  # r^6
    r12 = r6 * r6
    # 4 * sum(r^-12 - r^-6)
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    r12 = r6 * r6
    inv_r6 = 1.0 / r6
    inv_r12 = 1.0 / r12
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    
    # Gradient: dE/dr_i = 4 * sum_j [12*r_i^-13 - 6*r_i^-7] * (r_i - r_j)/r_i
    # = 24 * sum_j [2*r_i^-14 - r_i^-8] * (r_i - r_j)
    # Using r2: r^-14 = r2^-7, r^-8 = r2^-4
    # dE/dpos_i = 24 * sum_j [(2/r2_j^7 - 1/r2_j^4)] * (pos_i - pos_j)
    # Note: the factor (pos_i - pos_j) already includes the direction
    
    # coeff[j] = 24 * (2/r2_j^7 - 1/r2_j^4) for each pair
    coeff = 24.0 * (2.0 / (r2 ** 7) - 1.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _fcc_truncated_octahedron(scale=1.12, distortion=0.0):
    """
    Construct a truncated octahedron from FCC lattice.
    
    The ideal FCC truncated octahedron for N=38 has:
    - 1 atom at center
    - 12 atoms at <100> directions (cuboctahedron vertices)
    - 14 atoms at <110> directions (square face centers)  
    - 12 atoms at <111> directions (hexagonal face centers)
    
    Actually, let me use the standard construction:
    FCC lattice points within a truncated octahedron of radius ~3.
    
    For LJ38, the structure is:
    - Shell 0: 1 atom at origin
    - Shell 1: 12 atoms at FCC neighbors (1/2, 1/2, 0) type positions
    - Shell 2: 14 atoms 
    - Shell 3: 12 atoms
    
    Let me use a more direct approach: generate FCC lattice points and select the 38 closest to center.
    """
    # FCC lattice: points at integer combinations of basis vectors
    # FCC basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Generate points in a cube and select those within truncated octahedron
    
    # Generate FCC lattice points
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC condition: i+j+k even
                if (i + j + k) % 2 == 0:
                    points.append((0.5 * i, 0.5 * j, 0.5 * k))
    
    points = np.array(points)
    
    # Sort by distance from origin
    dists = np.sum(points * points, axis=1)
    idx = np.argsort(dists)
    
    # Take the 38 closest
    selected = points[idx[:38]]
    
    # Apply scale and distortion
    # Distortion: stretch along z-axis slightly
    selected[:, 2] *= (1.0 + distortion)
    selected *= scale
    
    return selected

def _relax_lbfqgs(pos, max_iter=2000, tol=1e-10):
    """
    Simple L-BFGS-like relaxation using scipy if available, else gradient descent.
    """
    try:
        from scipy.optimize import minimize
        def obj(x):
            p = x.reshape(-1, 3)
            return _lj_energy(p)
        def grad(x):
            p = x.reshape(-1, 3)
            e, g = _lj_energy_grad(p)
            return g.flatten()
        
        res = minimize(obj, pos.flatten(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        return res.x.reshape(-1, 3)
    except ImportError:
        # Fallback: simple gradient descent with adaptive step
        pos = pos.copy()
        lr = 0.01
        for i in range(max_iter):
            e, g = _lj_energy_grad(pos)
            if np.max(np.abs(g)) < tol:
                break
            pos -= lr * g
            # Adaptive learning rate
            e2, g2 = _lj_energy_grad(pos)
            if e2 > e:
                lr *= 0.5
            elif i % 100 == 0:
                lr *= 1.1
        return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Try multiple initial configurations
    configs = []
    
    # 1. FCC truncated octahedron with various scales and distortions
    for scale in [1.09, 1.10, 1.11, 1.12, 1.13, 1.14]:
        for distortion in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            pos = _fcc_truncated_octahedron(scale=scale, distortion=distortion)
            configs.append(pos)
    
    # 2. Simple cubic fragment (original)
    side = int(np.ceil(N ** (1 / 3)))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    configs.append(np.array(pts[:N], dtype=float) * 1.12)
    
    # 3. Icosahedral structure (may get trapped but try anyway)
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [-1, phi, 0],
    ])
    # This isn't quite right for 38 atoms; skip icosahedral for now
    
    # Relax each configuration
    for pos in configs:
        if time.time() > deadline:
            break
        
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25:  # 0.5^2
            continue
        
        try:
            relaxed = _relax_lbfqgs(pos, max_iter=1000, tol=1e-9)
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if best_pos is not None and time.time() < deadline - 20:
        try:
            polished = _relax_lbfqgs(best_pos, max_iter=5000, tol=1e-12)
            e = _lj_energy(polished)
            if e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: return the original seed
        side = int(np.ceil(N ** (1 / 3)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(pts[:N], dtype=float) * 1.12
    
    return best_pos
