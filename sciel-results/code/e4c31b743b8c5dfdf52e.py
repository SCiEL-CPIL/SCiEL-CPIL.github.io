"""
LJ38 cluster optimisation: construct a near-optimal truncated-octahedron
geometry and refine with L-BFGS relaxation.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

# Deadline guard
_START = time.time()
_TIME_LIMIT = 110.0  # seconds; leaves margin before 120 s kill


def _elapsed():
    return time.time() - _START


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6)."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)  # avoid self-interaction
    r6 = r2 ** 3
    r12 = r6 * r6
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy


def _lj_grad(pos):
    """Gradient of LJ energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero on diagonal
    r6 = r2 ** 3
    # dE/dr2 = 4 * (-6/r2^7 + 12/r2^10) * ... 
    # Actually: E = 4*(r^-12 - r^-6), dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dxi = dE/dr * dx/dr = dE/dr * (xi - xj)/r
    # Using r2: dE/dr2 = 2*r*dE/dr = 2*r*4*(-12*r^-13 + 6*r^-7)
    #           = 8*(-12*r^-12 + 6*r^-6) = 8*(-12/r12 + 6/r6)
    # So grad_i = sum_j 8*(-12/r12_ij + 6/r6_ij) * (xi - xj)
    r6_inv = 1.0 / r6
    r12_inv = r6_inv * r6_inv
    # coefficient for each pair: 8*(-12/r12 + 6/r6)
    coeff = 8.0 * (-12.0 * r12_inv + 6.0 * r6_inv)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad


def _enforce_min_separation(pos, min_dist=0.501):
    """Push atoms apart if any pair is closer than min_dist."""
    for _ in range(200):
        n = pos.shape[0]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        # Find violations
        mask = (r < min_dist) & (r > 1e-12)
        if not np.any(mask):
            break
        # Push apart
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-12:
                        # random small push
                        d = np.random.randn(3) * 0.01
                        dist = np.linalg.norm(d)
                    correction = (min_dist - dist) * d / dist * 0.5
                    pos[i] += correction
                    pos[j] -= correction
    return pos


def _build_fcc_truncated_octahedron(scale=1.1225):
    """
    Build 38-atom FCC truncated octahedron (14+12+12 or similar shell structure).
    
    The global minimum of LJ38 is known to be an FCC fragment.
    A common construction:
    - Core: 1 atom at origin
    - 1st shell: 12 atoms at FCC nearest-neighbour positions
    - 2nd shell: 25 atoms at next FCC positions (total 38)
    
    FCC lattice vectors with nearest-neighbour distance = scale:
    The 12 NN positions in FCC (relative to origin) are permutations of (a/2, a/2, 0)
    where a = scale * sqrt(2) since NN distance = a/sqrt(2) = scale => a = scale*sqrt(2).
    
    Actually for FCC: atoms at (0,0,0) and face-centers. NN distance = a/sqrt(2).
    So if we want NN distance = d, then a = d*sqrt(2).
    
    12 NN positions: all permutations of (±a/2, ±a/2, 0) and (±a/2, 0, ±a/2) and (0, ±a/2, ±a/2)
    That gives 12 positions.
    
    Next shell (2nd coordination): positions at distance a/sqrt(2)*sqrt(2) = a, or other FCC positions.
    In FCC, the 2nd shell has 6 atoms at (±a, 0, 0) etc. (cubic positions), and 24 at (±a, ±a/2, ±a/2)... 
    Actually let's just generate FCC points and pick the 38 closest to origin.
    """
    d = scale  # desired NN distance
    a = d * np.sqrt(2.0)  # cubic lattice constant
    
    # Generate FCC points in a box large enough
    # FCC points: (i*a, j*a, k*a) + (0,0,0) and (i*a+a/2, j*a+a/2, k*a) etc.
    # Standard FCC: points at (i,j,k)*a where i+j+k even, plus (i+0.5, j+0.5, k+0.5)*a
    # Let's just generate all FCC lattice points in range [-2a, 2a]
    
    points = []
    extent = int(np.ceil(2.0 * a / a)) + 2  # about 4
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # Conventional FCC: atom at (i,j,k)*a if i+j+k even
                if (i + j + k) % 2 == 0:
                    points.append(np.array([i, j, k]) * a)
                # Face-center: (i+0.5, j+0.5, k)*a etc.
                # Actually simpler: FCC = simple cubic with basis (0,0,0) and (0.5,0.5,0)*a... 
                # Let me use: all points (x,y,z) where x,y,z are multiples of a/2 and x+y+z is a multiple of a
                # i.e., 2x/a + 2y/a + 2z/a is even... 
    
    # Simpler approach: generate on the FCC lattice directly
    # FCC lattice: all vectors (n1, n2, n3) * a/2 where n1+n2+n3 is even
    pts = []
    R = int(np.ceil(3.0 * a / (a/2)))  # radius in units of a/2
    for n1 in range(-R, R + 1):
        for n2 in range(-R, R + 1):
            for n3 in range(-R, R + 1):
                if (n1 + n2 + n3) % 2 == 0:
                    pt = np.array([n1, n2, n3]) * (a / 2.0)
                    pts.append(pt)
    
    pts = np.array(pts)
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    # Take 38 closest
    cluster = pts[:38].copy()
    
    # Center of mass to origin
    cluster -= np.mean(cluster, axis=0)
    
    return cluster


def _relax(pos, max_time=30.0):
    """L-BFGS relaxation with time limit."""
    t0 = time.time()
    
    def objective(x):
        if time.time() - t0 > max_time:
            raise TimeoutError
        return _lj_energy(x)
    
    def gradient(x):
        if time.time() - t0 > max_time:
            raise TimeoutError
        return _lj_grad(x)
    
    try:
        result = minimize(
            objective, pos, jac=gradient, method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 5000, 'maxls': 50,
                     'gtol': 1e-8, 'ftol': 1e-10}
        )
        return result.x
    except (TimeoutError, Exception):
        return pos


def construct():
    """
    Construct LJ38 cluster near global minimum.
    Returns array of shape (38, 3).
    """
    # Build initial FCC truncated octahedron
    # The optimal NN distance for LJ is 2^(1/6) ≈ 1.12246
    scale = 2.0 ** (1.0 / 6.0)
    
    pos = _build_fcc_truncated_octahedron(scale)
    
    # Ensure minimum separation
    pos = _enforce_min_separation(pos)
    
    # Relax with L-BFGS
    remaining = _remaining()
    if remaining > 10.0:
        pos = _relax(pos, max_time=min(remaining - 5.0, 60.0))
        pos = _enforce_min_separation(pos)
    
    # If we have more time, try a second relaxation from slightly perturbed start
    if _remaining() > 15.0:
        # Try with a slightly different scale
        for s in [scale * 0.99, scale * 1.01]:
            if _remaining() < 10.0:
                break
            pos2 = _build_fcc_truncated_octahedron(s)
            pos2 = _enforce_min_separation(pos2)
            pos2 = _relax(pos2, max_time=min(_remaining() - 5.0, 40.0))
            pos2 = _enforce_min_separation(pos2)
            if _lj_energy(pos2) < _lj_energy(pos):
                pos = pos2
    
    # Final separation check
    pos = _enforce_min_separation(pos)
    
    return pos
