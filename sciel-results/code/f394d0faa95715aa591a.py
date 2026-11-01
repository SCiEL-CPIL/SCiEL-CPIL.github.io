"""Lennard-Jones cluster global minimum, 55 atoms."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from itertools import combinations

N = 55
DEADLINE = 145.0  # Safety margin before 150s kill

def _lj_energy(positions):
    """Compute total Lennard-Jones energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = np.sum((positions[i] - positions[j]) ** 2)
            if r2 < 1e-12:
                return 1e10
            r = np.sqrt(r2)
            r6 = r ** 6
            r12 = r6 * r6
            energy += 4.0 * (r12 / (r6 * r6) - 1.0 / r6)
    return energy

def _lj_energy_and_grad(positions):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r2 = np.dot(diff, diff)
            if r2 < 1e-12:
                return 1e10, grad
            r = np.sqrt(r2)
            r6 = r ** 6
            r12 = r6 * r6
            # E = 4*(r^-12 - r^-6) = 4*(r12/r12 - 1/r6)
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # dE/dxi = dE/dr * (xi-xj)/r
            dEdr = 4.0 * (-12.0 / (r * r12) + 6.0 / (r * r6))
            dE = dEdr * diff
            energy += 4.0 * (1.0 / r12 - 1.0 / r6)
            grad[i] += dE
            grad[j] -= dE
    return energy, grad

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    min_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d

def _make_mackay_icosahedron():
    """
    Construct the two-shell Mackay icosahedron for N=55.
    Inner shell: 12 atoms on a regular icosahedron.
    Outer shell: 42 atoms on a larger icosahedral surface.
    
    The Mackay icosahedron M(2,2,2) has:
    - 1 center atom
    - 12 atoms on inner icosahedron (radius r1)
    - 30 atoms on middle positions (radius r2) 
    - 12 atoms on outer icosahedron vertices (radius r3)
    Total: 1 + 12 + 30 + 12 = 55
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (unit)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # For Mackay icosahedron M(2,2,2):
    # The radii are proportional to sqrt(1), sqrt(5), sqrt(9) for shells
    # Actually, for the standard Mackay icosahedron with n=2:
    # Shell 1 (inner 12): radius ~ 1.0 (relative)
    # Shell 2 (middle 30): radius ~ sqrt(5) ≈ 2.236 (relative)  
    # Shell 3 (outer 12): radius ~ 3.0 (relative)
    # But we need to scale so that nearest-neighbor distances are ~1.12
    
    # Let's use known approximate radii for the 55-atom Mackay icosahedron
    # The optimal configuration has specific radial distances
    # Using literature values (scaled to sigma=1):
    # Inner shell (12 atoms): r1 ≈ 1.0
    # Middle shell (30 atoms): r2 ≈ 2.0  
    # Outer shell (12 atoms): r3 ≈ 3.0
    # These need to be optimized; let's start with reasonable values and relax
    
    # Actually, for the Mackay icosahedron, the atoms are arranged as:
    # - 12 vertices of an icosahedron at radius a
    # - 30 face-center positions at radius b  
    # - 12 vertices at radius c
    # with specific ratios. Let me use a simpler construction.
    
    # Standard Mackay icosahedron M(n,n,n) for n=2:
    # Uses icosahedral symmetry. The positions can be generated from
    # the icosahedron vertices and their combinations.
    
    # Let's use a known parametrization:
    # Inner 12: icosahedron vertices at radius r1
    # Middle 30: midpoints of edges of the icosahedron, scaled to radius r2
    # Outer 12: icosahedron vertices at radius r3
    
    # Edge midpoints of icosahedron:
    ico_edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 1.2:  # Adjacent vertices (edge length of unit icosahedron)
                ico_edges.append((i, j))
    
    # Get unique edge midpoints
    midpoints = []
    for i, j in ico_edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    
    # Remove duplicates
    unique_mids = []
    for m in midpoints:
        is_dup = False
        for um in unique_mids:
            if np.linalg.norm(m - um) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_mids.append(m)
    
    unique_mids = np.array(unique_mids)
    
    # Now construct with proper radii
    # For the 55-atom Mackay icosahedron, typical radii (in LJ units):
    # r1 ≈ 1.0, r2 ≈ 2.0, r3 ≈ 3.0 (these will be optimized by relaxation)
    
    r1 = 1.0
    r2 = 2.0  
    r3 = 3.0
    
    positions = np.zeros((55, 3))
    # Center atom
    positions[0] = [0, 0, 0]
    # Inner shell (12)
    positions[1:13] = ico_verts * r1
    # Middle shell (30)
    positions[13:43] = unique_mids[:30] * r2
    # Outer shell (12)
    positions[43:55] = ico_verts * r3
    
    return positions

def _make_cubic_lattice():
    """Simple cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def _make_fcc_like():
    """FCC-like arrangement for better initial guess."""
    # Use a small FCC lattice fragment
    a = 1.12 * np.sqrt(2)  # FCC lattice parameter for nearest-neighbor distance 1.12
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # FCC basis
                points.append([i * a, j * a, k * a])
                points.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                points.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                points.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    
    points = np.array(points, dtype=float)
    # Sort by distance from center
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _relax(positions, tol=1e-6, maxiter=500):
    """Relax positions using L-BFGS."""
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'gtol': tol, 'maxiter': maxiter, 'maxls': 50})
    if result.success or result.fun < 1e6:
        return result.x.reshape(-1, 3), result.fun
    return positions, _lj_energy(positions)

def _perturb(positions, scale=0.1, seed=None):
    """Add small random perturbation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return positions + rng.randn(*positions.shape) * scale

def construct():
    start_time = time.time()
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Mackay icosahedron (best guess for global minimum)
    try:
        mackay = _make_mackay_icosahedron()
        candidates.append(('mackay', mackay))
    except Exception:
        pass
    
    # 2. FCC-like arrangement
    try:
        fcc = _make_fcc_like()
        candidates.append(('fcc', fcc))
    except Exception:
        pass
    
    # 3. Cubic lattice
    try:
        cubic = _make_cubic_lattice()
        candidates.append(('cubic', cubic))
    except Exception:
        pass
    
    # 4. Perturbed Mackay variants
    for i in range(5):
        try:
            mackay_pert = _perturb(_make_mackay_icosahedron(), scale=0.05 * (i + 1), seed=i)
            candidates.append((f'mackay_pert_{i}', mackay_pert))
        except Exception:
            pass
    
    # Relax each candidate and track best
    best_energy = np.inf
    best_positions = None
    
    # Sort candidates by initial energy for efficiency
    candidate_energies = []
    for name, pos in candidates:
        try:
            e = _lj_energy(pos)
            candidate_energies.append((e, name, pos))
        except Exception:
            pass
    
    candidate_energies.sort(key=lambda x: x[0])
    
    for e0, name, pos in candidate_energies:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        try:
            # Stage 1: Coarse relaxation
            relaxed, e1 = _relax(pos, tol=1e-4, maxiter=200)
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                if e1 < best_energy:
                    best_energy = e1
                    best_positions = relaxed
                continue
            
            # Stage 2: Fine relaxation
            relaxed2, e2 = _relax(relaxed, tol=1e-6, maxiter=300)
            elapsed = time.time() - start_time
            if e2 < best_energy:
                best_energy = e2
                best_positions = relaxed2
            
            if elapsed > DEADLINE - 3:
                continue
            
            # Stage 3: Ultra-fine polish
            relaxed3, e3 = _relax(relaxed2, tol=1e-8, maxiter=500)
            if e3 < best_energy:
                best_energy = e3
                best_positions = relaxed3
                
        except Exception:
            continue
    
    # If we have time, try perturbation of best solution to escape local minima
    elapsed = time.time() - start_time
    if best_positions is not None and elapsed < DEADLINE - 15:
        for i in range(10):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            try:
                pert = _perturb(best_positions, scale=0.02 * (i + 1), seed=100 + i)
                relaxed, e = _relax(pert, tol=1e-6, maxiter=300)
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed
            except Exception:
                continue
    
    # Final validation
    if best_positions is None:
        best_positions = _make_cubic_lattice()
    
    # Ensure minimum distance constraint
    min_d = _min_distance(best_positions)
    if min_d < 0.5:
        # Spread out atoms if too close
        scale = 0.5 / min_d
        best_positions = best_positions * scale
    
    return best_positions
