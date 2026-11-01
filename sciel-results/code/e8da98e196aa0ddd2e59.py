"""Rewritten program for Lennard-Jones 55-atom global minimum search."""
import numpy as np
import time
import sys

N = 55

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions."""
    n = pos.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            r = np.sqrt(r2)
            if r < 1e-8:
                return 1e10
            r6 = r2 ** 3
            energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return energy


def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient for a set of positions."""
    n = pos.shape[0]
    energy = 0.0
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e10, np.zeros_like(pos)
            r = np.sqrt(r2)
            r6 = r2 ** 3
            r12 = r6 * r6
            # energy contribution: 4*(r^-12 - r^-6)
            e_ij = 4.0 * (1.0 / r12 - 1.0 / r6)
            energy += e_ij
            # gradient: d/dr of 4*(r^-12 - r^-6) * (rij/r)
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # grad_i = -dE/dr * rij/r = -4*(-12*r^-13 + 6*r^-7) * rij/r
            #        = 4*(12*r^-13 - 6*r^-7) * rij/r
            #        = (48/r^13 - 24/r^7) * rij
            # Since rij/r is unit vector, grad_i = (48/r^13 - 24/r^7) * rij
            # Wait, let me be more careful.
            # E = 4*(r^-12 - r^-6)
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # grad_i wrt pos[i]: dE/dpos[i] = dE/dr * dr/dpos[i] = dE/dr * (pos[i]-pos[j])/r = dE/dr * rij/r
            # So grad_i = (4*(-12*r^-13 + 6*r^-7)) * rij/r = (-48*r^-13 + 24*r^-7) * rij/r
            # = (-48/r^14 + 24/r^8) * rij  ... wait
            # rij/r has magnitude 1, direction from j to i.
            # grad_i = dE/dr * (rij/r) = 4*(-12/r^13 + 6/r^7) * (rij/r)
            #        = (-48/r^14 + 24/r^8) * rij
            # Hmm, let me just compute it directly.
            # dE/dpos[i] = 24*(2/r^13 - 1/r^7) * rij  ... let me re-derive
            # E_pair = 4*(r^-12 - r^-6) where r = |rij|
            # dE_pair/d(rij) = 4*(-12*r^-13 * d(r)/d(rij) + 6*r^-7 * d(r)/d(rij))
            # d(r)/d(rij) = rij/r
            # So dE_pair/d(rij) = 4*(-12*r^-13 + 6*r^-7) * rij/r
            # grad_i = dE_pair/d(rij) = 4*(-12/r^14 + 6/r^8) * rij  ... 
            # Wait: r^-13 * (1/r) = r^-14, r^-7 * (1/r) = r^-8
            # So grad_i = 4*(-12/r^14 + 6/r^8) * rij = (-48/r^14 + 24/r^8) * rij
            # grad_j = -grad_i
            factor = 4.0 * (-12.0 / (r2 * r6) + 6.0 / r6) / r
            # Let me just use: factor such that grad_i = factor * rij
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # grad_i = dE/dr * rij/r = 4*(-12*r^-13 + 6*r^-7) * rij / r
            # = 4*(-12/r^14 + 6/r^8) * rij
            # r^14 = r2^7, r^8 = r2^4
            g_ij = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4)) * rij
            grad[i] += g_ij
            grad[j] -= g_ij
    return energy, grad


def _make_mackay_icosahedron():
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    # Mackay icosahedron: 1 center + 12 first shell + 42 second shell
    # First shell: 12 vertices of icosahedron at distance ~1.0
    # Second shell: 42 atoms
    
    # Icosahedron vertices (12 vertices)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # First shell at radius r1 (tune to ~1.0 for LJ)
    r1 = 1.0
    first_shell = verts * r1
    
    # Second shell: 42 atoms. In Mackay icosahedron, the second shell consists of:
    # 30 atoms at face centers of the icosahedron (scaled), and 12 atoms at edge midpoints (scaled)
    # Actually, for Mackay icosahedron n=2: 
    # - 1 center
    # - 12 at first shell (icosahedron vertices)
    # - 30 at face centers of the icosahedron, at radius r2
    # - 12 at edge midpoints of the icosahedron, at radius r3
    # Total: 1 + 12 + 30 + 12 = 55
    
    # Face centers of icosahedron (20 faces)
    # The 20 faces of the standard icosahedron... let me compute them.
    # Actually, let me use a simpler approach: generate the second shell positions
    # by scaling the icosahedron geometry.
    
    # For a Mackay icosahedron, the second shell atoms are at:
    # - 30 atoms: centers of the 20 triangular faces, but that's only 20. 
    # Let me look this up more carefully.
    # 
    # Mackay icosahedron with n=2:
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms = 30 + 12
    # The 30 are at the centers of the 20 faces... no, that's 20.
    # 
    # Actually, the standard Mackay icosahedron for 55 atoms:
    # - 1 center
    # - 12 first shell
    # - 42 second shell
    # The 42 second-shell atoms are arranged as:
    #   - 30 atoms forming a dodecahedron-like arrangement
    #   - 12 atoms at the 12 "edge" positions
    #
    # Let me just use known coordinates or a simpler construction.
    # I'll use the fact that for LJ, we can start with an approximate icosahedral
    # structure and relax it.
    
    # Simple approach: place 42 atoms on a sphere of radius ~1.8 with some structure
    # Better: use the actual Mackay icosahedron geometry
    
    # The 20 face centers of the icosahedron:
    # Each face is defined by 3 vertices. Let me enumerate them.
    # For the standard icosahedron with the vertices above, the faces are:
    # This is getting complex. Let me use a different approach.
    
    # I'll create a reasonable initial guess and let relaxation fix it.
    # Place second shell atoms on a sphere with icosahedral symmetry.
    
    # 30 atoms: at the midpoints of the 30 edges of the icosahedron, scaled to radius ~1.6
    # 12 atoms: at positions corresponding to the 12 vertices but slightly larger radius
    
    # Edge midpoints of icosahedron:
    # The icosahedron has 30 edges. Let me find them.
    # Two vertices are connected if their distance is the edge length.
    edge_len = np.linalg.norm(verts[0] - verts[1])  # approximate
    # Actually, let me compute all pairwise distances
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    # Find edges (minimum non-zero distance)
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    
    # That gives 30 unit vectors. Scale to radius r2.
    r2 = 1.6
    second_shell_edges = np.array(edge_mids) * r2
    
    # 12 atoms: at the icosahedron vertex directions but at a larger radius
    r3 = 1.8
    second_shell_verts = verts * r3
    
    # Combine: center + first shell + second shell
    center = np.array([[0.0, 0.0, 0.0]])
    all_pos = np.vstack([center, first_shell, second_shell_edges, second_shell_verts])
    
    # We have 1 + 12 + 30 + 12 = 55 atoms
    return all_pos


def _relax(pos, max_iter=2000, tol=1e-10):
    """Simple gradient descent with adaptive step size for LJ relaxation."""
    pos = pos.copy()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    step = 0.01
    for it in range(max_iter):
        e, g = _lj_energy_grad(pos)
        if not np.isfinite(e):
            break
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
        
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Normalize gradient
        g_unit = g / g_norm
        
        # Adaptive step size
        new_pos = pos - step * g
        new_e = _lj_energy(new_pos)
        
        if np.isfinite(new_e) and new_e < e:
            pos = new_pos
            step *= 1.05  # increase step
            if step > 0.1:
                step = 0.1
        else:
            step *= 0.5
            if step < 1e-6:
                break
    
    return best_pos, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Start with Mackay icosahedron approximation
    try:
        mackay = _make_mackay_icosahedron()
        e0 = _lj_energy(mackay)
        if np.isfinite(e0):
            best_energy = e0
            best_pos = mackay.copy()
    except Exception:
        pass
    
    # If Mackay construction failed, use cubic lattice
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = _lj_energy(best_pos)
    
    # Relax the initial structure
    if time.time() < deadline:
        try:
            relaxed, e_rel = _relax(best_pos, max_iter=5000, tol=1e-12)
            if np.isfinite(e_rel) and e_rel < best_energy:
                best_energy = e_rel
                best_pos = relaxed
        except Exception:
            pass
    
    # Try multiple restarts with perturbations
    rng = np.random.RandomState(42)
    for restart in range(50):
        if time.time() > deadline - 15.0:
            break
        
        # Create perturbed version
        if restart == 0:
            pos = best_pos.copy()
        else:
            # Add random perturbation
            noise_scale = 0.1 + 0.05 * restart
            pos = best_pos + noise_scale * rng.randn(N, 3)
        
        # Ensure minimum distance constraint
        # Check and fix any violations
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Push apart overlapping atoms
            for _ in range(100):
                min_d = np.inf
                min_pair = None
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < min_d:
                            min_d = d
                            min_pair = (i, j)
                if min_d >= 0.5:
                    break
                i, j = min_pair
                direction = pos[i] - pos[j]
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                push = (0.5 - min_d) / 2.0 + 0.01
                pos[i] += push * direction
                pos[j] -= push * direction
        
        # Relax
        try:
            relaxed, e_rel = _relax(pos, max_iter=3000, tol=1e-12)
            if np.isfinite(e_rel) and e_rel < best_energy:
                best_energy = e_rel
                best_pos = relaxed
        except Exception:
            pass
    
    # Final check: ensure minimum distance
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5:
                # Push apart
                direction = best_pos[i] - best_pos[j]
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                push = (0.5 - d) / 2.0 + 0.01
                best_pos[i] += push * direction
                best_pos[j] -= push * direction
    
    return best_pos
