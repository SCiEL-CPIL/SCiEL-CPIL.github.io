"""
Dzugutov cluster global minima search using continuous softness annealing
with parallel diverse polytetrahedral starts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A0, B0, a0, b0, c0, d0, m0 = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16

def _soft_dzugutov_energy(pos, c_soft, d_soft):
    """Compute Dzugutov energy with softened exponential parameters."""
    A, B, a, b, m = A0, B0, a0, b0, m0
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    dist = np.triu(dist, 1).flatten()
    
    energy = 0.0
    mask_a = dist < a
    mask_b = dist < b
    
    if np.any(mask_a):
        ra = dist[mask_a]
        # Clip to avoid division by zero or negative
        ra = np.clip(ra, 0.1, a - 1e-6)
        energy += np.sum((ra**(-m) - B) * np.exp(c_soft / (ra - a)))
    if np.any(mask_b):
        rb = dist[mask_b]
        rb = np.clip(rb, 0.1, b - 1e-6)
        energy += np.sum(B * np.exp(d_soft / (rb - b)))
    return energy

def _soft_dzugutov_gradient(pos, c_soft, d_soft):
    """Compute gradient of softened Dzugutov energy."""
    A, B, a, b, m = A0, B0, a0, b0, m0
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-6:
                continue
            
            dE_dr = 0.0
            if r < a:
                # d/dr [(r^-m - B) * exp(c/(r-a))]
                exp_term = np.exp(c_soft / (r - a))
                dE_dr += (-m * r**(-m - 1)) * exp_term + (r**(-m) - B) * exp_term * (-c_soft / (r - a)**2)
            if r < b:
                # d/dr [B * exp(d/(r-b))]
                exp_term = np.exp(d_soft / (r - b))
                dE_dr += B * exp_term * (-d_soft / (r - b)**2)
            
            grad[i] -= dE_dr * rij / r
            grad[j] += dE_dr * rij / r
    
    return grad

def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral structure with n atoms."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron (unit radius)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Scale factor for inner shell
    inner_scale = 1.0
    
    positions = []
    # Center atom
    if n >= 1:
        positions.append(np.zeros(3))
    
    # 12 vertices
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v * inner_scale)
    
    # 30 edge midpoints (for larger clusters)
    if n > 13:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 0.9 < d < 1.3:  # Adjacent vertices
                    edges.append((i, j))
        
        edge_scale = inner_scale * 0.85
        for i, j in edges:
            if len(positions) >= n:
                break
            mid = (verts[i] + verts[j]) / 2 * edge_scale
            positions.append(mid)
    
    # Outer shell for even larger clusters
    if n > 43:
        outer_radius = inner_scale * 1.8
        # 62 vertices of a dodecahedron-like shell
        # Use icosahedron vertices scaled up
        for v in verts:
            if len(positions) >= n:
                break
            positions.append(v * outer_radius)
        
        # Additional points
        while len(positions) < n:
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = outer_radius * np.array([
                sin_phi * np.cos(theta),
                sin_phi * np.sin(theta),
                cos_phi
            ])
            if all(np.linalg.norm(p - q) >= 0.5 for q in positions):
                positions.append(p)
            else:
                p = p + 0.1 * rng.randn(3)
                positions.append(p)
    
    # Fill any remaining with random points on surface
    while len(positions) < n:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        r = inner_scale * (1.5 + 0.3 * rng.random())
        p = r * np.array([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi
        ])
        positions.append(p)
    
    return np.array(positions[:n])

def _generate_compact_random(n, rng):
    """Generate a compact random cluster with minimum distance constraint."""
    radius = 1.3 * n ** (1.0 / 3.0)
    positions = [np.zeros(3)]
    
    attempts = 0
    while len(positions) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_d = min(np.linalg.norm(p - q) for q in positions)
            if min_d >= 0.55:
                positions.append(p)
        attempts += 1
    
    # Force-fill if needed
    while len(positions) < n:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        r = rng.uniform(0.5, radius)
        p = r * np.array([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi
        ])
        positions.append(p)
    
    return np.array(positions[:n])

def _softness_anneal(positions, c_start, d_start, c_end, d_end, 
                     n_steps=80, lr_start=0.02, lr_end=0.001,
                     deadline=None):
    """
    Continuous annealing of softness parameters c and d.
    Starts with very soft (small c, d) and anneals to exact values.
    Uses gradient descent with adaptive step size.
    """
    pos = positions.copy().astype(float)
    n = len(pos)
    
    best_pos = pos.copy()
    best_energy = _soft_dzugutov_energy(pos, c_end, d_end)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal c and d from soft to exact
        t = step / max(1, n_steps - 1)
        # Use exponential annealing for smoother transition
        c_cur = c_start + (c_end - c_start) * (1 - np.exp(-3 * t))
        d_cur = d_start + (d_end - d_start) * (1 - np.exp(-3 * t))
        
        # Adaptive learning rate
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute gradient
        try:
            grad = _soft_dzugutov_gradient(pos, c_cur, d_cur)
        except:
            break
        
        # Gradient descent step
        pos_new = pos - lr * grad
        
        # Enforce minimum distance constraint via projection
        for _ in range(3):  # A few iterations of distance enforcement
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(pos_new[i] - pos_new[j])
                    if d < 0.5 and d > 1e-8:
                        # Push apart
                        direction = (pos_new[i] - pos_new[j]) / d
                        push = (0.5 - d) / 2
                        pos_new[i] += push * direction
                        pos_new[j] -= push * direction
                        changed = True
            if not changed:
                break
        
        # Accept if valid
        if all(np.linalg.norm(pos_new[i] - pos_new[j]) >= 0.49 for i in range(n) for j in range(i+1, n)):
            pos = pos_new
        
        # Track best with exact potential
        if step % 10 == 0 or step == n_steps - 1:
            try:
                e_exact = _soft_dzugutov_energy(pos, c_end, d_end)
                if e_exact < best_energy:
                    best_energy = e_exact
                    best_pos = pos.copy()
            except:
                pass
    
    return best_pos, best_energy

def _worker(args):
    """Worker: generate diverse start, anneal softness, then polish."""
    seed_idx, n, rng_seed, deadline = args
    rng = np.random.default_rng(rng_seed)
    
    # Choose initial structure type
    start_type = seed_idx % 3
    if start_type == 0:
        positions = _generate_mackay_icosahedron(n, rng)
    elif start_type == 1:
        positions = _generate_compact_random(n, rng)
    else:
        # Mixed: icosahedron with random perturbation
        positions = _generate_mackay_icosahedron(n, rng)
        positions += 0.05 * rng.randn(*positions.shape)
    
    # Run softness annealing
    # Start very soft (small c, d) to have smooth landscape
    c_start, d_start = 0.1, 0.02
    c_end, d_end = c0, d0
    
    try:
        best_pos, best_energy = _softness_anneal(
            positions, c_start, d_start, c_end, d_end,
            n_steps=60, lr_start=0.015, lr_end=0.0005,
            deadline=deadline
        )
    except:
        best_pos = positions
        best_energy = _soft_dzugutov_energy(positions, c_end, d_end)
    
    # Final polish with exact potential using the provided relax
    try:
        e, polished = relax(best_pos, 'dzugutov')
        if e < best_energy:
            best_pos = polished
            best_energy = e
    except:
        pass
    
    # One more polish pass
    try:
        e, polished = relax(best_pos, 'dzugutov')
        if e < best_energy:
            best_pos = polished
            best_energy = e
    except:
        pass
    
    return best_energy, best_pos

def construct(instance):
    """Main entry point: find low-energy Dzugutov cluster."""
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    # Number of parallel workers (max 4 threads)
    n_workers = 4
    
    # Create worker arguments with different seeds
    rng = np.random.default_rng(n * 1000 + 42)
    worker_args = [(i, n, int(rng.integers(0, 2**31)), deadline) for i in range(n_workers)]
    
    best_energy = float('inf')
    best_positions = None
    
    # Run parallel annealing
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker, args) for args in worker_args]
        
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 20:
                break
            try:
                energy, positions = future.result(timeout=5)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                pass
    
    # Final refinement with remaining time
    if best_positions is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            try:
                energy, positions = relax(best_positions, potential)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                break
    
    # Fallback if nothing found
    if best_positions is None:
        rng = np.random.default_rng(42)
        best_positions = _generate_mackay_icosahedron(n, rng)
        try:
            best_energy, best_positions = relax(best_positions, potential)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
