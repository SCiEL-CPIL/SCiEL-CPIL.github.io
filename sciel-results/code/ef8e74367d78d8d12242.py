# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard: must return within ~280s to be safe (evaluator kills at 330s)
# The evaluator calls construct() 6 times. We need to manage time across calls or per call?
# "called once for each of six instances... all six calls together must finish within 300 s"
# So we have 300s total.
# We need a global timer.

TOTAL_DEADLINE = 290.0
_START_TIME = time.time()

def _time_left():
    return max(0.0, TOTAL_DEADLINE - (time.time() - _START_TIME))

def _morse_energy_fast(positions, rho):
    """Vectorized Morse energy."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    # Avoid overflow for very small distances if rho is large? 
    # exp(x) where x can be large positive if r < 1.
    # rho=10, r=0.5 -> x=5, exp(5)=148. OK.
    # rho=10, r=0.1 -> x=9, exp(9)=8103. OK.
    # If r=0, x=rho, exp(rho). 
    # If r is very small, energy is huge.
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return np.sum(e[iu])

def _enforce_min_distance(positions, min_dist=0.5):
    """Nudge atoms apart if they are too close."""
    n = len(positions)
    positions = positions.copy()
    for _ in range(10): # Iterate a few times to resolve overlaps
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    if d < 1e-10:
                        diff = np.array([0.01, 0, 0])
                    else:
                        diff = (positions[i] - positions[j]) / d
                    shift = (min_dist - d + 0.01) / 2.0
                    positions[i] += diff * shift
                    positions[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return positions

def _make_fcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * np.sqrt(2)
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * a
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_bcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * (2.0 / np.sqrt(3.0)) # Nearest neighbor distance a*sqrt(3)/2 = spacing -> a = 2*spacing/sqrt(3)
    basis = np.array([[0,0,0], [0.5,0.5,0.5]]) * a
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_hcp_cluster(n, spacing=1.0):
    a = spacing
    c = a * np.sqrt(8/3)
    points = []
    radius = int(np.ceil(np.sqrt(n / 3))) + 2
    for layer in range(2 * radius + 1):
        z = (layer - radius) * c
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                x = a * (q + r/2)
                y = a * (np.sqrt(3)/2) * r
                if layer % 2 == 0:
                    points.append([x, y, z])
                else:
                    points.append([x + a/2, y + a*np.sqrt(3)/6, z])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_random_compact(n, rng, spacing=1.0):
    radius = spacing * n ** (1/3) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.8 * spacing:
                points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _make_spherical_shell(n, spacing=1.0):
    points = []
    if n == 1:
        return np.array([[0,0,0]])
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        phi = 2 * np.pi * i / golden
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    points = np.array(points)
    # Scale to have typical NN ~ 1.0
    # Approx radius R ~ (3n/4pi)^(1/3) for close packing volume?
    # Simpler: adjust scale so min dist is approx 1.0
    # For now, use a heuristic scale
    scale = 1.0 * (n / 100.0) ** 0.33 
    # Better: estimate radius from n
    # Volume ~ n * (4/3 pi (0.5)^3) ? No.
    # Just use a fixed scale factor that works for mid-range N and let relaxation fix it.
    scale = 1.5 * (n ** (1/3)) / 3.0 
    points *= scale
    return points

def _relax_config(pos, rho):
    """Relax a single configuration using the provided relax function."""
    try:
        # The relax function signature: relax(positions, 'morse', rho)
        # It returns (energy, positions)
        energy, relaxed_pos = relax(pos.copy(), 'morse', rho)
        # Ensure min distance
        relaxed_pos = _enforce_min_distance(relaxed_pos, 0.5)
        # Recompute energy to be sure
        final_energy = _morse_energy_fast(relaxed_pos, rho)
        return final_energy, relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    """
    Find low-energy Morse cluster configurations using parallel multi-start relaxation.
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    
    initial_configs = []
    
    # 1. Crystalline templates
    initial_configs.append(_make_fcc_cluster(n))
    initial_configs.append(_make_bcc_cluster(n))
    initial_configs.append(_make_hcp_cluster(n))
    
    # 2. Random compact
    initial_configs.append(_make_random_compact(n, rng))
    
    # 3. Spherical shell
    initial_configs.append(_make_spherical_shell(n))
    
    # 4. Jittered versions of crystalline templates
    for i in range(3):
        base = initial_configs[i]
        jitter = rng.normal(0, 0.2, size=base.shape)
        initial_configs.append(base + jitter)
    
    # 5. More random starts
    for i in range(3):
        initial_configs.append(_make_random_compact(n, rng))
    
    # Shuffle to avoid bias
    rng.shuffle(initial_configs)
    
    # Determine how many configs we can process in the time budget
    # Assume ~2-5 seconds per relax for N=80, ~0.5s for N=38.
    # We have ~50s per instance on average (300s / 6).
    # Let's try to process all generated configs (12) in parallel.
    
    time_limit = _time_left()
    if time_limit < 10:
        # Emergency: just return the best initial config
        for pos in initial_configs:
            e = _morse_energy_fast(pos, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        return _enforce_min_distance(best_positions, 0.5)
    
    # Use ThreadPoolExecutor
    # max_workers=4 as per constraint
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all relax tasks
        future_to_pos = {executor.submit(_relax_config, pos, rho): pos for pos in initial_configs}
        
        # As results complete, update best
        for future in concurrent.futures.as_completed(future_to_pos):
            if _time_left() < 5.0:
                # If we are close to deadline, stop waiting for others?
                # We can't easily cancel, but we can break the loop if we have a good result.
                # However, we need to wait for at least one to finish to get a result.
                pass
            
            try:
                energy, pos = future.result(timeout=1.0) # Timeout to avoid hanging on one bad config
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                continue

    # If we didn't get any valid result, fallback
    if best_positions is None:
        best_positions = initial_configs[0]
        best_energy = _morse_energy_fast(best_positions, rho)
        
    # Final check and enforce min distance
    best_positions = _enforce_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
