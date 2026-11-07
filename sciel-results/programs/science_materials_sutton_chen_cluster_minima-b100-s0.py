# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _bcc_start(n, rng, noise_amp=0.1):
    """
    Generate a starting configuration by packing n spheres into a BCC lattice.
    BCC is a good approximation for the local structure of many transition metal clusters.
    """
    # Estimate the required box size. 
    # Density of BCC is 2 atoms per (a^3) unit cell. 
    # We want roughly n atoms.
    # Let side = s. Volume = s^3. 
    # Number of unit cells ~ n/2. s ~ (n/2)^(1/3).
    # We add a margin to ensure we don't run out of space.
    side = (n / 2.0) ** (1.0 / 3.0) * 1.2
    
    # Generate BCC points in a box of size 'side'
    # BCC points: corners (0,0,0) etc and center (0.5, 0.5, 0.5)
    points = []
    # We need to generate enough points to pick n of them.
    # Let's generate points in a grid of unit cells.
    n_cells = int(np.ceil(side))
    
    # To make it more isotropic, we can generate points in a sphere or just a box and filter.
    # Let's generate points in a box [-side/2, side/2]^3
    half_side = side / 2.0
    
    # Generate lattice points
    # Step 1: Corners of unit cells
    # Step 2: Centers of unit cells
    
    # We'll iterate over a grid of cells
    # To ensure we get 'n' points, we might need to generate more than 'n' and then select.
    # A simple way: generate points in a slightly larger box, then randomly select 'n' of them 
    # that are closest to the center to form a compact cluster.
    
    max_pts = int(n * 1.5) + 100
    generated = []
    
    # Iterate over a range of integer coordinates for the unit cells
    # We want to cover the box [-half_side, half_side]
    # Number of cells along each axis:
    n_cells_axis = int(np.ceil(side))
    
    for i in range(-n_cells_axis//2 - 1, n_cells_axis//2 + 2):
        for j in range(-n_cells_axis//2 - 1, n_cells_axis//2 + 2):
            for k in range(-n_cells_axis//2 - 1, n_cells_axis//2 + 2):
                # Corner
                p_corner = np.array([i, j, k])
                # Center
                p_center = np.array([i + 0.5, j + 0.5, k + 0.5])
                
                # Check if within bounds (roughly)
                # We will filter later based on distance from origin to ensure compactness
                generated.append(p_corner)
                generated.append(p_center)
                
                if len(generated) > max_pts * 2:
                    break
            if len(generated) > max_pts * 2:
                break
        if len(generated) > max_pts * 2:
            break
            
    if not generated:
        # Fallback to random if something went wrong
        return rng.uniform(-2, 2, size=(n, 3))
        
    generated = np.array(generated)
    
    # Center the cluster
    generated -= generated.mean(axis=0)
    
    # Select 'n' points closest to the origin to form a compact cluster
    dists = np.linalg.norm(generated, axis=1)
    sorted_indices = np.argsort(dists)
    
    # We want exactly n points. 
    # If we have fewer, we'll have to generate more or use random.
    if len(sorted_indices) < n:
         # Fallback
         return rng.uniform(-2, 2, size=(n, 3))
         
    selected = generated[sorted_indices[:n]]
    
    # Add small noise to break symmetry and allow relaxation
    selected += rng.normal(0, noise_amp, size=selected.shape)
    
    # Scale to ensure minimum distance > 0.35 initially? 
    # BCC lattice constant is 1.0 in this generation. 
    # Closest distance in BCC is sqrt(3)/2 * a ~ 0.866. 
    # So 0.35 is safe.
    
    return selected

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    parameter = kind if kind is not None else instance.get("rho")
    
    # Deadline management
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Deterministic seed for reproducibility, but varied by n
    seed_val = 20240613 ^ (n * 101)
    rng = np.random.default_rng(seed_val)
    
    best_e = np.inf
    best_x = None
    
    def try_relax(x, time_budget, tag=''):
        nonlocal best_e, best_x
        if not np.all(np.isfinite(x)):
            return
        # Check min distance
        if n > 1:
             # Quick check: if any distance < 0.35, skip? 
             # Relax might fix it, but if it's too close, it might explode.
             # Let's just try. The evaluator rejects < 0.35.
             pass
             
        try:
            # Stage 1: Coarse relaxation (faster, less precise)
            # We can't easily control relax's internal settings, 
            # but we can do a quick check or just use relax.
            # Since relax is a black box, we'll just call it.
            # To make it "coarse", we could just use it as is.
            
            e, p = relax(np.asarray(x, dtype=float), 'sutton_chen', parameter)
            
            if np.isfinite(e) and np.all(np.isfinite(p)):
                # Check validity
                if n > 1:
                    d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
                    np.fill_diagonal(d, np.inf)
                    if d.min() < 0.35:
                        return
                
                if e < best_e:
                    best_e = e
                    best_x = np.array(p, dtype=float)
                    
        except Exception:
            return

    # Strategy:
    # 1. Try BCC start with low noise.
    # 2. Try BCC start with higher noise.
    # 3. Try random starts if time permits.
    
    # Start 1: BCC, low noise
    x_bcc = _bcc_start(n, rng, noise_amp=0.05)
    try_relax(x_bcc, 30.0, tag='bcc_low')
    
    # Start 2: BCC, medium noise
    if time.time() < deadline - 30.0:
        x_bcc2 = _bcc_start(n, rng, noise_amp=0.15)
        try_relax(x_bcc2, 30.0, tag='bcc_med')
        
    # Start 3: BCC, high noise
    if time.time() < deadline - 30.0:
        x_bcc3 = _bcc_start(n, rng, noise_amp=0.3)
        try_relax(x_bcc3, 30.0, tag='bcc_high')
        
    # Start 4: Random Hard Core
    if time.time() < deadline - 30.0:
        radius = 0.72 * n ** (1.0 / 3.0) * 0.8
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 20000:
            p = rng.uniform(-radius, radius, 3)
            attempts += 1
            if np.linalg.norm(p) > radius:
                continue
            if pts:
                ok = True
                for q in pts:
                    if np.linalg.norm(p - q) < 0.4:
                        ok = False
                        break
                if not ok:
                    continue
            pts.append(p)
        if len(pts) == n:
            try_relax(np.array(pts), 30.0, tag='random')
            
    # Start 5: Perturbation of best
    if best_x is not None and time.time() < deadline - 20.0:
        # Try to escape local minima
        for i in range(5):
            if time.time() > deadline - 10.0:
                break
            amp = 0.1 + 0.05 * i
            cand = best_x + rng.normal(0.0, amp, size=best_x.shape)
            try_relax(cand, 10.0, tag=f'perturb_{i}')
            
    if best_x is None:
        # Last resort: Grid
        x = np.zeros((n, 3))
        per = int(np.ceil(n ** (1.0 / 3.0))) + 2
        for i in range(n):
            xi = i % per
            yi = (i // per) % per
            zi = i // (per * per)
            x[i] = [xi * 0.5 - per * 0.25, yi * 0.5 - per * 0.25, zi * 0.5]
        best_x = x
        try:
            e, p = relax(best_x, 'sutton_chen', parameter)
            if np.isfinite(e):
                best_x = p
        except Exception:
            pass

    return np.asarray(best_x, dtype=float)
# EVOLVE-BLOCK-END
