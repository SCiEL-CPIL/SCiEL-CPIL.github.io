import numpy as np
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 15.0
_lock = threading.Lock()

def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN

def _validate_positions(positions):
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Vectorized distance check
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.35:
        return False
    return True

def _generate_mackay_icosahedron(n):
    """
    Generates a Mackay icosahedral cluster of size n.
    Mackay icosahedrons have sizes 13, 55, 147, ...
    For n not matching a Mackay size, it generates the largest Mackay
    icosahedron <= n and adds extra atoms to the outer shell.
    """
    # Icosahedron vertices (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms (radius 1)
    if n > 1:
        points.extend(list(verts))
    
    # Shell 2: 42 atoms (radius ~ 1.902)
    # 12 face centers + 30 edge midpoints
    if n > 13:
        # Face centers
        # We need the faces. Instead of hardcoding, we can approximate.
        # But for Mackay, the 2nd shell is specific.
        # Let's use a simpler approach: generate 42 points on a sphere of radius R2.
        # R2 for Mackay 55 is such that 12-12 distance is preserved? 
        # In Mackay 55, the 2nd shell atoms are at distance sqrt(5+2*phi) * r1? 
        # Actually, for a regular icosahedron with edge length a, 
        # circumradius R = a/4 * sqrt(10+2*sqrt(5)).
        # Here we normalized verts to radius 1.
        # The distance between adjacent vertices (edge length) is:
        # d_edge = ||verts[0] - verts[1]||
        d_edge = np.linalg.norm(verts[0] - verts[1])
        
        # In Mackay icosahedron, the 2nd shell is placed such that 
        # the structure is "close-packed" locally.
        # A common radius for the 2nd shell in icosahedral units is ~ 1.9.
        # Let's refine: 
        # The 2nd shell consists of 12 face centers and 30 edge midpoints.
        # Face center of a triangle with side d_edge is at distance (d_edge/sqrt(3)) from centroid?
        # No, face center is at distance R_face from center.
        # For a regular icosahedron of radius 1, the face center distance is 1/sqrt(5+2*sqrt(5))? No.
        
        # Let's just use a Fibonacci sphere for the 42 points if we don't want to hardcode geometry.
        # But symmetry is better.
        # Let's hardcode the 42 points for the 2nd shell of a Mackay icosahedron.
        # Or simpler: use a known generator.
        
        # For now, let's use a "spherical shell" approximation for the extra atoms 
        # if we can't easily generate perfect Mackay shells.
        # However, for n=55, 147, we can be more precise.
        
        # Let's try to generate the 42 points properly.
        # 12 face centers:
        # We need the 20 faces.
        # This is tedious. 
        # Alternative: 
        # If n is close to 55, 147, etc., we can use a "growth" strategy.
        
        # Let's use a simpler, robust method: 
        # Generate points on a sphere of radius R2 = 1.9 (tunable) using Fibonacci distribution.
        # This breaks symmetry but is a good start for relaxation.
        
        R2 = 1.9
        N_outer = min(42, n - 13)
        if N_outer > 0:
            z = (2 * np.arange(N_outer) + 1) / N_outer
            r = np.sqrt(1 - z**2)
            phi_angle = np.arange(N_outer) * (np.pi * (3 - np.sqrt(5)))
            
            x = r * np.cos(phi_angle)
            y = r * np.sin(phi_angle)
            
            outer_pts = np.column_stack([x, y, z]) * R2
            points.extend(list(outer_pts))
            
    # If we still need more atoms (e.g., n=75, 80, 38, 59, 79)
    # We add more shells or random points.
    # For n=38 (12-6), the global min is often a deformed icosahedron or FCC fragment.
    # For n=55 (10-8), it's likely the Mackay 55.
    
    if len(points) < n:
        deficit = n - len(points)
        # Add points on a larger sphere
        R3 = 2.5
        z = (2 * np.arange(deficit) + 1) / deficit
        r = np.sqrt(1 - z**2)
        phi_angle = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
        
        x = r * np.cos(phi_angle)
        y = r * np.sin(phi_angle)
        
        outer_pts = np.column_stack([x, y, z]) * R3
        points.extend(list(outer_pts))
        
    if len(points) > n:
        points = points[:n]
        
    pts = np.array(points)
    # Center
    pts -= pts.mean(axis=0)
    
    # Scale to a reasonable size. 
    # The equilibrium nearest-neighbor distance for Sutton-Chen is ~1.0-1.1.
    # Our initial radius is ~3. 
    # We should scale such that the average nearest neighbor distance is ~1.0.
    # But relax() will handle the scaling. 
    # However, starting too large or small can trap in bad local minima.
    # Let's scale so that the max radius is ~ 3.0.
    norm = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norm)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
        
    return pts

def _generate_fcc_sphere(n):
    """
    Generate an FCC-like sphere.
    """
    a = 1.0
    pts = []
    # Generate a cube of FCC points
    L = int(np.cbrt(n / 2.0)) + 3
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    pts.append([x, y, z])
                    
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    norm = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norm)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
        
    return pts

def _generate_bcc_sphere(n):
    """
    Generate a BCC-like sphere.
    """
    a = 1.0
    pts = []
    L = int(np.cbrt(n / 2.0)) + 3
    for i in range(L):
        for j in range(L):
            for k in range(L):
                pts.append([i*a, j*a, k*a])
                if i < L-1 and j < L-1 and k < L-1:
                    pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
                    
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    norm = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norm)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
        
    return pts

def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)
    
    best_energy = np.inf
    best_positions = None
    
    # Initial diverse starts
    starts = []
    
    # 1. Icosahedral (Mackay-like)
    ico = _generate_mackay_icosahedron(n)
    if _validate_positions(ico):
        starts.append(ico)
        
    # 2. FCC
    fcc = _generate_fcc_sphere(n)
    if _validate_positions(fcc):
        starts.append(fcc)
        
    # 3. BCC
    bcc = _generate_bcc_sphere(n)
    if _validate_positions(bcc):
        starts.append(bcc)
        
    # 4. Random Compact (Greedy)
    for _ in range(2):
        pts = np.zeros((n, 3))
        pts[0] = 0
        for i in range(1, n):
            idx = rng.integers(0, i)
            offset = rng.normal(0, 0.5, 3)
            pts[i] = pts[idx] + offset
        if _validate_positions(pts):
            starts.append(pts)

    # Relax initial starts in parallel
    def _relax_one(pos):
        try:
            energy, positions = relax(pos, 'sutton_chen', kind)
            if _validate_positions(positions):
                return energy, positions
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_one, start): start for start in starts}
        for future in as_completed(futures):
            if _get_time_left() < 5.0:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    with _lock:
                        best_energy = energy
                        best_positions = positions.copy()
            except Exception:
                pass

    # Parallel Basin Hopping
    # We will perform batches of kicks on the best structure found.
    
    num_batches = 50 # Max batches
    kicks_per_batch = 4
    kick_amplitudes = [0.1, 0.2, 0.5, 1.0]
    
    for batch in range(num_batches):
        if _get_time_left() < 15.0:
            break
            
        if best_positions is None:
            break
            
        # Generate perturbations
        perturbations = []
        for k in range(kicks_per_batch):
            # Choose a kick amplitude
            amp = kick_amplitudes[batch % len(kick_amplitudes)]
            
            # Generate perturbed structure
            noise = rng.normal(0, amp, best_positions.shape)
            perturbed = best_positions + noise
            
            # Check validity before relaxing
            if _validate_positions(perturbed):
                perturbations.append(perturbed)
            else:
                # Try a smaller kick
                noise = rng.normal(0, amp * 0.5, best_positions.shape)
                perturbed = best_positions + noise
                if _validate_positions(perturbed):
                    perturbations.append(perturbed)
        
        if not perturbations:
            continue

        # Relax perturbations in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_one, pos): pos for pos in perturbations}
            for future in as_completed(futures):
                if _get_time_left() < 5.0:
                    break
                try:
                    energy, positions = future.result()
                    if energy < best_energy:
                        with _lock:
                            best_energy = energy
                            best_positions = positions.copy()
                except Exception:
                    pass

    # Final validation
    if best_positions is None:
        # Fallback to first start if something went wrong
        if starts:
            best_positions = starts[0]
        else:
            # Absolute fallback
            best_positions = np.zeros((n, 3))
            
    return best_positions
