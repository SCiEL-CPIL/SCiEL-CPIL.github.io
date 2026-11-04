import numpy as np
import time
from scipy.optimize import minimize

N = 12

# Precompute normals for the outer hexagon (flat-top orientation)
# Vertices at angles 0, 60, 120, 180, 240, 300
# Normals to sides are at 30, 90, 150, 210, 270, 330
angles_n = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
normals = np.column_stack((np.cos(angles_n), np.sin(angles_n)))

# Vertices of a unit hexagon (flat-top)
angles_v = np.deg2rad(np.arange(0, 360, 60))
unit_vertices = np.column_stack((np.cos(angles_v), np.sin(angles_v)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Objective function
    def objective(params):
        # params: 3N vector [x0, y0, x1, y1, ..., x11, y11]
        # We fix rotation to 0 for all inner hexagons to simplify.
        # If this yields a good result, we can try optimizing rotation later.
        
        centers = params.reshape((N, 2))
        
        # Calculate L required to contain all hexagons
        # L = (2/sqrt(3)) * max_{i,j,k} (V_{i,j} dot n_k)
        # V_{i,j} = centers[i] + unit_vertices[j] (since rotation is 0)
        
        # Vectorized calculation of max dot product
        # centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
        # Shape: (N, 6, 2)
        verts = centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
        
        # Dot with normals
        # verts: (N, 6, 2), normals: (6, 2)
        # We need dot product of each vertex with each normal.
        # Result shape: (N, 6, 6)
        # dot_k = sum(verts * normals[k])
        # Using einsum: 'ij,kl->ikl' ? No.
        # verts[i, j] dot normals[k]
        # np.einsum('ij,kl->ikl', verts, normals.T) -> (N, 6, 6)
        dots = np.einsum('ij,kl->ikl', verts, normals.T)
        
        max_dot = np.max(dots)
        L = (2.0 / np.sqrt(3.0)) * max_dot
        
        # Calculate overlap penalties
        # For each pair i, m
        penalty = 0.0
        for i in range(N):
            for m in range(i+1, N):
                d = centers[i] - centers[m]
                # d dot normals
                ds = np.dot(d, normals.T) # shape (6,)
                # abs(ds)
                abs_ds = np.abs(ds)
                # We want max_k (abs_ds[k] - sqrt(3))
                # If this max is negative, they are inside the forbidden hexagon (overlap)
                margin = np.max(abs_ds) - np.sqrt(3.0)
                if margin < 0:
                    # Penalty for being inside
                    # Smooth approximation of max(0, -margin)
                    # margin is negative, so -margin is positive
                    # We use a soft penalty
                    penalty += (-margin) ** 2
        
        # Total objective
        # We want to minimize L + lambda * penalty
        # If penalty is 0, we just minimize L.
        # If penalty > 0, we push them apart.
        # We need to balance L and penalty.
        # A large penalty weight ensures non-overlap first.
        lambda_pen = 100.0
        return L + lambda_pen * penalty

    # Initial guess
    # Try a hexagonal lattice arrangement
    # Side length of lattice spacing for tangent hexagons is sqrt(3)
    # Let's create a 3x4 grid or similar
    # Or just random
    best_score = np.inf
    best_params = None
    
    # Try a few initial configurations
    configs = []
    
    # Config 1: 3x4 Grid
    # Spacing x = 1.5, y = sqrt(3)/2 * 2 = sqrt(3) ?
    # Hex grid:
    # Row 0: 0, 1.5, 3.0
    # Row 1: 0.75, 2.25, 3.75 (offset)
    # This is getting complicated. Let's just use a simple grid with spacing 2.0 (safe)
    # and let the optimizer close them.
    
    x0 = np.zeros(N * 2)
    # Place them in a 3x4 block
    for i in range(N):
        row = i // 4
        col = i % 4
        x0[i*2] = col * 2.0
        x0[i*2+1] = row * 1.8
    configs.append(x0)
    
    # Config 2: Hexagonal Cluster (1 center, 6 ring, 5 ring)
    # Center
    x0 = np.zeros(N * 2)
    x0[0] = 0.0
    x0[1] = 0.0
    # Ring 1
    r = 1.75 # slightly more than sqrt(3) ~ 1.732
    for k in range(6):
        ang = np.deg2rad(k * 60)
        idx = k + 1
        x0[idx*2] = r * np.cos(ang)
        x0[idx*2+1] = r * np.sin(ang)
    # Ring 2 (5 hexagons)
    # Place them in gaps or further out
    # Just place them at radius 3.5
    for k in range(5):
        ang = np.deg2rad(k * 72 + 30) # spread them
        idx = k + 7
        x0[idx*2] = 3.5 * np.cos(ang)
        x0[idx*2+1] = 3.5 * np.sin(ang)
    configs.append(x0)
    
    # Config 3: Random
    np.random.seed(42)
    x0 = np.random.uniform(-5, 5, N*2)
    configs.append(x0)
    
    for cfg in configs:
        if time.time() > deadline:
            break
            
        # Run optimizer
        try:
            res = minimize(objective, cfg, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-6})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to initial config 0
        best_params = configs[0]
        best_score = objective(best_params)

    # Extract centers
    centers = best_params.reshape((N, 2))
    
    # Calculate final L
    verts = centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
    dots = np.einsum('ij,kl->ikl', verts, normals.T)
    max_dot = np.max(dots)
    L = (2.0 / np.sqrt(3.0)) * max_dot
    
    # If the penalty was significant, L might be underestimated if overlaps exist.
    # But the optimizer should have resolved overlaps.
    # To be safe, we can slightly inflate L if there's any chance of overlap.
    # Check overlaps again
    has_overlap = False
    for i in range(N):
        for m in range(i+1, N):
            d = centers[i] - centers[m]
            ds = np.dot(d, normals.T)
            abs_ds = np.abs(ds)
            if np.max(abs_ds) < np.sqrt(3.0):
                has_overlap = True
                break
        if has_overlap:
            break
    
    if has_overlap:
        # Increase L slightly to force validity? No, L is determined by containment.
        # If they overlap, the solution is invalid.
        # We should return a valid solution.
        # If optimization failed, use a known valid packing (e.g. large grid)
        # Let's just return a very large L to be safe? No, score is L.
        # Better to return a valid, albeit larger, packing.
        # Use the 3x4 grid with spacing 2.0
        centers = np.array([[(i%4)*2.0, (i//4)*1.8] for i in range(N)])
        verts = centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
        dots = np.einsum('ij,kl->ikl', verts, normals.T)
        max_dot = np.max(dots)
        L = (2.0 / np.sqrt(3.0)) * max_dot

    inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), L, 0.0

