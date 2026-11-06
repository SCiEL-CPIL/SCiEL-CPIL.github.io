# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

# Global deadline tracking
_START_TIME = None
_DEADLINE = 300.0  # seconds, with safety margin
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    # Only consider pairs i < j
    triu_idx = np.triu_indices(n, k=1)
    r = dist[triu_idx]
    
    E = 0.0
    # Check minimum distance
    if np.min(r) < 0.5:
        return 1e10
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        E += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2 = r[mask2]
        E += np.sum(B * np.exp(d / (r2 - b)))
    
    return float(E)

def _get_mackay_directions(n_atoms):
    """
    Generate unit direction vectors for atoms in a Mackay icosahedron of size n_atoms.
    Returns:
        dirs: list of arrays, where dirs[k] is an array of shape (count_k, 3) containing
              unit vectors for atoms in shell k.
        counts: list of counts for each shell.
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Shell 0: Center
    shells = [np.zeros((1, 3))]
    remaining = n_atoms - 1
    
    if remaining <= 0:
        return shells, [1]
    
    # Shell 1: 12 vertices
    if remaining >= 12:
        shells.append(verts)
        remaining -= 12
    else:
        # If n is not a magic number, we might need to handle partial shells or just stop
        # For this task n is 38-60, which are not magic numbers (13, 55, 147...).
        # 38, 45, 50, 55, 57, 60.
        # 55 is a magic number (1+12+42).
        # For non-magic numbers, we will just take the first 'remaining' atoms from the next shell
        # or distribute them. However, standard Mackay construction for non-magic numbers
        # usually involves adding atoms to the outermost shell or using a more flexible model.
        # Given the "polytetrahedral" hint, maybe we should allow the outer shell to be incomplete
        # or use a different packing.
        # Let's stick to full shells and then add the remainder to the outermost shell
        # by picking random directions or filling gaps.
        # Actually, for 38, 45, 50, 57, 60, we are close to 55.
        # 55 = 1 + 12 + 42.
        # 38 is 1 + 12 + 25 (partial 2nd shell).
        # 45 is 1 + 12 + 32 (partial 2nd shell).
        # 50 is 1 + 12 + 37 (partial 2nd shell).
        # 57 is 1 + 12 + 42 + 2 (partial 3rd shell).
        # 60 is 1 + 12 + 42 + 5 (partial 3rd shell).
        
        # We will generate the full shell directions and then truncate or pad.
        # But first we need the directions for Shell 2 (42 atoms) and Shell 3 (92 atoms).
        
        # Shell 2: 42 atoms (12 face centers + 30 edge midpoints)
        # We need to generate these directions.
        
        # Identify faces and edges of the icosahedron defined by `verts`
        # Distances between vertices
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                dists[i, j] = np.linalg.norm(verts[i] - verts[j])
        
        # Nearest neighbors (edges)
        min_dist = np.min(dists[dists > 0])
        edge_set = set()
        for i in range(12):
            for j in range(12):
                if abs(dists[i, j] - min_dist) < 1e-6 and i < j:
                    edge_set.add((i, j))
        
        # Edge midpoints (30 directions)
        edge_dirs = []
        for (i, j) in edge_set:
            mid = verts[i] + verts[j]
            mid /= np.linalg.norm(mid)
            edge_dirs.append(mid)
        edge_dirs = np.array(edge_dirs)
        
        # Face centers (20 directions)
        # Find triangles (i, j, k) such that all pairs are edges
        face_dirs = []
        for i in range(12):
            for j in range(i+1, 12):
                if (i, j) in edge_set:
                    for k in range(j+1, 12):
                        if (i, k) in edge_set and (j, k) in edge_set:
                            fc = verts[i] + verts[j] + verts[k]
                            fc /= np.linalg.norm(fc)
                            face_dirs.append(fc)
        face_dirs = np.array(face_dirs)
        
        # Combine for Shell 2 (42 atoms)
        # Note: The 42 atoms in a Mackay icosahedron are typically placed at:
        # 12 positions corresponding to the 12 vertices of a dodecahedron (which are the face centers of the icosahedron? No, dodecahedron vertices are face centers of icosahedron).
        # Wait, the duality: Vertices of Dodecahedron correspond to Faces of Icosahedron.
        # So the 20 face centers of the icosahedron correspond to the 20 vertices of the dodecahedron.
        # But the 2nd shell has 42 atoms.
        # In a Mackay icosahedron, the 2nd shell consists of:
        # - 12 atoms at the "face centers" of the icosahedron? No, that would be 20.
        # - Let's check the literature.
        # Mackay icosahedron shells:
        # N=1: 1
        # N=13: 12 (vertices of ico)
        # N=55: 42 (12 face centers of ico? No. 30 edge midpoints + 12 ?)
        # Actually, the 42 points are:
        # 12 points at the centers of the 20 faces? No, 20 faces.
        # The 42 points are the vertices of the icosidodecahedron? No, that's 30.
        # The 42 points are: 12 vertices of a dodecahedron + 30 vertices of an icosahedron?
        # Let's look at the geometry.
        # The 2nd shell of a Mackay icosahedron is often described as having atoms at:
        # - 12 positions corresponding to the 12 vertices of a dodecahedron inscribed in the sphere.
        # - 30 positions corresponding to the 30 vertices of an icosahedron? No.
        
        # Let's use the standard construction:
        # The 42 directions are the union of:
        # 1. The 12 face-center directions of the icosahedron? (There are 20 faces, so 20 directions).
        # 2. The 30 edge-midpoint directions.
        # Total 50? No.
        
        # Correct Mackay 2nd shell (42 atoms):
        # It consists of 12 atoms at the positions of the vertices of a dodecahedron
        # and 30 atoms at the positions of the vertices of an icosahedron rotated?
        
        # Let's just generate 42 directions by taking the 20 face centers and 30 edge midpoints
        # and selecting 42? Or maybe the 42 are the 12 "polar" vertices of the icosahedron
        # plus 30 "equatorial"?
        
        # Actually, a simpler robust way for non-magic numbers:
        # Use the 13-atom icosahedron, then add atoms to fill the space
        # with icosahedral symmetry as much as possible.
        
        # For the purpose of this variational optimization, we can define the "shells"
        # as the 1 center, 12 vertices, and then the rest as a single "outer shell"
        # with icosahedral symmetry broken by the specific count.
        # However, the prompt suggests "shell radii and twist angles".
        # This implies we treat groups of atoms as shells.
        
        # Let's define shells based on the magic numbers.
        # If n <= 13: Shell 0 (1), Shell 1 (12).
        # If 13 < n <= 55: Shell 0 (1), Shell 1 (12), Shell 2 (n-13).
        # If 55 < n <= 147: Shell 0 (1), Shell 1 (12), Shell 2 (42), Shell 3 (n-55).
        
        # For Shell 2 (42 atoms), we need 42 directions.
        # The 42 directions of the Mackay 2nd shell are:
        # 12 directions: Face centers of the icosahedron? No, there are 20.
        # 30 directions: Edge midpoints?
        # Actually, the 42 points are the vertices of the "icosidodecahedron" (30) plus the 12 vertices of the icosahedron? No, those are shell 1.
        
        # Let's look at the coordinates of the 42 points.
        # They are:
        # 12 points at (0, 1, phi) type rotations?
        
        # Let's use the 20 face centers and 30 edge midpoints.
        # If we need 42, we can take all 20 face centers and 22 edge midpoints?
        # Or maybe the 42 points are:
        # 12 points at the "face centers" of the 12 faces of a dodecahedron?
        
        # Let's try a different approach for the directions of the outer shell:
        # Use the 12 icosahedron vertices, 20 face centers, and 30 edge midpoints.
        # Total 62 unique directions.
        # For n atoms, we can pick the first n directions from this set,
        # sorted by some criterion, or just use a subset.
        
        # However, for the variational optimization, it is better to have a fixed set of directions
        # for a given n.
        
        # Let's define the directions for the "outer shell" (shell index >= 1)
        # as the union of icosahedron vertices, face centers, and edge midpoints.
        
        all_dirs = []
        # Shell 1 dirs (12)
        all_dirs.extend([v for v in verts])
        # Face centers (20)
        all_dirs.extend([v for v in face_dirs])
        # Edge midpoints (30)
        all_dirs.extend([v for v in edge_dirs])
        
        # Total 62 directions.
        # We need to assign these to atoms.
        # For n=38: 1 center, 12 shell 1, 25 shell 2.
        # We can take the first 25 from all_dirs?
        # This breaks symmetry.
        
        # Better: Assign atoms to shells based on the magic numbers.
        # Shell 1: 12 atoms (verts).
        # Shell 2: Up to 42 atoms.
        # If n-13 <= 42, we use a subset of the 42 Mackay directions.
        # Which 42?
        # The standard 42 are: 12 face centers of the icosahedron? No.
        # Let's assume the 42 directions are the 20 face centers + 30 edge midpoints,
        # but we only need 42.
        # Actually, the Mackay 55-atom cluster has 42 atoms in the 2nd shell.
        # These 42 atoms are located at:
        # - 12 positions: centers of the 12 faces of a dodecahedron?
        # - 30 positions: centers of the 30 faces of an icosahedron?
        
        # Let's just use the 20 face centers and 30 edge midpoints.
        # If we need 42, we can use all 20 face centers and 22 edge midpoints.
        # Or we can use the 30 edge midpoints and 12 face centers (chosen symmetrically).
        
        # For simplicity and robustness in the variational step,
        # we will define the "outer shell" directions as the 62 unique directions
        # (12 verts + 20 faces + 30 edges).
        # We will assign the first M directions to the outer shell atoms.
        # To maintain symmetry as much as possible, we should group them.
        
        # Let's return the full list of 62 directions and let the caller pick.
        
        return verts, face_dirs, edge_dirs, remaining

    # This logic is getting complicated. Let's simplify.
    # We will use the 13-atom icosahedron as the core.
    # The remaining atoms will be placed in an "outer shell" with directions
    # chosen from the set of icosahedral symmetry directions.
    
    # For the variational optimization, we will parameterize:
    # - Radius of Shell 1 (12 atoms)
    # - Radius of Shell 2 (remaining atoms)
    # - Twist of Shell 1
    # - Twist of Shell 2
    
    # The directions for Shell 2 will be fixed to the first (n-13) from the
    # combined list of face centers and edge midpoints, to ensure they are
    # distinct from Shell 1 vertices.
    
    # Let's restructure _get_mackay_directions to return:
    # - core_dirs: 12 vectors
    # - outer_dirs: list of vectors for the remaining atoms
    
    # We already have verts (12), face_dirs (20), edge_dirs (30).
    # Let's combine face_dirs and edge_dirs to get 50 directions.
    # If we need more than 50, we can add more shells or just repeat?
    # For n=60, we need 47 outer atoms. 50 is enough.
    
    outer_candidates = np.vstack([face_dirs, edge_dirs])
    
    # We need `remaining` directions from outer_candidates.
    # To be symmetric, we should probably pick them in a way that preserves
    # some symmetry, but for now, just take the first `remaining`.
    # Note: This might break symmetry, but the variational optimization
    # of radius and twist might help.
    
    outer_dirs = outer_candidates[:remaining]
    
    shells = [np.zeros((1, 3)), verts, outer_dirs]
    counts = [1, 12, remaining]
    
    return shells, counts

def _build_structure(n, params):
    """
    Build Cartesian coordinates from parameters.
    params: [r1, theta1, r2, theta2, ...]
    r1: radius of shell 1
    theta1: twist of shell 1 around z-axis
    r2: radius of shell 2
    theta2: twist of shell 2 around z-axis
    """
    shells, counts = _get_mackay_directions(n)
    
    pos = []
    for k, (dirs, count) in enumerate(zip(shells, counts)):
        if count == 0:
            continue
        
        # Extract parameters for this shell
        # Assume params are ordered: r0, theta0, r1, theta1, ...
        # Shell 0 is center, no params needed (r=0, theta=0)
        if k == 0:
            pos.append(np.zeros(3))
            continue
        
        # For shell k, parameters are at index 2*(k-1) and 2*(k-1)+1
        p_idx = 2 * (k - 1)
        r_k = params[p_idx]
        theta_k = params[p_idx + 1]
        
        # Rotate directions by theta_k around z-axis
        cos_t = np.cos(theta_k)
        sin_t = np.sin(theta_k)
        rot = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
        
        rotated_dirs = dirs @ rot.T
        
        # Scale by radius
        shell_pos = r_k * rotated_dirs
        
        pos.append(shell_pos)
        
    pos = np.vstack(pos)
    return pos[:n]

def _energy_and_grad_params(params, n):
    """
    Compute energy and gradient with respect to parameters.
    """
    pos = _build_structure(n, params)
    
    # Check for invalid positions
    if not np.all(np.isfinite(pos)):
        return 1e10, np.zeros_like(params)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    if np.min(dist) < 0.5:
        return 1e10, np.zeros_like(params)
    
    # Compute Cartesian gradient
    try:
        e, grad_cart = dzugutov_energy_gradient(pos)
    except:
        return 1e10, np.zeros_like(params)
    
    if not np.isfinite(e):
        return 1e10, np.zeros_like(params)
    
    # Compute parameter gradient using chain rule
    # dE/dp_j = sum_i grad_cart_i . dpos_i/dp_j
    
    shells, counts = _get_mackay_directions(n)
    
    grad_params = np.zeros_like(params)
    
    for k, (dirs, count) in enumerate(zip(shells, counts)):
        if count == 0:
            continue
        if k == 0:
            continue
            
        p_idx = 2 * (k - 1)
        r_k = params[p_idx]
        theta_k = params[p_idx + 1]
        
        # Get the indices of atoms in this shell
        # We need to know which atoms correspond to this shell in the final pos array
        # Since we built pos by vstacking, we can reconstruct the indices
        # But it's easier to just re-compute the derivatives analytically for the shell
        
        # dpos_i/dr_k = dirs_i (unit vector)
        # dpos_i/dtheta_k = r_k * d(dirs_i)/dtheta
        # d(dirs_i)/dtheta = rot_z'(theta) dirs_i
        # rot_z'(theta) = [[-sin, -cos, 0], [cos, -sin, 0], [0, 0, 0]]
        
        cos_t = np.cos(theta_k)
        sin_t = np.sin(theta_k)
        
        # Gradient of position w.r.t radius
        dpos_dr = dirs
        
        # Gradient of position w.r.t theta
        # rot_z'(theta) = [[-sin, -cos, 0], [cos, -sin, 0], [0, 0, 0]]
        # d(dirs)/dtheta = [[-sin, -cos, 0], [cos, -sin, 0], [0, 0, 0]] @ dirs
        # Let dirs = [x, y, z]
        # d(x)/dtheta = -sin*x - cos*y
        # d(y)/dtheta = cos*x - sin*y
        # d(z)/dtheta = 0
        
        ddirs_dtheta = np.zeros_like(dirs)
        ddirs_dtheta[:, 0] = -sin_t * dirs[:, 0] - cos_t * dirs[:, 1]
        ddirs_dtheta[:, 1] = cos_t * dirs[:, 0] - sin_t * dirs[:, 1]
        ddirs_dtheta[:, 2] = 0
        
        dpos_dtheta = r_k * ddirs_dtheta
        
        # Sum gradients
        # We need the Cartesian gradient for the atoms in this shell
        # Since we don't track indices explicitly in _build_structure,
        # let's modify _build_structure to return indices or just re-compute.
        # For now, let's assume the order in pos matches the vstack order.
        
        # Re-constructing indices:
        # Shell 0: 0
        # Shell 1: 1 to 12
        # Shell 2: 13 to 12+count
        
        start_idx = 1
        for j in range(1, k):
            start_idx += counts[j]
            
        end_idx = start_idx + count
        
        grad_shell = grad_cart[start_idx:end_idx]
        
        # dE/dr_k = sum(grad_shell . dpos_dr)
        grad_params[p_idx] = np.sum(grad_shell * dpos_dr)
        
        # dE/dtheta_k = sum(grad_shell . dpos_dtheta)
        grad_params[p_idx + 1] = np.sum(grad_shell * dpos_dtheta)
        
    return e, grad_params

def _optimize_params(n, initial_params):
    """
    Optimize the parameters using L-BFGS-B.
    """
    result = minimize(
        _energy_and_grad_params,
        initial_params,
        args=(n,),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return result.x, result.fun

def construct(instance):
    """Find low-energy Dzugutov cluster structure."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    
    # Initial parameters:
    # r1 ~ 1.0, theta1 ~ 0
    # r2 ~ 1.9, theta2 ~ 0
    # r3 ~ 3.0, theta3 ~ 0
    
    # Determine number of shells
    # Shell 0: 1
    # Shell 1: 12
    # Shell 2: up to 42
    # Shell 3: up to 92
    
    shells, counts = _get_mackay_directions(n)
    num_shells = len(shells)
    
    # Initial guess for radii
    # Shell 1: 1.0
    # Shell 2: 1.9
    # Shell 3: 3.0
    initial_r = [1.0, 1.9, 3.0, 4.0]
    
    initial_params = []
    for k in range(1, num_shells):
        initial_params.append(initial_r[k-1])
        initial_params.append(0.0)
    
    # Run optimization
    try:
        opt_params, opt_energy = _optimize_params(n, initial_params)
    except Exception as e:
        # Fallback to relaxed icosahedron
        opt_params = initial_params
        opt_energy = 1e10
    
    # Build final structure
    pos = _build_structure(n, opt_params)
    
    # Relax the Cartesian structure to find the true local minimum
    try:
        e, pos = relax(pos, 'dzugutov', None)
    except:
        pass
    
    # Verify minimum distance
    if pos is not None and np.all(np.isfinite(pos)):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        if min_dist < 0.5:
            # Nudge atoms apart
            for _ in range(100):
                if min_dist >= 0.5:
                    break
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                if i > j:
                    i, j = j, i
                dir_vec = pos[i] - pos[j]
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-10:
                    dir_vec = np.array([1.0, 0, 0])
                else:
                    dir_vec = dir_vec / norm
                pos[i] += 0.01 * dir_vec
                pos[j] -= 0.01 * dir_vec
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
            
            try:
                _, pos = relax(pos, 'dzugutov', None)
            except:
                pass
    
    return pos
# EVOLVE-BLOCK-END
