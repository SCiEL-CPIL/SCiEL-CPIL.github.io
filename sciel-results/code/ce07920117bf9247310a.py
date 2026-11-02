import math
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts, dtype=float)

def _normals(beta):
    out = []
    for k in range(6):
        a = beta + math.pi / 6.0 + k * math.pi / 3.0
        out.append((math.cos(a), math.sin(a)))
    return out

def _enclosure_d(pts, cx, cy, beta):
    # max |dot(p-c, n)| over 6 normals
    best = 0.0
    for nx, ny in _normals(beta):
        d = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        m = max(abs(d.max()), abs(d.min()))
        if m > best:
            best = m
    return best

def _overlap_violation(pts, tol=1e-9):
    n = pts.shape[0]
    mind = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = math.hypot(dx, dy)
            if d < mind:
                mind = d
    return max(0.0, 2.0 - mind + tol)

def _generate_polyhexes(n_cells, max_radius=3):
    """
    Generate all connected sets of n_cells on a hex grid within a bounding box.
    Returns a list of lists of (q, r) coordinates.
    """
    # Generate all possible sets of n_cells from a pool
    # To ensure connectivity, we can use a growth algorithm or filter combinations.
    # For n=11, the number of polyhexes is 552.
    # We can generate them by DFS on the grid.
    
    # Grid coordinates
    coords = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            coords.append((q, r))
    
    coord_set = set(coords)
    
    # Directions in axial coordinates
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    polyhexes = set()
    
    # Use a canonical form to avoid duplicates (e.g., min coordinate is (0,0) and lexicographically smallest)
    # Actually, simpler: generate all connected sets containing (0,0) and then translate/rotate if needed?
    # No, we just need unique shapes. 
    # Let's do a simple DFS that builds connected sets.
    
    def dfs(current_set, current_min_q, current_min_r):
        if len(current_set) == n_cells:
            # Normalize to remove translation invariance? 
            # The problem allows translation of the whole cluster. 
            # The enclosing hexagon calculation is translation invariant if we center it.
            # So we just need unique *shapes*.
            # We can store the tuple of sorted coordinates relative to the first cell?
            # To keep it simple, we store the absolute coordinates but we will center them later.
            # However, different translations of the same shape will be generated.
            # To avoid explosion, we restrict the "first" cell to be (0,0) and ensure no cell has q<0 or (q==0 and r<0) ? 
            # This is tricky for hex grids.
            
            # Let's just collect them and deduplicate by canonical form.
            # Canonical form: translate so min(q) is 0, then min(r) is 0? 
            # Hex grid axial:
            # min_q = min(q for q,r in current_set)
            # min_r = min(r for q,r in current_set if q == min_q) ?? No.
            
            # Simpler canonicalization:
            # 1. Translate so that the lexicographically smallest (q, r) is (0,0).
            # 2. Sort the tuple.
            min_q = min(q for q, r in current_set)
            min_r = min(r for q, r in current_set)
            # This doesn't work well for hex because of the skew.
            
            # Let's use a standard canonical form for polyhexes:
            # Translate so min(q) = 0. Among those, translate so min(r) = 0.
            # Actually, just sorting the coordinates after translating the min-q, min-r point to 0,0 is sufficient for deduplication if we are careful.
            
            # Let's try:
            min_q = min(q for q, r in current_set)
            # Find r for min_q? No, just translate by -min_q, -min_r?
            # Let's just store the sorted tuple of (q, r) relative to the first element added (0,0).
            # Since we start at (0,0) and only add neighbors, the set is connected.
            # We can just use the sorted tuple of the set.
            # To avoid generating the same shape shifted, we enforce that (0,0) is in the set and it's the "top-left" in some sense?
            # Easiest: Generate all connected sets containing (0,0). 
            # Then normalize by translating so min(q)=0, min(r)=0 (in axial).
            # Two sets are the same shape if their normalized forms are identical.
            
            norm_set = set()
            min_q = min(q for q, r in current_set)
            min_r = min(r for q, r in current_set)
            for q, r in current_set:
                norm_set.add((q - min_q, r - min_r))
            polyhexes.add(tuple(sorted(norm_set)))
            return

        # Pruning: if we've gone too far out, stop?
        # For n=11, max radius 3 is enough.
        
        for dq, dr in dirs:
            new_q, new_r = 0 + dq, 0 + dr # Wait, we need to add to existing points
            # We need to add a point adjacent to ANY point in current_set.
            # To avoid duplicates, we only add points that are "new" and maintain a canonical order?
            # Standard polyhex generation:
            # Keep a list of "free" sites adjacent to the current polyhex.
            # Add one free site.
            # To avoid duplicates, ensure the new site is "greater" than all existing sites in some order?
            # Or just use a set of generated shapes.
            
            pass
        
    # The DFS above is a bit messy. Let's use a simpler "combinations" filter for small N.
    # Number of cells in 5x5 grid is 25. C(25, 11) is ~4.4M. Too many.
    
    # Let's use the "growth" method properly.
    # Start with {(0,0)}.
    # State: (frozenset of cells)
    # To avoid duplicates, we only add cells that are "larger" than the last added cell? 
    # Or simpler: Just generate all connected sets containing (0,0) and deduplicate by canonical form.
    # The number of connected sets containing (0,0) of size 11 is manageable.
    
    # Let's implement a clean DFS.
    
    def generate():
        stack = [frozenset([(0, 0)])]
        seen_shapes = set()
        
        while stack:
            current = stack.pop()
            if len(current) == n_cells:
                # Canonicalize
                min_q = min(q for q, r in current)
                min_r = min(r for q, r in current)
                norm = tuple(sorted((q - min_q, r - min_r) for q, r in current))
                if norm not in seen_shapes:
                    seen_shapes.add(norm)
                    yield norm
                continue
            
            # Find all neighbors of current
            neighbors = set()
            for q, r in current:
                for dq, dr in dirs:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) not in current:
                        neighbors.add((nq, nr))
            
            # To reduce duplicates, we can impose an ordering on addition.
            # But since we deduplicate at the end, we can just add all.
            # However, this generates the same shape multiple times (e.g. adding A then B vs B then A).
            # For N=11, the branching factor is ~6. 6^11 is huge.
            # We MUST prune.
            
            # Pruning strategy: Only add neighbors that are "new" in a specific order?
            # Actually, the number of polyhexes is small. The number of *paths* is large.
            # We can use a set `visited` for the current path? No, that prevents different orders.
            
            # Better: Use the "canonical growth" where we only add cells that are greater than the max cell in the set?
            # This is hard to define in hex grid.
            
            # Alternative: Just use the known list of polyhexes? No, must generate.
            
            # Let's limit the depth and use a heuristic:
            # Only add neighbors if they don't create a "hole" that can't be filled? No.
            
            # Let's try a different approach: 
            # Enumerate all subsets of a small grid (e.g. 5x5=25 cells) that are connected.
            # C(25, 11) is 4.4M. Checking connectivity for 4.4M sets is slow in Python.
            
            # Let's go back to DFS with a `visited` set for the *shape* (canonical form) to prune?
            # If we reach a shape that we've seen before at a smaller size? No.
            
            # Okay, let's use a simple BFS/DFS that keeps a set of `current_shape_canonical` to avoid re-exploring the same shape via different orders.
            # State: (frozen_set)
            # If we see the same frozen_set again, skip? No, that's just the same node.
            # The issue is that {A,B,C} can be reached from {A,B}+C or {A,C}+B.
            # If we mark {A,B,C} as visited, we don't re-process it.
            # But we need to generate all *shapes*.
            
            # Let's just generate all connected sets containing (0,0) and deduplicate.
            # To prevent explosion, we limit the bounding box of the set to, say, 5x5.
            
            for nb in neighbors:
                new_set = frozenset(list(current) + [nb])
                # Check bounding box
                qs = [q for q, r in new_set]
                rs = [r for q, r in new_set]
                if max(qs) - min(qs) > 6 or max(rs) - min(rs) > 6:
                    continue
                stack.append(new_set)
                
        return seen_shapes

    # The above stack approach might still be slow if it doesn't prune well.
    # Let's use a recursive generator with a `visited` set of canonical forms for intermediate steps?
    # No, canonical forms of subsets are not unique to the final shape.
    
    # Let's try a simpler, robust method:
    # 1. Define a grid of size 7x7 (49 cells).
    # 2. Use a bitmask or list of indices.
    # 3. Use a DFS that builds the set.
    # 4. To avoid duplicates, enforce that the new cell added is the "maximum" in some total order of the cells in the set?
    #    This is the standard way to generate unique connected sets.
    
    # Define total order on coordinates
    # Map (q, r) to index
    # We only add a cell if it is greater than all existing cells? 
    # This generates each connected set exactly once.
    
    # Let's implement this.
    
    grid_coords = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            grid_coords.append((q, r))
            
    coord_to_idx = {c: i for i, c in enumerate(grid_coords)}
    
    # We want sets of size N.
    # We can use a recursive function that takes a `current_set` and `last_added_idx`.
    # It can only add neighbors with index > last_added_idx.
    
    def gen_unique():
        results = []
        
        # Start with (0,0) which is index 0? 
        # Find index of (0,0)
        start_idx = coord_to_idx.get((0,0))
        if start_idx is None:
            return []
            
        def dfs(current_set, last_idx):
            if len(current_set) == n_cells:
                # Canonicalize for storage? 
                # Since we generated it uniquely, we just need to store it.
                # But we need to center it later.
                # Store the list of (q,r)
                results.append(list(current_set))
                return
                
            # Find neighbors of current_set
            # Optimization: Precompute neighbors for each index
            for q, r in current_set:
                for dq, dr in dirs:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) in coord_to_idx:
                        n_idx = coord_to_idx[(nq, nr)]
                        if n_idx > last_idx: # Ensures unique generation
                            # Check if n_idx is already in current_set? 
                            # Since we only add > last_idx, and last_idx is the max, 
                            # and we start with 0, this implies we are adding in increasing index order?
                            # No, "greater than last_idx" means we can only add cells with higher index than the one we just added.
                            # This generates each set exactly once.
                            
                            # Check if neighbor is already in set (shouldn't be if n_idx > last_idx and we only add new ones)
                            if n_idx not in [coord_to_idx[c] for c in current_set]:
                                new_set = list(current_set) + [(nq, nr)]
                                dfs(new_set, n_idx)
                                
        dfs([(0,0)], start_idx)
        return results

    # This might be slow in Python due to list membership checks.
    # Let's optimize using a set for current_set indices.
    
    def gen_unique_fast():
        results = []
        start_idx = coord_to_idx.get((0,0))
        
        def dfs(current_indices, last_idx):
            if len(current_indices) == n_cells:
                # Convert indices back to (q,r)
                res = [grid_coords[i] for i in current_indices]
                results.append(res)
                return
                
            # Find all neighbors of current_indices
            # Precomputed neighbors for each index
            for idx in current_indices:
                for n_idx in neighbors_map[idx]:
                    if n_idx > last_idx:
                        # Add n_idx
                        new_indices = list(current_indices) + [n_idx]
                        # To speed up membership check, we could use a set, but N is small.
                        # However, we are doing this many times.
                        # Let's just check if n_idx is in current_indices (list).
                        # Since n_idx > last_idx and last_idx is max(current_indices), n_idx is NOT in current_indices.
                        # So we can skip the check!
                        
                        dfs(new_indices, n_idx)
                        
        # Precompute neighbors_map
        neighbors_map = {i: [] for i in range(len(grid_coords))}
        for i, (q, r) in enumerate(grid_coords):
            for dq, dr in dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) in coord_to_idx:
                    n_idx = coord_to_idx[(nq, nr)]
                    neighbors_map[i].append(n_idx)
                    
        dfs([start_idx], start_idx)
        return results

    return gen_unique_fast()

def _optimize_candidate(pts, beta):
    # Optimize center and slight rotations/translations? 
    # The task says "each inner hexagon may translate and rotate".
    # My lattice approach assumes they are on the lattice.
    # To improve, I will allow a small global rotation of the entire cluster and optimize the center.
    # I will also allow a small "jitter" to each point to break symmetry?
    # Let's stick to: Optimize (cx, cy) for a given cluster and beta.
    
    c0 = pts.mean(axis=0)
    vec0 = np.concatenate([[c0[0], c0[1]], pts.ravel()])
    
    def obj(vec):
        cx, cy = vec[0], vec[1]
        p = vec[2:].reshape(-1, 2)
        d = _enclosure_d(p, cx, cy, beta)
        S = (d + APOTHEM) / APOTHEM
        ov = _overlap_violation(p)
        return S + 1000.0 * ov
        
    try:
        res = minimize(obj, vec0, method='L-BFGS-B', options={'maxiter': 100})
        if res.success:
            vec_opt = res.x
            cx, cy = vec_opt[0], vec_opt[1]
            p_opt = vec_opt[2:].reshape(-1, 2)
            ov = _overlap_violation(p_opt)
            if ov < 1e-6:
                d = _enclosure_d(p_opt, cx, cy, beta)
                S = (d + APOTHEM) / APOTHEM
                return S, vec_opt
    except Exception:
        pass
    return float('inf'), None

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate polyhexes
    polyhexes = _generate_polyhexes(N, max_radius=4)
    
    if not polyhexes:
        # Fallback
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (1, 2)]
        pts = _axial_to_cartesian(axial)
        c0 = pts.mean(axis=0)
        beta = 0.0
        d = _enclosure_d(pts, c0[0], c0[1], beta)
        S = (d + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(c0[0]), float(c0[1])], float(S), 0.0

    best_S = float('inf')
    best_vec = None
    best_beta = 0.0
    
    # Test a few betas
    beta_grid = [k * (math.pi / 6) / 12 for k in range(12)]
    
    # Sort polyhexes by some heuristic? 
    # Maybe by bounding box area?
    # Let's just iterate.
    
    count = 0
    for ph in polyhexes:
        if not _time_ok():
            break
            
        pts = _axial_to_cartesian(ph)
        
        for beta in beta_grid:
            if not _time_ok():
                break
                
            S, vec = _optimize_candidate(pts, beta)
            
            if S < best_S:
                best_S = S
                best_vec = vec
                best_beta = beta
                
        count += 1
        if count % 100 == 0:
            # Check time
            if not _time_ok():
                break

    if best_vec is None or not (best_S < float('inf')):
        # Fallback to first polyhex
        ph = polyhexes[0]
        pts = _axial_to_cartesian(ph)
        c0 = pts.mean(axis=0)
        beta = 0.0
        d = _enclosure_d(pts, c0[0], c0[1], beta)
        best_S = (d + APOTHEM) / APOTHEM
        best_vec = np.concatenate([[c0[0], c0[1]], pts.ravel()])
        best_beta = 0.0

    cx, cy = float(best_vec[0]), float(best_vec[1])
    pts = best_vec[2:].reshape(-1, 2)
    angle_deg = math.degrees(best_beta)

    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [cx, cy]
    d = _enclosure_d(pts, cx, cy, best_beta)
    side = float((d + APOTHEM) / APOTHEM)
    rotation = float(angle_deg)

    return inner, center, side, rotation

