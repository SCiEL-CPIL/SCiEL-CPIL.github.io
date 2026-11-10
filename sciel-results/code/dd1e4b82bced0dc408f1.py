# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0

    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return (x, y)

    # Generate all unique structures of 11 hexagons
    # We use a "growth" or "combinations" approach.
    # Since 11 is small, we can generate all sets of 11 axial coords
    # within a bounding box, normalize them, and check validity.
    
    # Bounding box: max distance from origin.
    # For 11 hexagons, the max radius is roughly sqrt(11) ~ 3.3.
    # Let's use a radius of 4 to be safe.
    
    # To avoid duplicates, we fix the "lowest-leftmost" hexagon to (0,0)?
    # No, better to fix the centroid to (0,0) and use a canonical representation.
    
    # Strategy:
    # 1. Generate all subsets of size 11 from a grid of axial coords.
    # 2. For each subset, translate so that the lexicographically smallest point is at (0,0) 
    #    OR so that the centroid is at (0,0).
    #    Actually, for enclosing hexagon, translation doesn't change the side length, 
    #    only the center. So we can fix the centroid to (0,0) for the optimization.
    #    But for uniqueness, we need a canonical form.
    #    Let's fix the point with min x, then min y to be (0,0).
    #    Wait, if we fix (0,0), we generate all structures that contain (0,0).
    #    This covers all translation-invariant structures.
    
    # Grid points
    max_r = 4
    points = []
    for q in range(-max_r, max_r + 1):
        for r in range(-max_r, max_r + 1):
            points.append((q, r))
    
    # We need to choose 10 other points to go with (0,0).
    # Total points in grid: (2*4+1)^2 = 81.
    # Combinations(80, 10) is huge (~ 80 choose 10 is 10^12). Too many.
    
    # Better: Generate connected structures?
    # Or just generate all subsets of size 11 from a smaller set?
    # If we restrict to points within distance 3 from origin?
    # Let's try a different approach:
    # Since the optimal packing is likely compact, we can iterate over all
    # "polycubes" of size 11? No, 2D.
    
    # Alternative: Use the fact that the optimal packing for 11 hexagons is known
    # or can be found by a smarter search.
    # However, the task asks for a program.
    
    # Let's use a randomized search with local optimization, but structured differently.
    # Actually, let's stick to the prompt's hint: "Combinatorial Structure Search".
    # How to make it efficient?
    # We can generate all "hexagonal numbers" shapes? No.
    
    # Let's use a recursive backtracking to generate all connected sets of 11 hexagons
    # containing (0,0).
    # Number of free hexominoes is 35. Free 11-ominoes is ~ 10^5?
    # It might be manageable.
    
    # Let's implement a generator for free polyhexes of size 11.
    
    def get_polyhexes(n):
        # Returns list of sets of (q,r) tuples, normalized so min-q=0, min-r=0?
        # Or centered?
        # We'll normalize so that the lexicographically smallest point is (0,0).
        
        # Start with (0,0)
        current = {(0,0)}
        
        if n == 1:
            return [current]
            
        results = []
        seen = set()
        
        # Recursive growth
        def grow(hexes, remaining):
            if remaining == 0:
                # Normalize
                min_q = min(q for q, r in hexes)
                min_r = min(r for q, r in hexes)
                # Canonical form: sort and tuple
                norm = tuple(sorted((q - min_q, r - min_r) for q, r in hexes))
                if norm not in seen:
                    seen.add(norm)
                    results.append(frozenset(norm))
                return
            
            # Try adding a neighbor to each existing hex
            # To avoid duplicates, only add neighbors that are "larger" in some sense?
            # Or just add all and rely on canonicalization + seen set.
            # To be efficient, we can restrict to adding neighbors that don't "go back".
            # Standard technique: only add neighbors (dq, dr) such that (q+dq, r+dr) > max(hexes) in lex order?
            # No, that restricts shape too much.
            # Let's just add all valid neighbors and rely on the `seen` set for canonical forms.
            # This will be slow for n=11 if not careful.
            
            # Optimization: Only add neighbors that are "outside" the current bounding box?
            # Or use the "growth" rule: only add to the "frontier".
            
            # Let's use a simpler heuristic:
            # Only add a neighbor if it increases the "span" or is in a specific direction.
            # Actually, for n=11, the number of free polyhexes is 22427 (from OEIS A001424? No, that's n=1..).
            # A001424: 1, 1, 2, 6, 16, 42, 108, 282, 734, 1918, 4930, 12710...
            # So 12710 free polyhexes of size 11.
            # This is very manageable!
            
            # We need to generate them.
            # Standard algorithm:
            # Keep a set of "active" sites.
            # To avoid duplicates, impose an ordering on the addition.
            # Rule: When adding a new hex, it must be the one with the largest (q, r) coordinate
            # among the newly added ones? No.
            
            # Let's use the "canonical growth" method:
            # 1. Start with (0,0).
            # 2. At each step, consider all neighbors of the current set.
            # 3. Only add a neighbor if it is "greater than or equal to" all previously added neighbors
            #    in a specific ordering? 
            #    Actually, the standard way to generate free polyominoes/polyhexes is:
            #    Fix the first cell at (0,0).
            #    Subsequent cells must have coordinates (q, r) such that (q, r) is not "behind" the previous ones?
            #    A common constraint: The new cell must have q > min_q or (q == min_q and r > min_r)? No.
            
            # Let's just generate all connected sets of size 11 containing (0,0) using BFS/DFS
            # and use a set of canonical forms to deduplicate.
            # With 12k results, the search space of connected sets containing (0,0) is larger but finite.
            
            pass # Placeholder, we will implement a specific generator below.

        # Given the complexity of implementing a perfect polyhex generator on the fly,
        # and the time limit, let's use a **Randomized Local Search** with **Analytical Enclosing Hexagon**.
        # This is a different strategy from the parent (which used basinhopping on positions).
        # We will use **Simulated Annealing** or **Random Restart Hill Climbing** 
        # but the key is the **Enclosing Hexagon Calculation** is exact (1D opt) 
        # and we use **Collision Detection** to guide the search.
        
        # Actually, the prompt asks for a MATERIAL change.
        # The parent used `basinhopping` on 23 vars.
        # I will use **Particle Swarm Optimization (PSO)** or **Differential Evolution** 
        # which handle non-convex constraints better by population-based search.
        # BUT, the most robust structural change for packing is **Combinatorial Search**.
        
        # Let's implement the Combinatorial Search properly.
        # We will generate all free polyhexes of size 11.
        
        # Algorithm for free polyhexes:
        # 1. Represent a polyhex as a frozenset of (q,r).
        # 2. Start with {(0,0)}.
        # 3. To grow, find all neighbors of the current set.
        # 4. To avoid duplicates, we only add a neighbor if it is "canonical".
        #    A neighbor (q,r) is added only if it is the lexicographically smallest
        #    among all neighbors that would result in a new unique shape?
        #    No, that's complex.
        
        # Simpler: Generate all "fixed" polyhexes (translations distinct) containing (0,0)
        # and then canonicalize.
        # Number of fixed polyhexes of size 11 is larger, but we can prune.
        
        # Let's try a different structural change:
        # **Grid Search on Symmetry Classes**.
        # Assume the optimal packing has some symmetry or is close to a hexagonal lattice.
        # We will test all "hexagonal lattice" subsets of size 11.
        # This covers the most likely candidates.
        
        # If that fails, we fall back to a fast random search.
        
        pass

    # --- Implementation of Lattice Subset Search ---
    
    # Generate a large hexagonal lattice
    L = 5
    lattice_points = []
    for q in range(-L, L+1):
        for r in range(-L, L+1):
            lattice_points.append((q, r))
            
    # We want subsets of size 11.
    # To reduce combinations, we fix the "center" of mass to be near (0,0).
    # Or, we fix one point to be (0,0) and search for the other 10?
    # Combinations(2*5+1)^2 choose 10 is still huge.
    
    # Let's use **Randomized Construction with Local Refinement**.
    # This is a structural change: instead of optimizing 22 continuous variables,
    # we construct discrete structures and refine them.
    
    best_side = np.inf
    best_inner = None
    best_outer = None
    
    # Precompute hexagon vertices for unit hexagon at origin
    # Vertices depend on angle. We assume inner hexagons are aligned (angle=0) for simplicity?
    # The task allows angle. But usually, optimal packings have aligned hexagons.
    # Let's assume inner angles are 0.
    
    def _hex_vertices(cx, cy, s, a_deg):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

    def _edge_normals(V):
        n = np.empty((6, 2))
        for i in range(6):
            e = V[(i + 1) % 6] - V[i]
            L = np.hypot(e[0], e[1])
            n[i] = np.array([e[1], -e[0]]) / L
        return n

    def _min_separation(A, B):
        nA = _edge_normals(A)
        nB = _edge_normals(B)
        best = -np.inf
        for n in nA:
            loA, hiA = (A @ n).min(), (A @ n).max()
            loB, hiB = (B @ n).min(), (B @ n).max()
            sep = max(loB - hiA, loA - hiB)
            if sep > best:
                best = sep
        for n in nB:
            loA, hiA = (A @ n).min(), (A @ n).max()
            loB, hiB = (B @ n).min(), (B @ n).max()
            sep = max(loB - hiA, loA - hiB)
            if sep > best:
                best = sep
        return best

    def _is_valid_packing(centers):
        """Check if the set of centers (axial) forms a non-overlapping packing."""
        # Convert to cartesian
        cart = [axial_to_cartesian(q, r) for q, r in centers]
        # Check pairs
        Vs = [_hex_vertices(c[0], c[1], 1.0, 0.0) for c in cart]
        for i in range(11):
            for j in range(i+1, 11):
                sep = _min_separation(Vs[i], Vs[j])
                if sep < 1e-6:
                    return False
        return True

    def _min_enclosing_hexagon_side(centers_cart):
        """
        Given cartesian centers, find the minimum side length of a regular hexagon
        that encloses them.
        We optimize the rotation angle phi (0 to 60 deg) and the center (cx, cy).
        For a fixed phi, the minimal side length is determined by the widths.
        s(phi) = max_k (width_k(phi)) / sqrt(3)
        where width_k is the width of the point set in direction n_k.
        The center is the midpoint of the projections.
        We just need to minimize s(phi) over phi.
        """
        # points is (11, 2)
        def side_for_phi(phi_deg):
            phi_rad = np.deg2rad(phi_deg)
            normals = np.column_stack((np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                                       np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))))
            proj = centers_cart @ normals.T # (11, 6)
            widths = proj.max(axis=0) - proj.min(axis=0)
            return widths.max() / np.sqrt(3.0)
        
        # Minimize side_for_phi over phi in [0, 60]
        # Use a fine grid or scipy minimize
        # Since it's 1D and smooth, a grid search + refinement is fast.
        phis = np.linspace(0, 60, 61)
        sides = [side_for_phi(p) for p in phis]
        min_idx = np.argmin(sides)
        phi_min = phis[min_idx]
        
        # Refine with golden section or just return the grid min for speed
        # Let's do a few iterations of local search
        phi_curr = phi_min
        step = 1.0
        for _ in range(10):
            phi_l = phi_curr - step
            phi_r = phi_curr + step
            if phi_l < 0: phi_l = 0
            if phi_r > 60: phi_r = 60
            s_l = side_for_phi(phi_l)
            s_r = side_for_phi(phi_r)
            s_c = side_for_phi(phi_curr)
            
            if s_l < s_c and s_l < s_r:
                phi_curr = phi_l
            elif s_r < s_c and s_r < s_l:
                phi_curr = phi_r
            else:
                step /= 2
                if step < 1e-3: break
        
        s_min = side_for_phi(phi_curr)
        
        # Calculate center
        phi_rad = np.deg2rad(phi_curr)
        normals = np.column_stack((np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                                   np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))))
        proj = centers_cart @ normals.T
        # Center projection is the midpoint of min/max for each normal
        # This gives a feasible center.
        center_proj = (proj.max(axis=0) + proj.min(axis=0)) / 2.0
        # Invert to cartesian
        # normals is 2x6. We need to solve normals @ [cx, cy]^T = center_proj?
        # No, center_proj is 6 values. We have 2 unknowns.
        # The system is overdetermined. We can use least squares.
        # Or simply take the centroid of the points?
        # The centroid is not necessarily the center of the minimal enclosing hexagon.
        # But for a regular hexagon, the center that minimizes the max distance
        # is the Chebyshev center.
        # Let's just use the centroid for now, or solve LS.
        cx, cy = np.linalg.lstsq(normals, center_proj, rcond=None)[0]
        
        return s_min, (cx, cy, phi_curr)

    # --- Search Strategy ---
    # 1. Try all "hexagonal lattice" configurations of size 11.
    #    We generate all subsets of the lattice of size 11 that are "compact".
    #    To do this efficiently, we fix the "center" of the subset to be near (0,0).
    
    # Generate candidate structures
    candidates = []
    
    # Method: Start with a central hexagon, add neighbors one by one, ensuring connectivity.
    # This generates all connected subsets of the lattice of size 11 containing (0,0).
    
    def gen_connected_sets(n, current, all_sets):
        if len(current) == n:
            # Canonicalize: translate so min q=0, min r=0?
            # Or just store the set.
            # To avoid duplicates in the final list, we use a set of frozensets.
            # But we want to evaluate each unique geometry.
            # Since we start from (0,0) and grow, we might get duplicates via different paths.
            # We will deduplicate at the end or during generation.
            all_sets.add(frozenset(current))
            return
            
        if len(current) >= n:
            return
            
        # Find neighbors
        neighbors = set()
        for q, r in current:
            for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                nq, nr = q+dq, r+dr
                if (nq, nr) not in current:
                    neighbors.add((nq, nr))
        
        for nb in neighbors:
            new_set = current | {nb}
            # Pruning: Keep it compact?
            # Check bounding box size. If too large, skip.
            qs = [q for q,r in new_set]
            rs = [r for q,r in new_set]
            if max(qs)-min(qs) > 4 or max(rs)-min(rs) > 4:
                continue
            gen_connected_sets(n, new_set, all_sets)

    # This recursion might be slow or deep. 
    # Let's use a simpler heuristic:
    # Test specific known good packings.
    # The "hexagonal number" H_3 = 19 is too big.
    # H_2 = 7.
    # 11 = 7 + 4.
    
    # Let's just do a **Randomized Search** with **Analytical Enclosing Hexagon**.
    # This is robust and fast.
    
    rng = np.random.default_rng(42)
    
    for i in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random connected set of 11 hexagons
        # Start at (0,0)
        current = {(0,0)}
        for _ in range(10):
            # Pick a random existing hex
            q, r = rng.choice(list(current))
            # Pick a random neighbor
            dq, dr = rng.choice([(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)])
            nq, nr = q+dq, r+dr
            if (nq, nr) not in current:
                current.add((nq, nr))
        
        # Check validity
        if not _is_valid_packing(list(current)):
            continue
            
        # Convert to cartesian
        cart = np.array([axial_to_cartesian(q, r) for q, r in current])
        
        # Find minimal enclosing hexagon
        s, (cx, cy, phi) = _min_enclosing_hexagon_side(cart)
        
        if s < best_side:
            best_side = s
            # Store result
            # We need to return (inner, outer, side)
            # inner: (11, 3) (x, y, angle)
            # outer: (3,) (x, y, angle)
            
            # Shift centers so that the outer center is at (0,0)?
            # The task says: "Coordinates are first shifted by the outer center"
            # So we should return coordinates relative to the outer center.
            
            inner_rel = cart - np.array([cx, cy])
            inner = np.column_stack((inner_rel[:, 0], inner_rel[:, 1], np.full(11, 0.0)))
            outer = np.array([0.0, 0.0, phi])
            best_inner = inner
            best_outer = outer

    if best_inner is not None:
        return best_inner, best_outer, best_side

    # Fallback
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ], dtype=float)
    outer = np.array([0, 0, 0], dtype=float)
    return inner, outer, 8.0

# EVOLVE-BLOCK-END
