"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

D = 11


def _all_vectors():
    """Generate all nonzero integer vectors with small coordinates.
    We use vectors with entries in {-2,-1,0,1,2} but restrict to those
    with norm <= 4 (i.e., sum of squares <= 16) to keep the set manageable.
    Actually, let's be more careful: we want vectors whose pairwise distances
    are at least the max norm. Let's generate vectors with entries in {-1,0,1}
    first (the seed), then try to extend.
    
    For a kissing configuration, we need ||v_i - v_j|| >= max_i ||v_i|| for all i != j.
    
    Let's generate candidate vectors with small integer entries and then find
    the largest subset satisfying the constraint.
    """
    # Generate vectors with entries in {-1, 0, 1}
    # There are 3^11 - 1 = 177146 nonzero vectors. That's too many for brute force MIS.
    # Let's be smarter.
    
    # Strategy: use vectors with exactly 2 nonzero entries (the seed gives 220).
    # Then try to add vectors with 1 nonzero entry (norm 1) - but wait, if max norm is 2,
    # then we need distance >= 2. A vector with 1 nonzero entry (norm 1) and a vector
    # with 2 nonzero entries (norm 2): their distance could be as small as... 
    # e.g., e1 and e1+e2: distance = ||e2|| = 1 < 2. So that doesn't work.
    
    # What if all vectors have the same norm? That's the classic kissing number setup.
    # If all vectors have norm r, then we need ||v_i - v_j|| >= r for all i != j.
    # Equivalently, (v_i - v_j).(v_i - v_j) >= r^2
    # v_i.v_i + v_j.v_j - 2 v_i.v_j >= r^2
    # 2r^2 - 2 v_i.v_j >= r^2
    # v_i.v_j <= r^2/2
    
    # For vectors with entries in {-1,0,1} and exactly k nonzero entries, norm^2 = k.
    # So we need v_i.v_j <= k/2.
    
    # Let's try k=2 (norm^2 = 2): need v_i.v_j <= 1.
    # Two vectors with 2 nonzero entries: their dot product can be -2,-1,0,1,2.
    # We need dot product <= 1, so they can't share both positions with same signs.
    # This is exactly what the seed does! The seed generates all vectors with exactly 2
    # nonzero entries (±1), and the constraint is that no two vectors are identical or negatives...
    # wait, let me re-examine.
    
    # Actually the seed generates all vectors with exactly 2 nonzero entries, each ±1.
    # Total: C(11,2) * 4 = 55 * 4 = 220 vectors.
    # The minimum pairwise distance among these: two vectors sharing one position.
    # e.g., e1+e2 and e1+e3: distance = ||e2-e3|| = sqrt(2). Max norm = sqrt(2). OK.
    # e1+e2 and e1-e2: distance = ||2e2|| = 2. OK.
    # e1+e2 and -e1-e2: distance = ||2e1+2e2|| = 2*sqrt(2) >= sqrt(2). OK.
    # So all 220 are valid? Let me check e1+e2 and e2+e1 = same vector. They're the same!
    # Wait, the seed generates (i,j) with i<j, and si,sj in {1,-1}. So e1+e2 appears once.
    # What about e2+e1? That's the same as e1+e2 since we fix i<j. So no duplicates.
    # 
    # So the seed gives 220 vectors all with norm sqrt(2), and min distance is sqrt(2).
    # 
    # Can we do better? What about adding vectors with 3 nonzero entries?
    # Norm^2 = 3. Then we'd need all pairwise distances >= sqrt(3).
    # But a vector with norm sqrt(2) and one with norm sqrt(3): distance could be small.
    # 
    # Alternatively, use only vectors with 3 nonzero entries. Norm^2 = 3.
    # Need v_i.v_j <= 3/2 = 1.5, so v_i.v_j <= 1 (integer).
    # Number of such vectors: C(11,3) * 8 = 165 * 8 = 1320.
    # We need to find the largest subset with pairwise dot product <= 1.
    # 
    # Or mix: use vectors with 2 nonzero entries (norm^2=2) and 3 nonzero entries (norm^2=3).
    # Then max norm^2 = 3, so we need all pairwise distances >= sqrt(3).
    # For two 2-norm vectors: distance^2 = 2+2-2*dot = 4-2*dot. Need >= 3, so dot <= 0.5, i.e., dot <= 0.
    # For two 3-norm vectors: distance^2 = 6-2*dot. Need >= 3, so dot <= 1.5, i.e., dot <= 1.
    # For a 2-norm and 3-norm: distance^2 = 2+3-2*dot = 5-2*dot. Need >= 3, so dot <= 1.
    
    # This is getting complex. Let me try a different approach:
    # Generate a pool of candidate vectors, build a conflict graph, and find MIS.
    
    # Pool: vectors with entries in {-1,0,1}, with exactly 2 or 3 nonzero entries.
    # For 2-nonzero: C(11,2)*4 = 220
    # For 3-nonzero: C(11,3)*8 = 1320
    # Total: 1540 candidates.
    
    # But the constraint depends on the max norm in the selected set.
    # If we select only 2-nonzero vectors, max norm^2 = 2.
    # If we select any 3-nonzero vector, max norm^2 = 3.
    
    # Let's try: select only 3-nonzero vectors. Max norm^2 = 3.
    # Constraint: for any two selected vectors, ||v_i - v_j||^2 >= 3.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j = 6 - 2*v_i.v_j >= 3
    # => v_i.v_j <= 1.5 => v_i.v_j <= 1 (since integer).
    
    # So we need the largest subset of 3-nonzero vectors with pairwise dot product <= 1.
    
    # Two 3-nonzero vectors have dot product in {-3,-2,-1,0,1,2,3}.
    # Conflict if dot product >= 2.
    
    # Let's generate these and build the conflict graph.
    
    vecs = []
    # 3-nonzero vectors
    for idx in combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[idx[0]] = s1
                    v[idx[1]] = s2
                    v[idx[2]] = s3
                    vecs.append(np.array(v, dtype=np.int64))
    
    return vecs


def _build_conflict_graph(vecs, max_norm_sq):
    """Build conflict graph. Two vectors conflict if ||v_i - v_j||^2 < max_norm_sq."""
    n = len(vecs)
    # Compute all pairwise dot products
    # vecs is (n, D)
    dots = vecs @ vecs.T  # (n, n)
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*dots[i,j]
    norms_sq = np.sum(vecs**2, axis=1)  # (n,)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots  # (n, n)
    
    # Conflict if dist_sq < max_norm_sq (strictly less, since we need >= max_norm)
    # Wait, the task says "minimum pairwise distance is at least the maximum vector norm"
    # So ||v_i - v_j|| >= max(||v_k||) for all i != j.
    # dist_sq >= max_norm_sq
    
    conflict = (dist_sq < max_norm_sq).astype(np.int64)
    np.fill_diagonal(conflict, 0)
    return conflict


def _greedy_mis(conflict, n):
    """Greedy MIS: repeatedly pick the vertex with minimum degree in remaining graph."""
    selected = []
    remaining = set(range(n))
    # Use a copy of conflict for efficiency
    conf = conflict.copy()
    
    while remaining:
        # Find vertex with minimum degree among remaining
        min_deg = float('inf')
        best_v = None
        for v in remaining:
            deg = sum(conf[v][u] for u in remaining if u != v)
            if deg < min_deg:
                min_deg = deg
                best_v = v
        selected.append(best_v)
        remaining.discard(best_v)
        # Remove all neighbors of best_v from remaining
        for u in list(remaining):
            if conf[best_v][u]:
                remaining.discard(u)
    
    return selected


def _branch_and_bound_mis(conflict, n, time_limit=100):
    """Branch and bound MIS with Tomita coloring bound.
    Uses Python integers as bitsets for speed."""
    start_time = time.time()
    
    # Convert conflict to bitsets
    # adj[v] is a Python int with bit u set if u is adjacent to v
    adj = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if v != u and conflict[v][u]:
                mask |= (1 << u)
        adj[v] = mask
    
    # Tomita's algorithm for maximum independent set
    best_size = 0
    best_set = []
    
    # Color the graph for upper bound
    # We'll use a simple greedy coloring
    def color_graph(vertices):
        """Greedy coloring of the subgraph induced by vertices.
        Returns list of colors (color class sizes)."""
        colors = []
        uncolored = set(vertices)
        while uncolored:
            color_class = set()
            # Pick a maximal independent set in the remaining
            # Greedy: pick vertex, remove its neighbors
            remaining = set(uncolored)
            while remaining:
                v = min(remaining)  # deterministic
                color_class.add(v)
                # Remove v and its neighbors
                neighbors_in_remaining = set()
                for u in remaining:
                    if (adj[v] >> u) & 1:
                        neighbors_in_remaining.add(u)
                remaining -= color_class
                remaining -= neighbors_in_remaining
            colors.append(len(color_class))
            uncolored -= color_class
        return colors
    
    def bb(R, P, current_set):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound: |R| + max possible from P
        # Simple bound: |R| + |P|
        if len(R) + len(P) <= best_size:
            return
        
        # Tomita bound: color P, upper bound is |R| + sum of color class sizes up to needed
        P_list = list(P)
        colors = color_graph(P_list)
        # Upper bound: |R| + len(P_list) but tighter: 
        # The number of vertices we can add from P is at most the number of color classes
        # Actually, in a proper coloring, an independent set can have at most one vertex per color class.
        # So upper bound from P is len(colors) (number of color classes).
        # Wait no, that's for the coloring of P. An independent set in P can have at most
        # one vertex from each color class, so at most len(colors) vertices.
        # But that's a weak bound. Better: the chromatic number gives a lower bound on clique,
        # but for independent set, the bound is different.
        
        # Actually, Tomita's bound uses the coloring of P to get an upper bound on the
        # size of a maximum independent set in P. Since P is colored with c colors,
        # any independent set in P has size at most... no, that's not right.
        # The coloring gives that the independence number is at most the number of vertices,
        # but that's trivial.
        
        # Let me use a simpler approach: just use |R| + |P| as the bound, and order
        # vertices by degree.
        
        # Order P by degree in the subgraph
        P_sorted = sorted(P_list, key=lambda v: bin(adj[v] & ((1 << len(P_list)) - 1)).count('1') if False else sum(1 for u in P_list if u != v and (adj[v] >> u) & 1), reverse=True)
        
        for v in P_sorted:
            # Branch: include v
            R_new = R | {v}
            # P_new: vertices in P that are not adjacent to v and not v itself
            P_new = {u for u in P_list if u != v and not ((adj[v] >> u) & 1)}
            
            bb(R_new, P_new, current_set | {v})
            
            # Branch: exclude v
            P_list.remove(v)
            if not P_list:
                break
            # Update bound
            if len(R) + len(P_list) <= best_size:
                return
    
    # Start with all vertices
    all_vertices = set(range(n))
    # Sort by degree for better pruning
    all_vertices_sorted = sorted(all_vertices, key=lambda v: bin(adj[v]).count('1'), reverse=True)
    
    # Use the standard Tomita approach
    best_size = 0
    best_set = []
    
    def bb_tomita(P, current_set):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        P_list = list(P)
        
        # Color for bound
        # Simple: upper bound is len(current_set) + len(P_list)
        if len(current_set) + len(P_list) <= best_size:
            return
        
        # Greedy coloring of P for tighter bound
        # We'll do a simple sequential coloring
        color_map = {}
        colors_used = 0
        uncolored = list(P_list)
        while uncolored:
            color_class = []
            i = 0
            while i < len(uncolored):
                v = uncolored[i]
                # Check if v can be added to current color class
                can_add = True
                for u in color_class:
                    if (adj[v] >> u) & 1:
                        can_add = False
                        break
                if can_add:
                    color_class.append(v)
                    color_map[v] = colors_used
                i += 1
            uncolored = [v for v in uncolored if v not in color_map]
            colors_used += 1
        
        # Upper bound: current_set + number of color classes (each color class contributes at most 1 to an independent set)
        # Wait, that's wrong. An independent set can have multiple vertices from different color classes.
        # The correct bound: in a proper coloring with c colors, the independence number is at most n (trivial).
        # Actually, the coloring bound for MIS is: alpha(G) <= sum over color classes of 1... no.
        
        # Let me just use a simple bound and rely on good vertex ordering.
        
        # Order by degree descending
        P_sorted = sorted(P_list, key=lambda v: sum(1 for u in P_list if u != v and (adj[v] >> u) & 1), reverse=True)
        
        for v in P_sorted:
            # Include v
            P_new = {u for u in P_list if u != v and not ((adj[v] >> u) & 1)}
            bb_tomita(P_new, current_set | {v})
            
            # Exclude v
            P_list.remove(v)
            if len(current_set) + len(P_list) <= best_size:
                return
    
    bb_tomita(set(range(n)), set())
    
    return list(best_set)


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Generate candidate vectors with 3 nonzero entries
    vecs = _all_vectors()
    n = len(vecs)
    max_norm_sq = 3  # since all have 3 nonzero entries with ±1
    
    # Build conflict graph
    conflict = _build_conflict_graph(vecs, max_norm_sq)
    
    # Try branch and bound with time limit
    remaining_time = deadline - time.time() - 5  # 5 second safety margin
    if remaining_time > 5:
        try:
            selected = _branch_and_bound_mis(conflict, n, time_limit=remaining_time)
        except Exception:
            selected = _greedy_mis(conflict, n)
    else:
        selected = _greedy_mis(conflict, n)
    
    # Fallback: if B&B didn't find anything, use greedy
    if not selected:
        selected = _greedy_mis(conflict, n)
    
    result = vecs[selected]
    return result
# EVOLVE-BLOCK-END
