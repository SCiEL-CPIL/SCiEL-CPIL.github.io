<<<<<<< SEARCH
import time
import random
import itertools

# Precompute Q3 edge sets on vertices 0..7
def generate_q3_edge_sets():
    # Q3 vertices are 0..7. Edges connect i, j if i xor j is power of 2.
    # But we need all isomorphic copies? No, just the structure on 0..7.
    # The set of edges of Q3 on 0..7 is fixed if we fix the labeling.
    # But we are checking if the induced subgraph CONTAINS a Q3.
    # So we need to check if the induced subgraph (on 8 vertices) has a subgraph isomorphic to Q3.
    # This is equivalent to checking if the edge set of the induced subgraph
    # contains any of the 840 edge sets corresponding to Q3 labelings.
    
    # Standard Q3 edges on 0..7 (binary 000..111)
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor == 1 or xor == 2 or xor == 4:
                base_edges.append((i, j))
                
    # Generate all permutations of 0..7
    # For each perm, map base_edges to permuted edges
    q3_sets = set()
    for perm in itertools.permutations(range(8)):
        # perm[i] is the vertex in the graph corresponding to Q3 vertex i
        # Wait, we want to map Q3 vertices to graph vertices.
        # Let's say Q3 vertex i is mapped to graph vertex perm[i].
        # Edge (i, j) in Q3 becomes (perm[i], perm[j]) in graph.
        # We want the set of edges in the graph (on 0..7) that form a Q3.
        e_set = frozenset(
            (min(perm[i], perm[j]), max(perm[i], perm[j])) 
            for (i, j) in base_edges
        )
        q3_sets.add(e_set)
        
    return list(q3_sets)

Q3_EDGE_SETS = generate_q3_edge_sets()

def has_q3(adj, n=16):
    # adj is list of ints (bitmasks)
    # Iterate through all 8-vertex subsets
    for vertices in itertools.combinations(range(n), 8):
        # Build induced edge set on these vertices
        # Map vertices to 0..7
        v_map = {v: i for i, v in enumerate(vertices)}
        
        induced_edges = set()
        for i in range(8):
            u = vertices[i]
            # Check neighbors in vertices
            mask = adj[u]
            for j in range(i+1, 8):
                v = vertices[j]
                if mask & (1 << v):
                    induced_edges.add((i, j))
                    
        # Check if any Q3 edge set is a subset of induced_edges
        # induced_edges is a set of (i,j) with i<j
        # Q3_EDGE_SETS are frozensets of (i,j) with i<j
        for q3_set in Q3_EDGE_SETS:
            if q3_set <= induced_edges:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    # Start with K3,13
    edges = set()
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    best_edges = set(edges)
    
    # Candidates: all edges not in best_edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidates = [e for e in all_possible if e not in best_edges]
    
    # Local search
    while time.time() < deadline:
        random.shuffle(candidates)
        improved = False
        for e in candidates:
            if time.time() > deadline:
                break
                
            u, v = e
            # Try adding e
            best_edges.add(e)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3(adj, n):
                # Remove e
                best_edges.remove(e)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep e
                candidates.remove(e)
                improved = True
                
        if not improved:
            # No more edges can be added in one pass
            # Try a different strategy or break?
            # Maybe restart with a different seed?
            # For now, break to save time, or try random removals?
            # Let's just break.
            break
            
    return [list(e) for e in best_edges]
=======
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n = 16
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_possible_set = set(all_possible)
    
    # Precompute: for each vertex, its incident edges
    incident = [[] for _ in range(n)]
    for u, v in all_possible:
        incident[u].append(v)
        incident[v].append(u)
    
    def has_q3_with_edge(adj, edge, n=16):
        """Check if adding 'edge' to adj creates a Q3.
        Only check 8-sets that contain both endpoints of 'edge'."""
        u, v = edge
        # Find all 6 other vertices from the remaining 14
        others = [x for x in range(n) if x != u and x != v]
        for six in itertools.combinations(others, 6):
            vertices = [u, v] + list(six)
            # Build induced edge set on these 8 vertices
            # Map to indices 0..7
            vmap = {vert: idx for idx, vert in enumerate(vertices)}
            
            # Count edges in induced subgraph - Q3 has exactly 12 edges
            # But we need to check if it CONTAINS a Q3 (subgraph, not induced)
            # So we need: the induced subgraph has a subgraph isomorphic to Q3
            # This means there exist 8 vertices where the edge set contains all 12 Q3 edges
            
            # Quick check: induced subgraph must have at least 12 edges
            edge_count = 0
            for i in range(8):
                a = vertices[i]
                for j in range(i+1, 8):
                    b = vertices[j]
                    if adj[a] & (1 << b):
                        edge_count += 1
            if edge_count < 12:
                continue
                
            # Build bitset of present edges (14 possible pairs -> 14 bits)
            # Pair index: (i,j) where i<j in 0..7 -> index = i*7 + j - i*(i+1)//2
            # Simpler: use a 14-bit mask
            edge_mask = 0
            pair_idx = 0
            for i in range(8):
                for j in range(i+1, 8):
                    a = vertices[i]
                    b = vertices[j]
                    if adj[a] & (1 << b):
                        edge_mask |= (1 << pair_idx)
                    pair_idx += 1
            
            # Check against precomputed Q3 masks for this vertex ordering
            # But Q3 can be labeled in many ways... 
            # Instead, check: does the edge_mask contain any of the 168 Q3 patterns?
            # Actually, let's use a different approach:
            # For the 8 vertices, check if they contain a Q3 by trying all bijections
            
            # Faster: use the fact that Q3 is bipartite with parts of size 4
            # The 12 edges of Q3 form a specific pattern
            
            # Let's just check all 8! / 2 = 20160 labelings? No, too slow.
            # Better: precompute all valid Q3 edge masks on 8 labeled vertices
            
            # Actually, let me use a cleaner approach with precomputation
            if _has_q3_in_8(vertices, adj, n):
                return True
        return False
    
    # Precompute Q3 edge masks for all 8-vertex orderings
    # A Q3 on vertices 0..7 has edges where xor is 1, 2, or 4
    # We need all distinct edge sets under permutation
    
    _q3_cache = {}
    
    def _get_q3_masks_for_order(vertices):
        """Get all Q3 edge masks (as 14-bit integers) for a given vertex ordering."""
        key = tuple(vertices)
        if key in _q3_cache:
            return _q3_cache[key]
        
        # For this specific set of 8 vertices, compute which 14-bit masks correspond to Q3
        # We need to check: is there a bijection from {0..7} to vertices such that
        # the image of the Q3 edge set is a subset of the present edges?
        
        # Precompute base Q3 edges on 0..7
        base_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                xor = i ^ j
                if xor in (1, 2, 4):
                    base_edges.append((i, j))
        
        # For each permutation of the 8 vertices, compute the edge mask
        masks = set()
        for perm in itertools.permutations(range(8)):
            # perm[i] = which position in vertices corresponds to Q3 vertex i
            mask = 0
            for (a, b) in base_edges:
                pos_a = perm[a]
                pos_b = perm[b]
                if pos_a > pos_b:
                    pos_a, pos_b = pos_b, pos_a
                # Get pair index
                pair_idx = pos_a * 7 - pos_a * (pos_a + 1) // 2 + (pos_b - pos_a - 1)
                # Simpler: precompute pair indices
                # Actually let me just use a lookup
                idx = _pair_index(pos_a, pos_b)
                mask |= (1 << idx)
            masks.add(mask)
        
        _q3_cache[key] = masks
        return masks
    
    def _pair_index(i, j):
        """Get the index of pair (i,j) with i<j in the 14 pairs of 8 vertices."""
        # Pairs: (0,1),(0,2),...,(0,7),(1,2),...,(1,7),...,(6,7)
        # Index = sum_{k=0}^{i-1} (7-k) + (j-i-1)
        # = 7*i - i*(i-1)/2 + j - i - 1
        # = 6*i - i*(i-1)/2 + j - 1
        # Let me just precompute
        return _PAIR_IDX[i][j]
    
    # Precompute pair indices
    _PAIR_IDX = [[-1]*8 for _ in range(8)]
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            _PAIR_IDX[i][j] = idx
            _PAIR_IDX[j][i] = idx
            idx += 1
    
    def _has_q3_in_8(vertices, adj, n=16):
        """Check if the induced subgraph on 'vertices' contains a Q3."""
        key = tuple(sorted(vertices))
        # Get Q3 masks for this vertex set
        # We need to map vertices to positions 0..7
        # The Q3 masks are computed for a specific ordering
        
        # Actually, let me simplify: for the given set of 8 vertices,
        # compute the edge mask (14 bits), then check if it contains any Q3 mask
        
        # Map vertices to 0..7
        vmap = {v: i for i, v in enumerate(vertices)}
        
        # Compute edge mask
        edge_mask = 0
        for i in range(8):
            a = vertices[i]
            for j in range(i+1, 8):
                b = vertices[j]
                if adj[a] & (1 << b):
                    edge_mask |= (1 << _PAIR_IDX[i][j])
        
        # Get all Q3 masks for this vertex ordering
        q3_masks = _get_q3_masks_for_order(vertices)
        
        # Check if edge_mask contains any q3_mask
        for q3m in q3_masks:
            if (edge_mask & q3m) == q3m:
                return True
        return False
    
    def greedy_fill(adj, edges, candidates, deadline):
        """Greedy: try to add each candidate edge if it doesn't create a Q3."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(candidates)
            for e in candidates[:]:
                if time.time() > deadline:
                    return
                u, v = e
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add(e)
                
                if has_q3_with_edge(adj, e, n):
                    # Remove
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges.discard(e)
                else:
                    candidates.remove(e)
                    improved = True
    
    def random_restart(adj, edges, candidates, deadline, num_removals):
        """Remove some random edges and try to re-add them plus others."""
        # Remove random edges
        removed = []
        removable = list(edges)
        for _ in range(min(num_removals, len(removable))):
            if not removable:
                break
            e = random.choice(removable)
            removable.remove(e)
            u, v = e
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.discard(e)
            removed.append(e)
        
        # Add removed edges back to candidates
        for e in removed:
            if e not in candidates:
                candidates.append(e)
        
        # Try to fill again
        greedy_fill(adj, edges, candidates, deadline)
    
    best_edge_count = -1
    best_edges_set = None
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random sparse graph
        adj = [0] * n
        edges = set()
        
        # Start with a random set of edges
        random.shuffle(all_possible)
        num_initial = random.randint(20, 40)
        for e in all_possible[:num_initial]:
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        
        # Check if initial graph is Q3-free (it should be, but verify)
        # For small graphs, just proceed
        candidates = [e for e in all_possible if e not in edges]
        
        # Greedy fill
        greedy_fill(adj, edges, candidates, deadline)
        
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges_set = set(edges)
        
        # Try random restarts with removal
        if time.time() > deadline:
            break
        for _ in range(5):
            if time.time() > deadline:
                break
            adj2 = [0] * n
            edges2 = set(edges)
            for u, v in edges2:
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
            candidates2 = [e for e in all_possible if e not in edges2]
            
            random_restart(adj2, edges2, candidates2, deadline, random.randint(2, 8))
            
            if len(edges2) > best_edge_count:
                best_edge_count = len(edges2)
                best_edges_set = set(edges2)
    
    if best_edges_set is None:
        # Fallback: return empty
        return []
    
    return [list(e) for e in best_edges_set]
>>>>>>> REPLACE

I made a mistake - I'm redefining `_PAIR_IDX` after using it in `_pair_index`. Let me fix the order and clean up:

