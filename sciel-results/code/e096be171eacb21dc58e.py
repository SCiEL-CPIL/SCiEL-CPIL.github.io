"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_k4_free(edge_set, n=18):
    """Check if graph with given edge_set is K4-free using bitmask."""
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check all 4-subsets for K4
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b + 1, n):
                if not ((adj[a] >> c) & 1) or not ((adj[b] >> c) & 1):
                    continue
                # Check if a,b,c form a triangle; now check if any d forms K4
                common = adj[a] & adj[b] & adj[c]
                # Remove a,b,c from common
                common &= ~(1 << a) & ~(1 << b) & ~(1 << c)
                if common:
                    return False
    return True

def _max_triangle_free_size(edge_set, n=18):
    """Find max size of triangle-free induced subgraph using branch and bound."""
    adj = [0] * n
    for u, v in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def has_triangle(mask):
        """Check if induced subgraph on mask has a triangle."""
        verts = [i for i in range(n) if (mask >> i) & 1]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                a, b = verts[i], verts[j]
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b] & mask
                common &= ~(1 << a) & ~(1 << b)
                if common:
                    return True
        return False
    
    # Branch and bound: try to find largest triangle-free subset
    best = 0
    full_mask = (1 << n) - 1
    
    # Upper bound: for a set S, max triangle-free subset <= |S|
    # We use a recursive approach with pruning
    def branch(mask, size):
        nonlocal best
        if size > best:
            best = size
        if mask == 0:
            return
        # Upper bound: size + popcount(mask)
        remaining = bin(mask).count('1')
        if size + remaining <= best:
            return
        # Pick a vertex
        v = (mask & -mask).bit_length() - 1
        # Option 1: include v
        new_mask = mask & ~(1 << v)
        # Check if adding v creates a triangle with existing vertices in current solution
        # Actually we need to track which vertices are in the current triangle-free set
        # Let me restructure: use a different approach
        
    # Better approach: iterate over all subsets from large to small
    # For n=18, we can use a smarter search
    # Try all subsets of size k from 18 down, check if triangle-free
    
    # Precompute: for each triple, check if it's a triangle
    # A subset is triangle-free iff it contains no triangle
    
    # Use the fact that we want the LARGEST triangle-free induced subgraph
    # Equivalent to: minimum vertex deletion to make graph triangle-free (complement)
    
    # Greedy + local search approach for speed
    # Start with all vertices, remove vertices to break triangles
    # Actually, let's use a different strategy:
    # Find the minimum vertex cover of the hypergraph of triangles
    
    # Get all triangles
    triangles = []
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            common &= ~(1 << a) & ~(1 << b)
            c = 0
            while common:
                bit = common & -common
                c = bit.bit_length() - 1
                triangles.append((a, b, c))
                common ^= bit
    
    if not triangles:
        return n
    
    # Minimum vertex cover in triangle hypergraph
    # Greedy: repeatedly pick vertex in most triangles
    best_size = n
    # Try to find a small hitting set
    # Use randomized greedy + local improvement
    
    def greedy_hitting_set(triangles, adj, n, seed=0):
        rng = random.Random(seed)
        # Weighted greedy
        tri_list = list(triangles)
        if not tri_list:
            return 0
        vertex_tri_count = [0] * n
        for t in tri_list:
            for v in t:
                vertex_tri_count[v] += 1
        
        removed = set()
        remaining = set(range(n))
        
        while True:
            # Find triangles not yet hit
            active_tris = [t for t in tri_list if not any(v in removed for v in t)]
            if not active_tris:
                break
            # Pick vertex in most active triangles
            counts = [0] * n
            for t in active_tris:
                for v in t:
                    if v in remaining:
                        counts[v] += 1
            # Pick max, break ties randomly
            max_count = max(counts)
            candidates = [v for v in remaining if counts[v] == max_count]
            v = rng.choice(candidates)
            removed.add(v)
            remaining.discard(v)
        
        return len(removed)
    
    # Try multiple seeds
    min_removed = n
    for seed in range(50):
        r = greedy_hitting_set(triangles, adj, n, seed)
        if r < min_removed:
            min_removed = r
    
    return n - min_removed

def _compute_score(edge_list):
    """Compute the score (largest triangle-free induced subgraph size)."""
    n = 18
    edge_set = set()
    for u, v in edge_list:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    if not _is_k4_free(list(edge_set), n):
        return 999  # Invalid
    return _max_triangle_free_size(list(edge_set), n)

def construct():
    n = 18
    deadline = time.time() + 110  # Leave 40s margin
    
    # Start with known good construction: K6,6,6 gives score 12
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    base_edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    base_edges.append([a, b])
    base_score = _compute_score(base_edges)
    best_edges = base_edges
    best_score = base_score
    
    # Try to improve with random K4-free graphs
    # Strategy: random graph, then remove edges to make K4-free
    # Or: build greedily ensuring K4-free
    
    def random_k4_free_graph(n, seed):
        rng = random.Random(seed)
        # Start with empty graph, add edges greedily
        adj = [0] * n
        edges = []
        all_pairs = list(combinations(range(n), 2))
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            # Check if adding (u,v) creates a K4
            # A K4 containing (u,v) needs two other vertices a,b such that
            # u-v, u-a, u-b, v-a, v-b, a-b all exist
            # So we need a,b both adjacent to both u and v, and a-b edge exists
            common = adj[u] & adj[v]
            common &= ~(1 << u) & ~(1 << v)
            # Check if any pair in common has an edge
            c = 0
            conflict = False
            while common:
                bit = common & -common
                c = bit.bit_length() - 1
                if (adj[c] >> v) & 1 and (adj[c] >> u) & 1:
                    # c is adjacent to both u and v
                    # Check if any other d in common is adjacent to c
                    rest = common & ~(1 << c)
                    d = 0
                    while rest:
                        bit2 = rest & -rest
                        d = bit2.bit_length() - 1
                        if (adj[c] >> d) & 1:
                            conflict = True
                            break
                        rest ^= bit2
                    if conflict:
                        break
                common ^= bit
            if not conflict:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return edges
    
    # Try many random K4-free graphs
    num_trials = 0
    for seed in range(10000):
        if time.time() > deadline:
            break
        edges = random_k4_free_graph(n, seed)
        score = _compute_score(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            print(f"Found better: score={score}, seed={seed}", flush=True)
        num_trials += 1
    
    # Also try complement of random regular-ish graphs
    for seed in range(5000):
        if time.time() > deadline:
            break
        # Generate a random graph, take complement, check K4-free
        rng = random.Random(seed + 100000)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.3:  # Sparse graph
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        # Complement
        comp_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    comp_edges.append([u, v])
        score = _compute_score(comp_edges)
        if score < best_score:
            best_score = score
            best_edges = comp_edges
            print(f"Found better (complement): score={score}, seed={seed}", flush=True)
    
    return best_edges
# EVOLVE-BLOCK-END
