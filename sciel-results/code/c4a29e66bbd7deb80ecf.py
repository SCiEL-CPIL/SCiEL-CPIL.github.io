import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 60.0

    n = 12
    all_pairs = list(combinations(range(n), 2))
    # Index mapping for pairs
    pair_index = {p: i for i, p in enumerate(all_pairs)}
    m = len(all_pairs)

    # Precompute triangle constraints: for each edge, which other edges form a triangle with it
    # edge (u,v) and (v,w) and (u,w) form a triangle
    # For each edge e, list of edges that conflict with e (forming a triangle)
    conflict = [[] for _ in range(m)]
    for i, (u, v) in enumerate(all_pairs):
        for w in range(n):
            if w == u or w == v:
                continue
            # edge (u,w) and (v,w)
            uw = (min(u, w), max(u, w))
            vw = (min(v, w), max(v, w))
            if uw in pair_index and vw in pair_index:
                conflict[i].append(pair_index[uw])
                conflict[i].append(pair_index[vw])

    # Remove duplicates in conflict lists
    for i in range(m):
        conflict[i] = list(set(conflict[i]))

    # We want to find a graph with min degree >= 3, triangle-free, girth >= 4,
    # and minimize the number of distinct cycle lengths.
    #
    # Strategy: Search for graphs where all cycles have the same length.
    # The most promising candidate is a graph where all cycles are 4-cycles.
    # A bipartite graph with parts of size 6, where each vertex has degree 3,
    # and the only cycles are 4-cycles.
    #
    # Actually, let's think about this more carefully.
    # A 3-regular bipartite graph on 12 vertices.
    # If we can make it so that the only cycles are 4-cycles, then raw value = 1.
    #
    # One way: take the incidence graph of a projective plane of order 2 (Fano plane).
    # The Fano plane has 7 points and 7 lines. The incidence graph has 14 vertices.
    # That's too many.
    #
    # What about the 3x3 grid graph? No, that's 9 vertices.
    #
    # Let's try: a 3-regular bipartite graph where the bipartite parts are A and B, |A|=|B|=6.
    # If every 4-cycle is a "square" and there are no longer cycles, we're good.
    # But in a 3-regular bipartite graph with 12 vertices, by the handshake lemma,
    # there are 18 edges. The number of 4-cycles can be computed.
    #
    # Actually, the simplest approach: try to find a graph where all cycles are 4-cycles.
    # This is equivalent to a graph where the girth is 4 and the circumference is 4.
    # Such graphs are called "4-cycle-only" graphs.
    #
    # A well-known example: the complete bipartite graph K_{3,3} has cycles of length 4 and 6.
    # So K_{3,3} doesn't work.
    #
    # What about two disjoint 4-cycles with some additional edges? No, min degree 3.
    #
    # Let's try a different approach: use a random search to find a graph with
    # as few distinct cycle lengths as possible.

    import random
    random.seed(42)

    best_edges = None
    best_cycle_lengths = None
    best_count = float('inf')

    # Precompute adjacency for quick cycle length enumeration
    def get_cycle_lengths(adj, n):
        """Enumerate all simple cycles and return the set of cycle lengths."""
        cycle_lengths = set()
        
        # For each vertex, do DFS to find cycles
        # To avoid duplicates, only count cycles where the smallest vertex is the start
        for start in range(n):
            # DFS with path tracking
            stack = [(start, [start], set([start]))]
            while stack:
                node, path, visited = stack.pop()
                for nb in adj[node]:
                    if nb == start and len(path) >= 4:
                        # Found a cycle
                        cycle_lengths.add(len(path))
                    elif nb not in visited and nb > start:
                        # Only visit vertices larger than start to avoid duplicates
                        stack.append((nb, path + [nb], visited | {nb}))
        
        return cycle_lengths

    # Quick check: is the graph triangle-free?
    def is_triangle_free(adj, n):
        for u in range(n):
            for i, v in enumerate(adj[u]):
                for w in adj[u][i+1:]:
                    if w in adj[v]:
                        return False
        return True

    # Check min degree
    def min_degree_ok(adj, n):
        return all(len(adj[u]) >= 3 for u in range(n))

    # Try random 3-regular bipartite graphs
    # Part A: 0-5, Part B: 6-11
    # Each vertex in A connects to 3 vertices in B, and vice versa.
    # This is a 3-regular bipartite graph.

    def random_3reg_bipartite():
        """Generate a random 3-regular bipartite graph on 12 vertices."""
        part_a = list(range(6))
        part_b = list(range(6, 12))
        
        # Each vertex in A has degree 3, each in B has degree 3.
        # Total edges = 18.
        # We need to assign 3 edges from each A vertex to B vertices,
        # such that each B vertex gets exactly 3 edges.
        
        # This is equivalent to a 3-regular bipartite graph, which can be
        # generated as a union of 3 perfect matchings.
        
        edges = []
        # Generate 3 disjoint perfect matchings
        for _ in range(3):
            perm = list(part_b)
            random.shuffle(perm)
            for a, b in zip(part_a, perm):
                edges.append((a, b))
        
        # Check for duplicate edges
        edge_set = set()
        for e in edges:
            if e in edge_set:
                return None
            edge_set.add(e)
        
        # Build adjacency
        adj = [[] for _ in range(12)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        return adj, edges

    # Try many random graphs
    max_iterations = 50000
    for it in range(max_iterations):
        if time.time() - start_time > deadline - 5.0:
            break
        
        result = random_3reg_bipartite()
        if result is None:
            continue
        adj, edges = result
        
        # Check triangle-free (bipartite graphs are automatically triangle-free)
        # So no need to check.
        
        # Get cycle lengths
        cycle_lengths = get_cycle_lengths(adj, 12)
        if not cycle_lengths:
            continue
        
        count = len(cycle_lengths)
        if count < best_count:
            best_count = count
            best_cycle_lengths = cycle_lengths
            best_edges = [[u, v] for u, v in edges]
            if count == 1:
                # Can't do better than 1
                break
    
    # If no graph found, fall back to two K_{3,3}
    if best_edges is None:
        edges = []
        part_a1 = [0, 1, 2]
        part_b1 = [3, 4, 5]
        for u in part_a1:
            for v in part_b1:
                edges.append([u, v])
        part_a2 = [6, 7, 8]
        part_b2 = [9, 10, 11]
        for u in part_a2:
            for v in part_b2:
                edges.append([u, v])
        return edges
    
    # Verify the best graph
    adj = [[] for _ in range(12)]
    for u, v in best_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    if not min_degree_ok(adj, 12):
        # Fallback
        edges = []
        part_a1 = [0, 1, 2]
        part_b1 = [3, 4, 5]
        for u in part_a1:
            for v in part_b1:
                edges.append([u, v])
        part_a2 = [6, 7, 8]
        part_b2 = [9, 10, 11]
        for u in part_a2:
            for v in part_b2:
                edges.append([u, v])
        return edges
    
    return best_edges
