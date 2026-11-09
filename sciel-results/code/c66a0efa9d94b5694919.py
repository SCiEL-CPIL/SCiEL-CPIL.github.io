import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Use bitmask representation for speed
    # adj[i] is a bitmask of neighbors of vertex i
    # has_edge[i][j] boolean for edge existence (i < j)

    def count_bits(x):
        return bin(x).count('1')

    def is_k33_free_bitmask(adj):
        """Check if graph with bitmask adjacency is K3,3-free."""
        for a1 in range(n):
            for a2 in range(a1 + 1, n):
                common_12 = adj[a1] & adj[a2]
                if count_bits(common_12) < 2:
                    continue
                for a3 in range(a2 + 1, n):
                    common_123 = common_12 & adj[a3]
                    if count_bits(common_123) >= 3:
                        return False
        return True

    def find_k33_bitmask(adj):
        """Find a K3,3 subgraph. Returns (a1,a2,a3,b1,b2,b3) or None."""
        for a1 in range(n):
            for a2 in range(a1 + 1, n):
                common_12 = adj[a1] & adj[a2]
                if count_bits(common_12) < 2:
                    continue
                for a3 in range(a2 + 1, n):
                    common_123 = common_12 & adj[a3]
                    if count_bits(common_123) >= 3:
                        # Get 3 common neighbors
                        bits = []
                        x = common_123
                        while len(bits) < 3 and x:
                            lb = x & (-x)
                            bits.append(lb.bit_length() - 1)
                            x ^= lb
                        b1, b2, b3 = bits
                        return (a1, a2, a3, b1, b2, b3)
        return None

    def edges_to_bitmask(edges):
        """Convert edge list to bitmask adjacency."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bitmask_to_edges(adj):
        """Convert bitmask adjacency to edge list."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
        return edges

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        # After adding (u,v), check all triples that include u or v
        # A new K3,3 would have u or v in one part
        # Case 1: u is in part A, need v in part B with 2 other common neighbors of u,v
        # Actually, the new K3,3 must use the new edge (u,v).
        # So either {u,v} are in opposite parts, or... 
        # In K3,3, each edge connects opposite parts.
        # So if we add (u,v), the K3,3 uses this edge with u in A, v in B (or vice versa).
        # Then we need 2 more vertices in A (neighbors of v, not u) and 2 more in B (neighbors of u, not v)
        # such that all cross edges exist.
        
        # More precisely: u in A, v in B. Need a2,a3 in A\{u} that are neighbors of v,
        # and b2,b3 in B\{v} that are neighbors of u,
        # such that a2,a3 are both neighbors of b2,b3 (and b2,b3 neighbors of a2,a3).
        # Wait, in K3,3 all 9 cross edges must exist.
        # a2 must be neighbor of v (already), and neighbor of b2,b3.
        # a3 must be neighbor of v (already), and neighbor of b2,b3.
        # b2 must be neighbor of u (already), and neighbor of a2,a3.
        # b3 must be neighbor of u (already), and neighbor of a2,a3.
        
        # So: candidates_A = neighbors of v (excluding u) 
        #     candidates_B = neighbors of u (excluding v)
        # Need to find a2,a3 in candidates_A and b2,b3 in candidates_B
        # such that a2,a3 are both adjacent to b2 and b3.
        
        cand_a = adj[v] & ~(1 << u)
        cand_b = adj[u] & ~(1 << v)
        
        if count_bits(cand_a) < 2 or count_bits(cand_b) < 2:
            return True
        
        # For each pair in cand_a, check if they share >= 2 common neighbors in cand_b
        ca_list = []
        x = cand_a
        while x:
            lb = x & (-x)
            ca_list.append(lb.bit_length() - 1)
            x ^= lb
        
        cb_list = []
        x = cand_b
        while x:
            lb = x & (-x)
            cb_list.append(lb.bit_length() - 1)
            x ^= lb
        
        for i in range(len(ca_list)):
            a2 = ca_list[i]
            for j in range(i + 1, len(ca_list)):
                a3 = ca_list[j]
                # Common neighbors of a2 and a3 that are in cand_b
                common = adj[a2] & adj[a3] & cand_b
                if count_bits(common) >= 2:
                    return False
        return True

    def greedy_construction():
        """Greedy construction with random edge ordering."""
        adj = [0] * n
        edges = []
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        return edges

    def local_search(edges, max_iter=5000):
        """Local search: try adding edges, removing edges from K3,3 violations."""
        adj = edges_to_bitmask(edges)
        edge_set = set(edges)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add a random missing edge
            added = False
            for _ in range(50):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append((u, v))
                    added = True
                    break
            
            if not added:
                # Try removing edges to enable additions
                # Find a K3,3 and remove an edge
                k33 = find_k33_bitmask(adj)
                if k33:
                    a1, a2, a3, b1, b2, b3 = k33
                    # Get all 9 edges
                    k33_edges = [
                        (a1, b1), (a1, b2), (a1, b3),
                        (a2, b1), (a2, b2), (a2, b3),
                        (a3, b1), (a3, b2), (a3, b3)
                    ]
                    # Normalize
                    k33_edges = [tuple(sorted(e)) for e in k33_edges]
                    # Remove a random edge
                    ue, ve = random.choice(k33_edges)
                    if (ue, ve) in edge_set:
                        edge_set.remove((ue, ve))
                        edges.remove((ue, ve))
                        adj[ue] &= ~(1 << ve)
                        adj[ve] &= ~(1 << ue)
        
        return edges

    def polarity_graph_seed():
        """Construct a polarity graph seed.
        Use a known construction: take vertices as elements of a finite field or use
        a quadratic form over a small field.
        
        For n=20, we can use a graph based on PG(2,4) or similar.
        Alternatively, use a random dense graph and fix violations.
        
        Let's use a construction based on the Erdos-Renyi random graph with p~0.5
        and then fix K3,3 violations by removing edges.
        """
        adj = [0] * n
        edges = []
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        # Start with p=0.6
        for u, v in all_edges:
            if random.random() < 0.6:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        # Fix K3,3 violations
        for _ in range(200):
            k33 = find_k33_bitmask(adj)
            if not k33:
                break
            a1, a2, a3, b1, b2, b3 = k33
            k33_edges = [
                (a1, b1), (a1, b2), (a1, b3),
                (a2, b1), (a2, b2), (a2, b3),
                (a3, b1), (a3, b2), (a3, b3)
            ]
            k33_edges = [tuple(sorted(e)) for e in k33_edges]
            ue, ve = random.choice(k33_edges)
            if (ue, ve) in [tuple(sorted(e)) for e in edges]:
                edges.remove((ue, ve))
                adj[ue] &= ~(1 << ve)
                adj[ve] &= ~(1 << ue)
        
        return edges

    # Initialize with multiple strategies
    best_edges = []
    best_count = 0
    
    # Start with polarity graph seed
    seed_edges = polarity_graph_seed()
    best_count = len(seed_edges)
    best_edges = list(seed_edges)
    
    # Greedy construction
    greedy_edges = greedy_construction()
    if len(greedy_edges) > best_count:
        best_count = len(greedy_edges)
        best_edges = list(greedy_edges)
    
    # Local search from best
    improved = local_search(list(best_edges), max_iter=3000)
    if len(improved) > best_count:
        best_count = len(improved)
        best_edges = list(improved)
    
    # More local search iterations
    while time.time() < deadline:
        # Try local search from current best
        improved = local_search(list(best_edges), max_iter=2000)
        if len(improved) > best_count:
            best_count = len(improved)
            best_edges = list(improved)
        
        # Occasionally try fresh greedy
        if time.time() < deadline - 10:
            greedy_edges = greedy_construction()
            if len(greedy_edges) > best_count:
                best_count = len(greedy_edges)
                best_edges = list(greedy_edges)
                improved = local_search(list(best_edges), max_iter=1000)
                if len(improved) > best_count:
                    best_count = len(improved)
                    best_edges = list(improved)
    
    return [[u, v] for u, v in best_edges]
