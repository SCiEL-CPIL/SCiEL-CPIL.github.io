import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def check_c4_fast(adj_bits, n, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        nu = adj_bits[u]
        nv = adj_bits[v]
        # Iterate over neighbors of u
        nu_iter = nu
        while nu_iter:
            lsb = nu_iter & (-nu_iter)
            x = lsb.bit_length() - 1
            nu_iter ^= lsb
            # Check if x is connected to any neighbor of v
            # We don't need to check if x == v because u-v is not an edge yet
            # But we must ensure x != v is handled? 
            # adj_bits[v] contains v's neighbors. If x is in adj_bits[v], 
            # then u-x-v-y-u is a C4 if y is in adj_bits[x] and y != u, v.
            # Actually, if x is neighbor of u and x is neighbor of v, 
            # and they share another neighbor, it's a C4.
            # Wait. If x is neighbor of u and x is neighbor of v, 
            # then u-x-v is a path of length 2.
            # A C4 is u-a-v-b-u.
            # If we add u-v, a C4 is formed if there exists a,b 
            # such that u-a, a-v, v-b, b-u are edges.
            # This is equivalent to: u and v have at least 2 common neighbors.
            # So we just need to count common neighbors of u and v.
            # If count >= 2, adding u-v creates a C4.
            pass
        # Optimization: Common neighbors count
        common = adj_bits[u] & adj_bits[v]
        return common.bit_count() >= 2

    def is_c4_free(adj_bits, n):
        """Verify no C4 exists."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bits[i] & (1 << j)):
                    common = adj_bits[i] & adj_bits[j]
                    if common.bit_count() >= 2:
                        return False
                else:
                    # If edge exists, it's not a "missing" edge, 
                    # but C4 definition involves 4 distinct vertices.
                    # If i and j are connected, they can still be part of a C4 
                    # with other vertices. 
                    # My check above only looks for "induced" C4s or 
                    # C4s where i,j are non-adjacent?
                    # No, a C4 is a cycle of length 4.
                    # u-v-w-x-u. 
                    # u and w are not connected in the cycle, but might be in the graph.
                    # The condition for a C4-free graph is that NO 4 vertices 
                    # form a cycle.
                    # This is equivalent to: For any pair of non-adjacent vertices, 
                    # they have at most 1 common neighbor? 
                    # No. If u and v are adjacent, they can have common neighbors.
                    # Example: Diamond graph. u-v, v-w, w-x, x-u. 
                    # If u-w is also an edge, it's not a C4 (it's a triangle + edge).
                    # But if u-w is NOT an edge, it is a C4.
                    # So, a graph is C4-free iff for every pair of vertices (u, v),
                    # the number of common neighbors is at most 1 IF u and v are NOT adjacent?
                    # Actually, if u and v ARE adjacent, they can have many common neighbors 
                    # (forming triangles). 
                    # The C4 is u-a-v-b-u. 
                    # Here u and v are NOT adjacent in the cycle. 
                    # So, for a C4 to exist, there must be two vertices u, v (not necessarily non-adjacent in G, 
                    # but distinct in the cycle) such that they have two common neighbors a, b.
                    # Wait. If u and v are adjacent in G, and they share neighbors a and b,
                    # then u-a-v-b-u is a C4 ONLY IF u-v is not part of the cycle?
                    # The cycle is u-a-v-b-u. The edges are (u,a), (a,v), (v,b), (b,u).
                    # The edge (u,v) is a chord. 
                    # A graph is C4-free if it contains NO induced C4? 
                    # No, "containing no C4" usually means no subgraph isomorphic to C4.
                    # If there is a chord, it's not a subgraph isomorphic to C4 (it has an extra edge).
                    # So, if u and v are adjacent, and share a, b, then {u,v,a,b} induces a K4 minus one edge? 
                    # Or K4? 
                    # If u-v exists, and u-a, a-v, v-b, b-u exist, 
                    # the subgraph on {u,v,a,b} has edges (u,v), (u,a), (a,v), (v,b), (b,u).
                    # This contains a triangle u-a-v and triangle u-b-v.
                    # Does it contain a C4? 
                    # A C4 requires 4 edges. 
                    # The cycle u-a-v-b-u uses 4 edges. 
                    # Even if u-v is present, the cycle u-a-v-b-u exists as a subgraph?
                    # Yes, the edges (u,a), (a,v), (v,b), (b,u) are all present.
                    # So it IS a C4 subgraph.
                    # Therefore, ANY pair of vertices (adjacent or not) having 2 common neighbors 
                    # creates a C4 subgraph.
                    # So the condition is: For ALL pairs (u, v), common_neighbors(u, v) <= 1.
                    pass
        
        # Re-evaluating is_c4_free
        for i in range(n):
            for j in range(i + 1, n):
                common = adj_bits[i] & adj_bits[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def min_degree(adj_bits, n):
        return min(adj_bits[i].bit_count() for i in range(n))

    def to_edge_list(edges_set):
        return [[a, b] for a, b in edges_set]

    def greedy_build(n, min_deg_req, time_limit, seed):
        t0 = time.time()
        rng = random.Random(seed)
        
        adj_bits = [0] * n
        edges_set = set()

        def add_edge(u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            a, b = min(u, v), max(u, v)
            edges_set.add((a, b))

        def remove_edge(u, v):
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            a, b = min(u, v), max(u, v)
            edges_set.discard((a, b))

        def can_add_edge(u, v):
            if u == v:
                return False
            if adj_bits[u] & (1 << v):
                return False
            # Check C4 condition: common neighbors < 2
            common = adj_bits[u] & adj_bits[v]
            return common.bit_count() < 2

        # Strategy: Randomized Greedy with Repair
        # 1. Randomly add edges until we hit a wall or time limit
        # 2. If min degree is low, focus on low degree vertices
        # 3. If stuck, remove "blocking" edges
        
        # Initial random fill
        # Target edges approx n * min_deg_req / 2
        target_edges = int(n * min_deg_req / 2) + 5
        attempts = 0
        while len(edges_set) < target_edges and time.time() - t0 < time_limit / 2:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v and can_add_edge(u, v):
                add_edge(u, v)
            attempts += 1
            if attempts > 10000:
                break

        # Refinement loop
        stuck_count = 0
        while time.time() - t0 < time_limit:
            md = min_degree(adj_bits, n)
            if md >= min_deg_req:
                if is_c4_free(adj_bits, n):
                    return {'n': n, 'edges': to_edge_list(edges_set)}
                return None

            # Find lowest degree vertex
            low_v = min(range(n), key=lambda i: adj_bits[i].bit_count())
            low_deg = adj_bits[low_v].bit_count()

            if low_deg < min_deg_req:
                # Try to add an edge to low_v
                candidates = [v for v in range(n) if v != low_v and not (adj_bits[low_v] & (1 << v))]
                rng.shuffle(candidates)
                
                added = False
                for v in candidates:
                    if can_add_edge(low_v, v):
                        add_edge(low_v, v)
                        added = True
                        stuck_count = 0
                        break

                if not added:
                    stuck_count += 1
                    if stuck_count > 50:
                        # Repair: Remove an edge that prevents low_v from connecting
                        # Find a neighbor x of low_v such that removing (low_v, x) 
                        # allows low_v to connect to a non-neighbor y?
                        # Or remove an edge (a, b) where a is neighbor of low_v 
                        # and b is non-neighbor of low_v, and (a,b) creates C4 risk?
                        
                        # Simple repair: Remove a random edge not incident to low_v 
                        # that is "high impact" (part of many potential C4s)
                        # For speed, just remove a random edge from a random pair 
                        # where both endpoints have degree > min_deg_req
                        safe_edges = [(a, b) for a, b in edges_set 
                                      if a != low_v and b != low_v 
                                      and adj_bits[a].bit_count() > min_deg_req 
                                      and adj_bits[b].bit_count() > min_deg_req]
                        if safe_edges:
                            u, w = rng.choice(safe_edges)
                            remove_edge(u, w)
                            stuck_count = 0
                        else:
                            # No safe edges, try removing from low_v neighbors
                            neighbors = [i for i in range(n) if adj_bits[low_v] & (1 << i)]
                            if neighbors:
                                # Remove one that is most "restrictive"? 
                                # Just pick random
                                u = rng.choice(neighbors)
                                if adj_bits[u].bit_count() > min_deg_req:
                                    remove_edge(low_v, u)
                                    stuck_count = 0
                            else:
                                break # No edges to remove?
            else:
                stuck_count = 0
        return None

    best_result = None
    best_n = 0

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline - 3:
            break

        min_deg_req = max(1, n - 39)
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break

        # Allocate time
        num_attempts = min(15, max(3, int(remaining / 3.0)))
        time_per_attempt = (remaining - 2) / max(1, num_attempts)

        for attempt in range(num_attempts):
            if time.time() - start_time > deadline - 2:
                break

            seed = 42 + attempt * 104729 + n * 7919
            result = greedy_build(n, min_deg_req, time_per_attempt, seed)
            if result is not None:
                best_n = n
                best_result = result
                return result

    if best_result is not None:
        return best_result

    # Fallback
    return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
