import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # For the specific hard case k=3, g=16, use voltage lift search
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None

        # Define base graphs: list of edges for 3-regular graphs
        # We'll generate candidate base graphs and try lifts
        
        def verify_girth(n, edges, girth_target):
            """Fast girth verification using non-backtracking walks.
            Returns True if girth >= girth_target."""
            if n == 0:
                return True
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    return False
            
            # Non-backtracking walk enumeration to find shortest cycle
            # State: (current_node, previous_node)
            # We want to find the shortest closed non-backtracking walk
            # which corresponds to a cycle.
            
            # For each starting edge (u, v), do BFS in the non-backtracking space
            # The state is (current, prev). Start from (v, u).
            # We stop when we return to (u, v) via a different path, or find a cycle.
            
            # Actually, simpler: for each directed edge (u,v), BFS over states (node, prev)
            # starting from (v, u) with distance 1. If we reach (u, w) where w != v and then... 
            # Actually, the standard approach:
            # For each directed edge (u, v), consider it as the start. 
            # We want to find the shortest walk u -> v -> ... -> u that doesn't immediately backtrack.
            # State: (current, previous). Start: (v, u), dist=1.
            # Target: any state (u, w) where w != v? No, we need to close the cycle back to u.
            # If we reach state (u, x), it means we have a walk v -> ... -> u of length L.
            # The cycle is u -> v -> ... -> u. Length = L + 1.
            # Wait, the state (u, x) means we are at u, coming from x. The walk was x -> u.
            # The full walk from the start was u -> v -> ... -> x -> u.
            # Length of this walk is dist[u, x] + 1 (the edge x->u).
            # But we started at (v, u) with dist 1. So dist is the number of edges taken from v.
            # Total cycle length = 1 (u->v) + dist (v->...->x) + 1 (x->u) = dist + 2.
            
            # Let's refine:
            # State: (curr, prev).
            # Start state for base edge (u, v): (v, u), distance 1.
            # From (curr, prev), we can go to (next, curr) for all next in adj[curr] if next != prev.
            # If we reach a state (u, w) where w != v, that means we have a path v -> ... -> w -> u.
            # The cycle is u -> v -> ... -> w -> u.
            # Length = 1 + (distance from v to w) + 1.
            # In our BFS, distance is the number of steps from the start state (v, u).
            # So if we are at state (w, prev_w) with distance d, and w == u, then cycle length = d + 1?
            # Let's trace:
            # Start: (v, u), d=1.
            # Step 1: (v2, v), d=2.
            # ...
            # If we reach (u, x) at distance d, the path is u->v->...->x->u.
            # The number of edges in the path from v to u is d-1? No.
            # State (curr, prev) represents the last edge was prev->curr.
            # Start: last edge u->v. State (v, u). d=1.
            # Next: last edge v->v2. State (v2, v). d=2.
            # If we reach state (u, x) at distance d, the last edge is x->u.
            # The sequence of edges is u->v, v->v2, ..., x->u.
            # Number of edges = d.
            # So cycle length = d.
            
            # We need to check if d < girth_target.
            # We can stop BFS early if d >= girth_target.
            
            for u in range(n):
                for v in adj[u]:
                    # Start BFS from (v, u)
                    dist = {}
                    q = deque()
                    q.append((v, u))
                    dist[(v, u)] = 1
                    
                    while q:
                        curr, prev = q.popleft()
                        d = dist[(curr, prev)]
                        
                        if d >= girth_target:
                            continue
                            
                        for next_node in adj[curr]:
                            if next_node == prev:
                                continue
                            new_state = (next_node, curr)
                            if new_state not in dist:
                                dist[new_state] = d + 1
                                # Check if we closed a cycle
                                if next_node == u:
                                    # Cycle length is d + 1
                                    if d + 1 < girth_target:
                                        return False
                                q.append(new_state)
                            # If new_state was seen, it's a cycle or shorter path, ignore for min girth search
                            # Actually, if we reach (u, x) via a shorter path, we'd have found it earlier.
                            # If we reach (u, x) via a longer path, it's not the shortest cycle.
            
            return True

        def generate_kautz_base(q):
            """Generate the underlying undirected graph of the Kautz digraph K(q, 2).
            The Kautz digraph has q^2 + q vertices? No, K(q,2) has q^2+q vertices?
            Actually, Kautz digraph K(q, 2) has q^2 + q vertices?
            Let's check: K(q, 2) has q^2+q vertices? No.
            Kautz digraph K(q, 2) has q^2+q vertices? 
            Wait, K(q, 2) has q^2 + q vertices? 
            Let's use the standard definition.
            Vertices are pairs (a, b) with a != b, a, b in {0, ..., q-1}.
            Number of vertices = q*(q-1).
            Edges: (a, b) -> (b, c) for c != a.
            Degree q-1.
            For q=4, degree 3. Vertices 12.
            This is a 3-regular directed graph.
            Treating it as undirected, it might have multiple edges or loops?
            (a,b) -> (b,c) and (b,c) -> (c,a)?
            If we treat it as undirected, edge between (a,b) and (b,c).
            Is it simple?
            (a,b) and (b,c) are distinct since a!=b and b!=c.
            Could (a,b)-(b,c) and (a,b)-(b,d) be the same? No, c!=d.
            Could (a,b)-(b,c) and (b,c)-(a,b) be the same? Yes, that's the undirected edge.
            Are there loops? (a,b) -> (b,a)? c=a. But c!=a is required. So no loops.
            So the underlying graph is simple 3-regular.
            """
            vertices = []
            index = {}
            for a in range(q):
                for b in range(q):
                    if a != b:
                        v = (a, b)
                        index[v] = len(vertices)
                        vertices.append(v)
            
            n = len(vertices)
            adj = [[] for _ in range(n)]
            for a in range(q):
                for b in range(q):
                    if a == b:
                        continue
                    u = index[(a, b)]
                    for c in range(q):
                        if c == a:
                            continue
                        v = index[(b, c)]
                        adj[u].append(v)
            
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            return n, edges

        # Generate base graph: Kautz K(4,2) has 12 vertices, 3-regular.
        base_n, base_edges = generate_kautz_base(4)
        
        # Now try voltage lifts of this base graph over Z_m
        # A voltage lift of a graph G with voltages in Z_m creates m copies of each vertex.
        # Edge (u, v) with voltage w becomes edges (u_i, v_{i+w}) for all i.
        # We need to assign voltages to each edge in the base graph.
        # To maximize girth, we want to avoid short closed walks.
        
        # For a 3-regular graph, each vertex has 3 edges.
        # We can assign voltages systematically.
        # Let's try assigning voltages based on a primitive root or simple patterns.
        
        # Strategy: For each edge in the base graph, assign a voltage.
        # We'll iterate through possible voltage assignments for small m.
        # Since the base graph has 18 edges (12*3/2), the search space is m^18, which is too large.
        # Instead, we can use a deterministic scheme.
        
        # Alternative: Use the fact that the Kautz graph is an expander.
        # Assign voltages randomly or using a hash function, and check girth.
        # But we need a deterministic approach to be safe.
        
        # Let's try a specific voltage assignment:
        # For each edge (u, v), assign voltage = (u * v) % m or something similar.
        # Or, use the structure of the Kautz graph.
        
        # Actually, let's try a simpler base: the Petersen graph (10 vertices, 3-regular).
        # It's well-known and small.
        
        def generate_petersen():
            # Outer pentagon: 0-1-2-3-4-0
            # Inner pentagram: 5-7-9-6-8-5
            # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
            ]
            return 10, edges

        # Try Petersen base with voltage lifts
        petersen_n, petersen_edges = generate_petersen()
        
        # For each m, assign voltages to the 15 edges of Petersen.
        # We'll use a deterministic scheme: voltage of edge i = i % m.
        # This is a rough heuristic, but let's see.
        
        for m in range(2, 100):
            if time.time() > deadline:
                break
            
            # Assign voltages
            voltages = {}
            for i, (u, v) in enumerate(petersen_edges):
                voltages[(u, v)] = i % m
                voltages[(v, u)] = -voltages[(u, v)] % m
            
            # Build lifted graph
            lifted_n = petersen_n * m
            lifted_edges = []
            for u in range(petersen_n):
                for v, w in [(v, voltages[(u, v)]) for (u2, v) in petersen_edges if u2 == u]:
                    for i in range(m):
                        lifted_edges.append((u * m + i, v * m + (i + w) % m))
            
            # Verify girth
            if verify_girth(lifted_n, lifted_edges, 16):
                if lifted_n < best_n:
                    best_n = lifted_n
                    best_edges = lifted_edges
                    # We can break early if we find a very good one, but let's continue to find smaller m
                    # Actually, smaller m gives smaller n, so we should check smaller m first.
                    # But m=2,3,4... might not work. We need to find the smallest m that works.
        
        # Also try Kautz base
        if time.time() < deadline:
            for m in range(2, 50):
                if time.time() > deadline:
                    break
                
                # Assign voltages to Kautz edges
                voltages = {}
                for i, (u, v) in enumerate(base_edges):
                    voltages[(u, v)] = i % m
                    voltages[(v, u)] = -voltages[(u, v)] % m
                
                lifted_n = base_n * m
                lifted_edges = []
                for u in range(base_n):
                    for v, w in [(v, voltages[(u, v)]) for (u2, v) in base_edges if u2 == u]:
                        for i in range(m):
                            lifted_edges.append((u * m + i, v * m + (i + w) % m))
                
                if verify_girth(lifted_n, lifted_edges, 16):
                    if lifted_n < best_n:
                        best_n = lifted_n
                        best_edges = lifted_edges
        
        if best_edges is not None:
            return best_n, best_edges

    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    return luw_construct(k, g)
