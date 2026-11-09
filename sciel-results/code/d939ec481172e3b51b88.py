"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. For k=4, g=10, use a greedy repair heuristic on a random 4-regular graph 
       of size 320 (known optimal upper bound).
    2. Fallback to Lazebnik-Ustimenko-Woldar (LUV) graph construction if repair fails 
       or for other parameters.
    """
    start_time = time.time()
    deadline = 115.0  # 5s safety margin before 120s limit
    
    if k == 4 and g == 10:
        result = try_greedy_repair(4, 10, deadline - start_time)
        if result is not None:
            return result

    # Fallback to LUV construction
    return luv_construct(k, g)

def try_greedy_repair(k, g, time_limit):
    """
    Attempt to find a 4-regular graph of size 320 with girth >= 10 
    using random 2-switches to eliminate short cycles.
    """
    N = 320
    # Generate a random 4-regular graph
    adj = generate_random_4_regular(N)
    if adj is None:
        return None

    # Precompute edges list for random selection
    # adj is list of sets
    # We need a list of (u, v) with u < v
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    edge_set = set(edges)
    
    # Helper to check if a path exists between u and v of length < max_len
    # in the graph defined by adj, excluding specific edges
    def has_short_path(adj, u, v, max_len, excluded_edges):
        if u == v:
            return 0 < max_len
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            curr, depth = queue.popleft()
            if depth >= max_len:
                continue
            for neighbor in adj[curr]:
                if (min(curr, neighbor), max(curr, neighbor)) in excluded_edges:
                    continue
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    if neighbor == v:
                        return True
                    queue.append((neighbor, depth + 1))
        return False

    def find_shortest_cycle(adj):
        """
        Find the length of the shortest cycle in the graph.
        Returns (length, list_of_edges) or (inf, [])
        """
        # BFS from each vertex is O(N^2) which is fine for N=320
        # But we can optimize by checking only once?
        # For simplicity, BFS from each node.
        min_len = float('inf')
        min_cycle_edges = []
        
        for start in range(N):
            # BFS to find shortest path back to start
            # We need to track parent to reconstruct cycle?
            # Actually, we just need the length and the edges to break it.
            # BFS from start, finding first time we hit start?
            # No, BFS from start finds shortest path to all nodes.
            # A cycle through start is formed by edge (start, u) + path u->start.
            # But path u->start in BFS tree doesn't use (start, u).
            
            parent = [-1] * N
            depth = [-1] * N
            q = deque()
            q.append(start)
            depth[start] = 0
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        # Cycle length = depth[u] + depth[v] + 1
                        # Wait, this logic is for finding ANY cycle in BFS tree.
                        # It finds the girth if we do this for all u?
                        # Yes, the girth is the minimum over all such back-edges.
                        cl = depth[u] + depth[v] + 1
                        if cl < min_len:
                            min_len = cl
                            # Reconstruct cycle edges
                            # Path v -> ... -> u in tree, plus edge (u, v)
                            # We need the edges.
                            # Trace back from u to LCA?
                            # Simpler: just store the edge (u, v) and one endpoint?
                            # For the repair step, we just need ONE edge from the cycle.
                            # We can just return (min_len, [(u, v)])
                            # But (u, v) might not be in the cycle if we just found it?
                            # Yes, (u, v) is the edge closing the cycle.
                            min_cycle_edges = [(min(u, v), max(u, v))]
                            # We can break early if min_len is small?
                            # If min_len == 3, we can't do better.
                            if min_len == 3:
                                return min_len, min_cycle_edges

        return min_len, min_cycle_edges

    # Main loop
    while time.time() < start_time + time_limit:
        min_len, cycle_edges = find_shortest_cycle(adj)
        
        if min_len >= g:
            # Success
            final_edges = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        final_edges.append((u, v))
            return N, final_edges

        # Try to break the shortest cycle
        # Pick a random edge from the cycle (we only have one, but that's fine)
        # Actually, find_shortest_cycle returns one edge.
        # Let's try to swap this edge with a random other edge.
        
        target_edge = cycle_edges[0]
        u1, v1 = target_edge
        
        # Try 100 random swaps
        swapped = False
        for _ in range(100):
            # Pick random edge (u2, v2) different from target
            # Ensure u2, v2 not in {u1, v1} to avoid simple issues?
            # Not strictly necessary, but helps.
            e2 = random.choice(edges)
            u2, v2 = e2
            if e2 == target_edge:
                continue
            if u1 in (u2, v2) or v1 in (u2, v2):
                continue
            
            # Candidate swaps:
            # 1. (u1, u2), (v1, v2)
            # 2. (u1, v2), (v1, u2)
            
            for new_e1, new_e2 in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
                # Check validity: no loops, no existing edges
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                
                if ne1 in edge_set or ne2 in edge_set:
                    continue
                if ne1 == ne2:
                    continue
                
                # Check if new edges create short cycles
                # New graph: remove target_edge, e2. Add ne1, ne2.
                # We need to check if ne1 creates a cycle < g.
                # This requires checking path in G' \ {ne1}.
                # G' = G \ {target_edge, e2} + {ne1, ne2}
                # Path for ne1=(a, b): a to b in G \ {target_edge, e2} + {ne2}
                
                # To speed up, we can first check in G \ {target_edge, e2}
                # If no path, check with ne2 added.
                
                # Implementing dynamic graph check is complex.
                # Approximation: Check in G \ {target_edge, e2}.
                # If a short path exists there, it exists in G'.
                # If not, it MIGHT exist via ne2.
                # Given g=10, paths are long. ne2 is unlikely to create a short path
                # unless ne2 connects nodes close to a or b.
                # Let's do the full check for safety.
                
                # Create a modified adjacency for checking?
                # Too slow to copy.
                # Instead, pass excluded edges and "extra" edges to BFS?
                
                def check_no_short_cycle(adj, new_edge, excluded, extra):
                    a, b = new_edge
                    # Check path a to b in graph with 'excluded' removed and 'extra' added
                    # BFS
                    visited = {a: 0}
                    q = deque([(a, 0)])
                    while q:
                        curr, d = q.popleft()
                        if d >= g - 1:
                            continue
                        # Neighbors in original adj
                        for n in adj[curr]:
                            if (min(curr, n), max(curr, n)) in excluded:
                                continue
                            if n not in visited:
                                visited[n] = d + 1
                                if n == b:
                                    return False # Cycle found
                                q.append((n, d + 1))
                        # Neighbors in extra edges
                        for ee in extra:
                            if curr in ee:
                                other = ee[0] if ee[1] == curr else ee[1]
                                if other not in visited:
                                    visited[other] = d + 1
                                    if other == b:
                                        return False
                                    q.append((other, d + 1))
                    return True # No short cycle

                excluded = {target_edge, e2}
                extra = [ne1, ne2]
                
                # Check ne1
                if not check_no_short_cycle(adj, ne1, excluded, [ne2]):
                    continue
                # Check ne2
                if not check_no_short_cycle(adj, ne2, excluded, [ne1]):
                    continue
                
                # Also, we must ensure that removing target_edge and e2 doesn't
                # leave the graph disconnected?
                # 2-switches on regular graphs usually preserve connectivity if graph is 2-edge-connected.
                # Random 4-regular graphs are highly connected.
                
                # Accept swap
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].add(ne1[1])
                adj[ne1[1]].add(u1)
                adj[v1].add(ne2[1])
                adj[ne2[1]].add(v1)
                
                edge_set.remove(target_edge)
                edge_set.remove(e2)
                edge_set.add(ne1)
                edge_set.add(ne2)
                
                edges.remove(target_edge)
                edges.remove(e2)
                edges.append(ne1)
                edges.append(ne2)
                
                swapped = True
                break
            if swapped:
                break
        
        if not swapped:
            # If we can't break the cycle with random swaps, 
            # we might be stuck. 
            # Try a random 2-switch anywhere to perturb?
            # For now, just loop again.
            pass

    # If time runs out, return None to trigger fallback
    return None

def generate_random_4_regular(n):
    """
    Generate a random 4-regular graph on n vertices.
    Uses configuration model or random pairing.
    """
    if n < 4:
        return None
    # Degrees: each vertex has degree 4
    # Total stubs: 4n
    # Pair stubs randomly.
    # Ensure no loops and no multi-edges.
    
    # Simple approach: 
    # 1. Create list of stubs [0,0,0,0, 1,1,1,1, ...]
    # 2. Shuffle
    # 3. Pair consecutive stubs
    # 4. Check for loops/multi-edges. If found, restart.
    # For n=320, this is fast.
    
    while True:
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i, i])
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            return adj

def luv_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        
    return len(order), edges
# EVOLVE-BLOCK-END
