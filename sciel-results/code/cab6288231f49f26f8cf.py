"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        if not edge_set:
            return set()
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        queue = deque([v for v in deg if deg[v] < 2])
        in_core = set(deg.keys())
        while queue:
            v = queue.popleft()
            if v not in in_core:
                continue
            in_core.discard(v)
            for nb in adj[v]:
                if nb in in_core:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        queue.append(nb)
        core_edges = set()
        for u, v in edge_set:
            if u in in_core and v in in_core:
                core_edges.add(tuple(sorted((u, v))))
        return core_edges

    def find_cycle_bfs(adj, start=None):
        """Find any cycle in the graph defined by adj. Returns list of vertices or None."""
        vertices = set()
        for u in adj:
            for v in adj[u]:
                vertices.add(u)
                vertices.add(v)
        
        starts = [start] if start else list(vertices)
        random.shuffle(starts)
        
        for s in starts:
            if len(adj.get(s, set())) < 2:
                continue
            visited = {s: None}
            parent = {}
            queue = deque([s])
            while queue:
                curr = queue.popleft()
                for nb in adj[curr]:
                    if nb == s and curr != s:
                        # Found cycle: reconstruct path
                        path = [s]
                        node = curr
                        while node != s:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        if len(path) >= 3:
                            return path
                    elif nb not in visited:
                        visited[nb] = True
                        parent[nb] = curr
                        queue.append(nb)
        return None

    def initial_decompose(edge_set, n):
        """Create initial decomposition: 2-core cycles + tree edges."""
        pieces = []
        rem = set(edge_set)
        
        core_edges = get_2_core(rem)
        forest_edges = rem - core_edges
        
        for e in forest_edges:
            pieces.append(list(e))
        
        rem = core_edges
        while rem:
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            
            cycle = find_cycle_bfs(adj)
            if not cycle:
                # No more cycles, add remaining as edges
                for e in rem:
                    pieces.append(list(e))
                break
            
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                rem.discard(e)
            pieces.append(cycle[:])
        
        return pieces

    def local_search(pieces, edge_set, n, time_limit):
        """
        Local search to minimize piece count.
        State: list of pieces (each piece is a list of vertices for cycle, or [u,v] for edge)
        Moves:
        1. Merge two adjacent single edges into a path (if they share a vertex) - doesn't help directly
        2. Convert a path of 2+ edges into a cycle if endpoints are connected
        3. Split a cycle into two smaller cycles if there's a chord
        4. Move an edge from a cycle to create a smaller cycle + path, then close path
        
        Actually, simpler effective moves:
        - For each pair of pieces, check if we can recombine to reduce count
        - Key move: if we have two edges [a,b] and [b,c] and edge [a,c] exists unused, 
          we can replace the two edges with cycle [a,b,c] (saves 1 piece)
        - Key move: if we have a cycle and there's a chord, split into two cycles (same count, 
          but may enable further merges)
        - Key move: if we have a path (from splitting) and can close it into a cycle
        
        Let me think differently. The objective is to minimize #pieces.
        Each cycle of length k uses k edges and counts as 1 piece.
        Each single edge uses 1 edge and counts as 1 piece.
        So we want to maximize the number of edges that are in cycles.
        
        Better approach: think of it as we want to find a maximum set of edge-disjoint cycles.
        The number of pieces = (#edges not in any cycle) + (#cycles)
        = total_edges - (#edges in cycles) + (#cycles)
        
        So we want to maximize (#edges in cycles) - (#cycles)
        = sum over cycles of (length - 1)
        
        This is equivalent to maximizing the total "savings" where a cycle of length k saves k-1 pieces.
        
        Local search moves:
        1. Find a chord in a cycle: split into two cycles. Savings change:
           Original: 1 cycle of length k, saving = k-1
           New: two cycles of lengths k1, k2 where k1+k2 = k+2 (chord counted twice)
           Wait, chord is one edge. Original cycle has k edges. Split uses chord + k edges = k+1 edges total?
           No. Original cycle: vertices v1,...,vk, edges (v1,v2),...,(vk,v1). k edges.
           Chord: (vi,vj). Split into cycle1: v1..vi,vj..vk,v1 (length = (i-1) + (k-j+1) + 1 = k - j + i)
           and cycle2: vi,vj,vj+1,...,vi (length = j - i + 1)
           Total edges in both cycles: (k-j+i) + (j-i+1) = k+1. But we only have k edges + 1 chord = k+1 edges available.
           So we use the chord. New saving: (k-j+i-1) + (j-i+1-1) = k-2.
           Original saving: k-1.
           Change: -1. So splitting a cycle via chord DECREASES savings by 1. Bad move!
           
        2. Merge two single edges into a cycle: if we have edges [a,b] and [b,c] and edge [a,c] is unused,
           replace with cycle [a,b,c]. 
           Before: 2 pieces, 2 edges. After: 1 piece, 3 edges.
           Savings before: 0 (two single edges). After: 2 (cycle of length 3 saves 2).
           Change: +2. Great move!
           
        3. More generally: if we have a path of m edges and the endpoints are connected by an unused edge,
           we can close it into a cycle of length m+1.
           Before: m pieces (if all single edges) or fewer if some were in cycles.
           This gets complex.
        
        Let me implement a simpler but effective local search:
        - Maintain the set of edges in cycles and the cycle structure
        - Repeatedly try to find "improving moves":
          a) Find two unused edges that share a vertex and whose other endpoints are connected by an unused edge
             -> form a triangle (cycle of length 3), gain +2 savings
          b) Find a path of unused edges of length m where endpoints are connected -> form cycle, gain +m savings
          c) Find a chord in a cycle that allows splitting into two cycles where one is a triangle
             -> this might not help directly
        
        Actually, let me just do: repeatedly find the longest cycle in the remaining unused edges.
        But that's what the original does. The issue is the DFS is too slow.
        
        New plan: Use a faster cycle-finding algorithm and more restarts.
        """
        if not pieces:
            return pieces
        
        # Reconstruct which edges are in which piece
        # Piece types: cycle (len >= 3) or edge (len == 2)
        
        best_pieces = [p[:] for p in pieces]
        best_count = len(pieces)
        
        # Build adjacency and track which edges are "free" (single edges)
        # and which are in cycles
        
        for iteration in range(200):
            if time.time() > time_limit:
                break
            
            # Try to improve: find three free edges that form a triangle
            free_edges = set()
            for p in best_pieces:
                if len(p) == 2:
                    free_edges.add(tuple(sorted(p)))
            
            if len(free_edges) < 3:
                break
            
            # Find triangles in free edges
            adj_free = defaultdict(set)
            for u, v in free_edges:
                adj_free[u].add(v)
                adj_free[v].add(u)
            
            improved = False
            for u in list(adj_free.keys()):
                if time.time() > time_limit:
                    break
                for v in adj_free[u]:
                    for w in adj_free[v]:
                        if w == u:
                            continue
                        if (min(u,w), max(u,w)) in free_edges and (min(v,w), max(v,w)) in free_edges:
                            # Triangle u-v-w
                            e1 = tuple(sorted((u, v)))
                            e2 = tuple(sorted((v, w)))
                            e3 = tuple(sorted((u, w)))
                            if e1 in free_edges and e2 in free_edges and e3 in free_edges:
                                # Replace the three single edges with a cycle
                                new_pieces = []
                                for p in best_pieces:
                                    if len(p) == 2:
                                        pe = tuple(sorted(p))
                                        if pe in (e1, e2, e3):
                                            continue
                                    new_pieces.append(p[:])
                                new_pieces.append([u, v, w])
                                if len(new_pieces) < best_count:
                                    best_pieces = new_pieces
                                    best_count = len(new_pieces)
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try to extend: find a free edge that can be added to form a cycle
                # or find a path of free edges that can be closed
                # For now, try random restarts of cycle finding on free edges
                if free_edges:
                    adj = defaultdict(set)
                    for u, v in free_edges:
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    # Try to find a cycle in free edges
                    cycle = find_cycle_bfs(adj)
                    if cycle and len(cycle) >= 4:  # Only useful if length >= 4 (triangle already handled)
                        # Remove the cycle edges from free_edges and add the cycle
                        cycle_edges = set()
                        for i in range(len(cycle)):
                            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                            cycle_edges.add(tuple(sorted((u, v))))
                        
                        new_pieces = []
                        for p in best_pieces:
                            if len(p) == 2:
                                pe = tuple(sorted(p))
                                if pe in cycle_edges:
                                    continue
                            new_pieces.append(p[:])
                        new_pieces.append(cycle[:])
                        if len(new_pieces) < best_count:
                            best_pieces = new_pieces
                            best_count = len(new_pieces)
                            improved = True
                
                if not improved:
                    break
        
        return best_pieces

    def decompose(edges, n, time_budget):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        best_pieces = None
        best_count = len(edge_set) + 1
        
        # Get initial decomposition
        init_pieces = initial_decompose(edge_set, n)
        best_count = len(init_pieces)
        best_pieces = [p[:] for p in init_pieces]
        
        # Local search
        remaining_time = time_budget - (time.time() - (deadline - time_budget))
        if remaining_time > 5:
            improved = local_search(best_pieces, edge_set, n, deadline - 5)
            if len(improved) < best_count:
                best_pieces = improved
                best_count = len(best_pieces)
        
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
