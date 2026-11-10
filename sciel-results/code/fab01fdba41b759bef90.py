# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    # Replicate the verifier's instance generator exactly.
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        remaining = set(edges)
        pieces = []
        
        # Find bridges in a set of edges
        def find_bridges(edge_set):
            if not edge_set:
                return set()
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            visited = set()
            bridges = set()
            time_counter = [0]
            disc = {}
            low = {}
            
            def dfs(u, parent):
                disc[u] = time_counter[0]
                low[u] = time_counter[0]
                time_counter[0] += 1
                visited.add(u)
                for v in adj[u]:
                    if v == parent:
                        continue
                    if v not in disc:
                        dfs(v, u)
                        low[u] = min(low[u], low[v])
                        if low[v] > disc[u]:
                            bridges.add(tuple(sorted((u, v))))
                    else:
                        low[u] = min(low[u], disc[v])
            
            for node in adj:
                if node not in visited:
                    dfs(node, -1)
            return bridges
            
        # Find the longest cycle in a bridgeless graph
        def find_longest_cycle(edge_set):
            if not edge_set:
                return None
            
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
                
            best_cycle = None
            best_len = 0
            
            # Try removing each edge to see if a cycle forms
            # We only need to check edges that are part of a cycle.
            # An edge (u,v) is in a cycle if v is reachable from u in G \ {(u,v)}.
            
            for (u, v) in list(edge_set):
                # Temporarily remove (u,v)
                # To avoid modifying the main set, we can just check reachability
                # with the edge ignored.
                
                # BFS from u to v, ignoring (u,v)
                visited = set()
                parent = {}
                queue = deque([u])
                visited.add(u)
                found = False
                
                while queue:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    for nb in adj[curr]:
                        # Skip the edge (u,v) if we are going back to v from u?
                        # Actually, we are looking for a path from u to v in G \ {(u,v)}.
                        # So we should not traverse the edge (u,v) directly.
                        # Since adj is undirected, if curr==u and nb==v, skip.
                        # Also if curr==v and nb==u, skip? No, we stop at v.
                        # So we just skip the specific edge (u,v).
                        if (curr == u and nb == v) or (curr == v and nb == u):
                            continue
                            
                        if nb not in visited:
                            visited.add(nb)
                            parent[nb] = curr
                            queue.append(nb)
                            
                if found:
                    # Reconstruct path from v back to u
                    path = []
                    curr = v
                    while curr != u:
                        path.append(curr)
                        curr = parent[curr]
                    path.append(u)
                    # The cycle is path + edge (u,v)
                    # Length is len(path)
                    # path is u -> ... -> v. Cycle is u -> ... -> v -> u.
                    # The vertices in the cycle are path + [u] (but u is already there).
                    # Wait, path ends with u.
                    # Cycle vertices: path[0], path[1], ..., path[-1]
                    # Edges: (path[i], path[i+1]) and (path[-1], path[0])
                    # Since path is a simple path from u to v, and we add edge (v,u), it's a cycle.
                    cycle_len = len(path)
                    if cycle_len > best_len:
                        best_len = cycle_len
                        best_cycle = path
                        
            return best_cycle

        # Main decomposition loop
        while remaining:
            # Find bridges
            bridges = find_bridges(remaining)
            
            if bridges:
                # Remove all bridges as single edges
                for b in bridges:
                    pieces.append(list(b))
                    remaining.discard(b)
            else:
                # No bridges, find longest cycle
                cycle = find_longest_cycle(remaining)
                if cycle:
                    pieces.append(cycle)
                    # Remove edges of the cycle
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        remaining.discard(edge)
                else:
                    # Fallback: should not happen if no bridges, but handle it
                    # This implies the graph is a single edge? Or empty?
                    # If no bridges and no cycles, it must be empty or isolated nodes?
                    # Actually, a graph with no bridges and no cycles is impossible for connected components with edges.
                    # Unless it's a single edge? A single edge is a bridge.
                    # So this case should not occur.
                    # If it does, just take an edge.
                    if remaining:
                        e = next(iter(remaining))
                        pieces.append(list(e))
                        remaining.discard(e)
                    else:
                        break
                        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
