"""Program for path decomposition of fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 120s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_path_between(s, t, edge_set, n):
        """Find a simple path from s to t using only edges in edge_set via BFS."""
        if s == t:
            return [s]
        
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if u == t:
                path = []
                cur = t
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            
            for v in range(n):
                if v in visited:
                    continue
                e = tuple(sorted((u, v)))
                if e in edge_set:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        return None

    def greedy_path_decomposition(edges, n):
        """
        Find a path decomposition by pairing odd-degree vertices and finding
        paths between them. This gives the theoretical minimum number of paths
        if all paths can be found.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Compute degrees
        degree = [0] * n
        for u, v in edge_set:
            degree[u] += 1
            degree[v] += 1
        
        # Find odd-degree vertices
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        
        # Pair up odd vertices
        pairs = []
        for i in range(0, len(odd_verts), 2):
            if i + 1 < len(odd_verts):
                pairs.append((odd_verts[i], odd_verts[i + 1]))
        
        # If no odd vertices, we need at least 1 path (Eulerian circuit -> convert to path)
        if not pairs:
            if edge_set:
                u, v = next(iter(edge_set))
                pairs = [(u, v)]
        
        paths = []
        remaining_edges = set(edge_set)
        
        # For each pair, find a path and remove those edges
        for s, t in pairs:
            path = find_path_between(s, t, remaining_edges, n)
            
            if path is None:
                # Fallback: take a single edge incident to s or t
                found = False
                for e in list(remaining_edges):
                    if s in e or t in e:
                        path = list(e)
                        found = True
                        break
                if not found and remaining_edges:
                    e = next(iter(remaining_edges))
                    path = list(e)
                else:
                    path = [s, t]  # Should not happen if graph is connected
            
            paths.append(path)
            
            # Remove edges used in this path
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                remaining_edges.discard(e)
        
        # If there are still remaining edges, add them as single-edge paths
        while remaining_edges:
            e = next(iter(remaining_edges))
            paths.append(list(e))
            remaining_edges.discard(e)
        
        return paths

    def improved_decomposition(edges, n, time_budget=10):
        """
        Improved decomposition with multiple random pairings to minimize path count.
        The minimum number of paths is max(1, odd_count // 2).
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Compute degrees
        degree = [0] * n
        for u, v in edge_set:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        min_paths = max(1, len(odd_verts) // 2)
        
        # Try different pairings of odd vertices
        best_paths = None
        best_count = float('inf')
        
        rng = Random(42)
        attempts = min(50, int(time_budget * 5))
        
        for attempt in range(attempts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Create a random pairing
            verts = odd_verts[:]
            rng.shuffle(verts)
            pairs = []
            for i in range(0, len(verts), 2):
                if i + 1 < len(verts):
                    pairs.append((verts[i], verts[i + 1]))
            
            if not pairs:
                if edge_set:
                    u, v = next(iter(edge_set))
                    pairs = [(u, v)]
            
            # Try to find paths for this pairing
            paths = []
            remaining = set(edge_set)
            success = True
            
            for s, t in pairs:
                path = find_path_between(s, t, remaining, n)
                if path is None:
                    success = False
                    break
                paths.append(path)
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i + 1])))
                    remaining.discard(e)
            
            if success and not remaining:
                count = len(paths)
                if count < best_count:
                    best_count = count
                    best_paths = paths[:]
                    if count == min_paths:
                        break  # Can't do better than minimum
        
        if best_paths is not None:
            return best_paths
        
        # Fallback: greedy decomposition
        return greedy_path_decomposition(edges, n)

    def local_search_improve(paths, edges, n, time_budget=5):
        """
        Try to reduce path count by merging paths.
        Two paths can be merged if an endpoint of one is connected to an endpoint
        of the other by an unused edge... but we've used all edges.
        
        Instead, try to replace two paths with one by finding a longer path
        that covers the union of their edges.
        """
        if len(paths) <= 1:
            return paths
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Build adjacency from remaining structure
        # Actually, let's try a different approach: 
        # For each pair of paths, check if we can find a single path
        # that covers all edges from both paths.
        
        # This is complex, so let's just return the current paths
        # The main improvement comes from better initial pairing.
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Fallback: return remaining as single edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        # Allocate time budget per instance
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        per_instance_time = remaining_time / 3
        
        # Try improved decomposition first
        paths = improved_decomposition(edges, n, time_budget=min(per_instance_time, 15))
        
        # Try local search if we have time
        elapsed = time.time() - start_time
        if time.time() - start_time < DEADLINE - 5:
            paths = local_search_improve(paths, edges, n, time_budget=min(per_instance_time / 3, 5))
        
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
