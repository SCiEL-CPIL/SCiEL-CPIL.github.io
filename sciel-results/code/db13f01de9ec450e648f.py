import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _find_path_bfs(start, end, adj, remaining_edges):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        # BFS
        parent = {start: None}
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            if curr == end:
                # Reconstruct path
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            
            for nb in sorted(adj[curr]):
                edge = tuple(sorted((curr, nb)))
                if edge in remaining_edges and nb not in parent:
                    parent[nb] = curr
                    queue.append(nb)
        
        return None  # No path found

    def _find_path_greedy(start, end, adj, remaining_edges, strategy="degree"):
        """Find a simple path from start to end using greedy heuristic."""
        if start == end:
            return [start]
        
        path = [start]
        visited = {start}
        current = start
        
        while current != end:
            # Get available neighbors
            avail = [nb for nb in adj[current] 
                     if tuple(sorted((current, nb))) in remaining_edges 
                     and nb not in visited]
            
            if not avail:
                return None  # Dead end
            
            if strategy == "degree":
                # Prefer lower degree neighbors (to preserve connectivity)
                avail.sort(key=lambda x: len([y for y in adj[x] if tuple(sorted((x, y))) in remaining_edges]))
            elif strategy == "random":
                random.shuffle(avail)
            elif strategy == "far":
                # Prefer neighbors further from end (less direct)
                avail.sort(key=lambda x: x)
            else:
                avail.sort()
            
            next_v = avail[0]
            edge = tuple(sorted((current, next_v)))
            remaining_edges.discard(edge)
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        return path

    def _decompose_optimal(edges, rng, max_time_per_graph):
        """Try to achieve the lower bound on number of paths."""
        if not edges:
            return []
        
        adj = _build_adj(edges)
        
        # Find odd-degree vertices
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        odd_verts = [v for v in deg if deg[v] % 2 == 1]
        num_paths_lower_bound = len(odd_verts) // 2
        
        if num_paths_lower_bound == 0:
            # All degrees even - Eulerian circuit exists
            # Find one path that covers all edges
            remaining = set(edges)
            if not remaining:
                return []
            start = list(remaining)[0][0]
            # Use Hierholzer's algorithm for Eulerian path/circuit
            path = _find_eulerian_path(start, adj, remaining)
            if path:
                return [path]
            # Fallback
            return _decompose_greedy(edges)
        
        # Try to find edge-disjoint paths between paired odd vertices
        best_paths = None
        best_count = float('inf')
        
        # Try multiple random pairings
        for attempt in range(50):
            if time.time() > start_time + max_time_per_graph:
                break
            
            # Random pairing of odd vertices
            pairs = list(odd_verts)
            rng.shuffle(pairs)
            pair_list = [(pairs[i], pairs[i+1]) for i in range(0, len(pairs), 2)]
            
            # Try to find edge-disjoint paths for each pair
            remaining = set(edges)
            paths = []
            success = True
            
            for (u, v) in pair_list:
                # Try BFS first
                path = _find_path_bfs(u, v, adj, remaining)
                if path is None:
                    # Try greedy
                    rem_copy = set(remaining)
                    path = _find_path_greedy(u, v, adj, rem_copy, "degree")
                    if path is None:
                        success = False
                        break
                    # Remove edges from path
                    for i in range(len(path)-1):
                        edge = tuple(sorted((path[i], path[i+1])))
                        remaining.discard(edge)
                else:
                    for i in range(len(path)-1):
                        edge = tuple(sorted((path[i], path[i+1])))
                        remaining.discard(edge)
                
                paths.append(path)
            
            if success and len(remaining) == 0:
                # Check if we've covered all edges
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = paths
                    if best_count == num_paths_lower_bound:
                        break  # Achieved lower bound!
            
            elif success and len(remaining) == 0:
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = paths
        
        if best_paths is not None:
            return best_paths
        
        # Fallback to greedy
        return _decompose_greedy(edges)

    def _find_eulerian_path(start, adj, edges_set):
        """Find an Eulerian path/circuit using Hierholzer's algorithm."""
        # Build adjacency with edge tracking
        adj_list = defaultdict(list)
        for u, v in edges_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Find start vertex with odd degree if exists
        deg = {v: len(adj_list[v]) for v in adj_list}
        odd_verts = [v for v in deg if deg[v] % 2 == 1]
        if odd_verts:
            start = odd_verts[0]
        
        # Hierholzer's
        stack = [start]
        circuit = []
        remaining = set(edges_set)
        
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            next_v = None
            for nb in adj_list[v]:
                edge = tuple(sorted((v, nb)))
                if edge in remaining:
                    next_v = nb
                    break
            
            if next_v is not None:
                edge = tuple(sorted((v, next_v)))
                remaining.discard(edge)
                stack.append(next_v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Verify all edges used
        if len(remaining) > 0:
            return None
        
        return circuit

    def _decompose_greedy(edges):
        """Greedy decomposition as fallback."""
        if not edges:
            return []
        
        adj = _build_adj(edges)
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            if time.time() > start_time + 135:
                # Fallback: return remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                return paths
            
            # Find start vertex: min degree in remaining graph
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            # Calculate degrees
            degs = {v: 0 for v in active_vertices}
            for u, v in remaining_edges:
                degs[u] += 1
                degs[v] += 1
            
            # Choose start: min degree
            start = min(active_vertices, key=lambda x: (degs[x], x))
            
            # Greedy path extension
            path = [start]
            current = start
            visited = {start}
            
            while True:
                avail = [nb for nb in adj[current] 
                         if tuple(sorted((current, nb))) in remaining_edges 
                         and nb not in visited]
                
                if not avail:
                    break
                
                # Prefer lower degree neighbors
                avail.sort(key=lambda x: degs[x])
                next_v = avail[0]
                
                edge = tuple(sorted((current, next_v)))
                remaining_edges.discard(edge)
                degs[current] -= 1
                degs[next_v] -= 1
                
                path.append(next_v)
                visited.add(next_v)
                current = next_v
            
            paths.append(path)
        
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    # Allocate time proportionally
    total_time = 135  # Leave 5s margin
    time_per_graph = total_time / 3
    
    for i, spec in enumerate(specs):
        if time.time() > start_time + total_time - 2:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            paths = _decompose_greedy(edges)
            results.append(paths)
            continue
        
        edges = instance(*spec)
        rng = random.Random(42 + i)
        
        # Try optimal decomposition first
        max_time = time_per_graph - 1
        paths = _decompose_optimal(edges, rng, max_time)
        
        # If still too many paths, try greedy as backup
        if time.time() < start_time + total_time - 1:
            greedy_paths = _decompose_greedy(edges)
            if len(greedy_paths) < len(paths):
                paths = greedy_paths
        
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
    print(f"Total: {sum(len(r) for r in res)} paths")
