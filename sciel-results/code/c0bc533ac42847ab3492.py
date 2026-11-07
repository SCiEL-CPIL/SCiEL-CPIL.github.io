import time
import random
from collections import deque

def construct():
    # Locate the input data in the global namespace
    P = None
    for name in ['P', 'p', 'data', 'graph', 'G', 'input', 'params', 'problem']:
        if name in globals():
            P = globals()[name]
            break

    if P is None:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Helper to find a path between start and end avoiding a set of blocked vertices
    # Returns the path as a list of vertices, or None if not found
    def find_path(start, end, blocked):
        if start == end:
            return [start]
        if start in blocked or end in blocked:
            return None
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == end:
                    return path + [end]
                if neighbor not in visited and neighbor not in blocked:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    best_branches = set()
    best_paths_dict = {} # (u, v) -> path
    best_size = 0

    # Heuristic to order candidates: prefer vertices with high degree or far from existing branches?
    # Let's try multiple strategies with random restarts.
    
    # Pre-compute degrees for initial heuristic
    degrees = [len(adj[i]) for i in range(n)]
    
    # To speed up, we can limit the number of restarts and use a time check.
    num_restarts = 1000
    random.seed(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_branches = set()
        current_paths = {}
        used_interiors = set()
        
        # Initialize with a random vertex or high-degree vertex
        if n > 0:
            # Strategy: Start with a random vertex, or one of the highest degree
            # Sometimes starting with a "central" vertex helps connect others
            # Let's pick a random vertex to ensure diversity
            start_v = random.randint(0, n - 1)
            current_branches.add(start_v)
            
            # Try to add other vertices
            # Order candidates by distance to start? Or just random shuffle?
            # Let's shuffle all other vertices
            candidates = [v for v in range(n) if v != start_v]
            random.shuffle(candidates)
            
            # To improve, we can sort candidates by some metric
            # e.g., degree descending
            # candidates.sort(key=lambda x: degrees[x], reverse=True)
            # But random shuffle + greedy might be better for exploration
            
            for v in candidates:
                if time.time() > deadline:
                    break
                
                if v in current_branches:
                    continue
                
                # Try to add v
                # We need paths from v to all u in current_branches
                new_paths = {}
                success = True
                new_used_interiors = set(used_interiors)
                
                for u in current_branches:
                    # The path must avoid current_branches (except endpoints) and used_interiors
                    # Note: used_interiors does not contain branches
                    blocked = used_interiors | (current_branches - {u, v})
                    # We also must not go through any other branch
                    # The find_path function handles 'blocked'
                    # But we must ensure the path does not pass through any branch in current_branches
                    # The 'blocked' set includes all branches except u and v.
                    # So find_path will avoid them.
                    
                    path = find_path(v, u, blocked)
                    if path is None:
                        success = False
                        break
                    
                    # Check if path interiors are disjoint from existing used_interiors
                    # find_path already avoided used_interiors, so they are disjoint.
                    # But we need to update new_used_interiors with the new path's interiors
                    # Note: If two new paths (v to u1 and v to u2) share an interior vertex, 
                    # the second find_path might have used a vertex that the first path used?
                    # No, find_path uses the 'blocked' set which is static during the loop.
                    # We must check if the new paths are mutually disjoint.
                    
                    interiors = set(path[1:-1])
                    if interiors & new_used_interiors:
                        success = False
                        break
                    
                    new_used_interiors.update(interiors)
                    new_paths[(min(v, u), max(v, u))] = path
                
                if success:
                    # Add v to branches
                    current_branches.add(v)
                    current_paths.update(new_paths)
                    used_interiors = new_used_interiors
                    # We don't break the for loop because we can add more vertices
                    # But we should restart the candidate list? 
                    # No, just continue. The order might matter.
                    # Actually, adding a vertex might make it harder to add others.
                    # Greedy: just continue.
        
        # Update best
        if len(current_branches) > best_size:
            best_size = len(current_branches)
            best_branches = set(current_branches)
            best_paths_dict = {k: list(p) for k, p in current_paths.items()}
            
            if best_size == n:
                break

    # Fallback
    if not best_paths_dict:
        if edges:
            u, v = edges[0]
            best_branches = {int(u), int(v)}
            best_paths_dict = {(min(u,v), max(u,v)): [int(u), int(v)]}
        else:
            best_branches = {0, 1}
            best_paths_dict = {(0, 1): [0, 1]}

    # Format output
    branches_out = [int(b) for b in best_branches]
    
    # Ensure paths are for all pairs
    paths_out = []
    for b1 in branches_out:
        for b2 in branches_out:
            if b1 < b2:
                key = (b1, b2)
                if key in best_paths_dict:
                    paths_out.append(best_paths_dict[key])
                else:
                    # Should not happen if valid
                    pass
    
    # Verify validity and fix if necessary (e.g. if path missing)
    # If any path is missing, the result is invalid.
    # We should ensure we only return valid results.
    # The greedy algorithm ensures validity if it succeeds.
    
    return {'branches': branches_out, 'paths': paths_out}
