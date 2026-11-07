"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations, permutations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Helper: Check if a path is valid (simple, endpoints correct, no branch interiors)
    # This is expensive, so we only do it when we have a candidate set of paths
    def is_valid_configuration(branches, paths_dict):
        # Check branches are distinct
        if len(set(branches)) != len(branches):
            return False
        
        # Check all required pairs have paths
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict and (v, u) not in paths_dict:
                return False
                
        # Collect all interior vertices
        used_interiors = set()
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] != u and path[0] != v:
                return False
            if path[-1] != u and path[-1] != v:
                return False
            if path[0] == path[-1]:
                return False
                
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
            # Check interiors are disjoint and avoid branches
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
            
        return True

    # Helper: Find a simple path between u and v avoiding a set of forbidden nodes
    # Uses BFS for shortest path, which is likely to have fewer conflicts
    def find_path(u, v, forbidden):
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v] # Degenerate, shouldn't happen for distinct branches
            
        visited = {u}
        queue = [[u]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in adj[node]:
                if neighbor == v:
                    return path + [v]
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    # Initial solution: Find a clique or just start with 2 nodes
    # Let's try to find a large clique first to seed the solution
    best_branches = []
    best_paths = {}
    
    # Try all subsets of size 3, 4, 5... to find an initial valid configuration
    # Since n=24, checking all subsets is expensive.
    # Let's just start with 2 random nodes or a known edge.
    # Better: Start with the seed's triangle if found, or just 2 nodes.
    
    # Greedy initialization: Pick a node, find its neighbors, try to form a clique
    # Actually, let's just start with 2 nodes that are connected.
    if not edges:
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    u0, v0 = edges[0]
    current_branches = [u0, v0]
    current_paths = {(u0, v0): [u0, v0]}
    
    # Try to expand this set
    # We will use a simple local search:
    # 1. Try to add a new branch from remaining nodes.
    # 2. If adding fails, try to swap a branch? (Complex)
    # 3. Random restarts.
    
    best_size = 2
    
    # Candidate pool for new branches: all nodes
    # To speed up, we can prioritize nodes with high degree?
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Pick a random node to try adding as a branch
        # Or pick from nodes not currently in branches
        candidates = [i for i in range(n) if i not in current_branches]
        if not candidates:
            break
            
        # Shuffle candidates to randomize
        random.shuffle(candidates)
        
        added = False
        for new_b in candidates:
            # Try to add new_b to current_branches
            new_branches = current_branches + [new_b]
            
            # We need paths from new_b to all existing branches
            # And we need to ensure these paths are disjoint from existing path interiors
            # AND disjoint from each other
            
            # This is a constraint satisfaction problem.
            # Simple approach: Try to find paths sequentially, updating forbidden set.
            
            # Sort existing branches? Maybe order by difficulty?
            # Let's just iterate through existing branches
            
            forbidden = set()
            # Add all current path interiors to forbidden
            for key, path in current_paths.items():
                forbidden.update(path[1:-1])
                
            # Also, the new paths cannot use any branch vertices (other than endpoints)
            # The `find_path` function handles `forbidden` which includes interiors.
            # But `find_path` also needs to avoid other branches?
            # Yes, "all path interiors must ... avoid every branch".
            # So forbidden should include all branches except the endpoints of the specific path.
            
            new_paths = dict(current_paths) # Copy existing paths
            
            success = True
            # We need to find paths for (new_b, b) for all b in current_branches
            # The order might matter for finding disjoint paths.
            # Let's try to find them one by one.
            
            # To be more robust, we can try multiple random orders of existing branches
            # But for speed, let's just try one order first. If it fails, we might retry with different order later.
            
            # Actually, a better heuristic:
            # Find paths for the new branch to all old branches.
            # We must ensure these new paths don't intersect each other or old paths.
            
            # Let's collect the new paths we need to find
            needed_paths = []
            for b in current_branches:
                needed_paths.append((new_b, b))
                
            # Try to find these paths
            found_paths = {}
            current_forbidden = set(forbidden)
            
            # We will pick a random order for needed_paths to break symmetry
            random.shuffle(needed_paths)
            
            for u, v in needed_paths:
                # Forbidden for this specific path:
                # 1. All nodes in current_forbidden (old interiors + previously found new interiors)
                # 2. All branches except u and v
                path_forbidden = set(current_forbidden)
                for b in new_branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                        
                p = find_path(u, v, path_forbidden)
                if p is None:
                    success = False
                    break
                found_paths[(u, v)] = p
                # Update current_forbidden with interiors of this new path
                current_forbidden.update(p[1:-1])
                
            if success:
                # Merge found_paths into new_paths
                # Note: keys in found_paths are (new_b, b). In new_paths, keys are tuples.
                # We need to ensure consistent key representation or just use the tuple as is.
                # The validation function checks `if (u, v) not in paths_dict and (v, u) not in paths_dict`.
                # So we can store with any orientation.
                for k, p in found_paths.items():
                    new_paths[k] = p
                    
                # Check if the whole configuration is valid (redundant if we did it carefully, but safe)
                # is_valid_configuration is O(V^2) roughly. With n=24, it's fast.
                if is_valid_configuration(new_branches, new_paths):
                    current_branches = new_branches
                    current_paths = new_paths
                    best_size = len(current_branches)
                    added = True
                    break # Restart the candidate loop with the new state? 
                    # Actually, just break out of the `for new_b` loop and continue the `while` loop
                    # to try adding another node.
            
        if not added:
            # Stuck. Try a random restart or perturbation?
            # Simple restart: Pick 2 random nodes again?
            # Or try to remove a branch and add a different one?
            # For now, let's just restart with a different random pair.
            # This is crude but might find a better basin.
            u0, v0 = random.sample(range(n), 2)
            if u0 in adj[v0]:
                current_branches = [u0, v0]
                current_paths = {(u0, v0): [u0, v0]}
            else:
                # Find a random edge
                u0, v0 = random.choice(edges)
                current_branches = [u0, v0]
                current_paths = {(u0, v0): [u0, v0]}
                
    # Final check and format
    if len(current_branches) < 2:
         # Fallback to seed behavior
         u, v = edges[0]
         return {'branches': [u, v], 'paths': [[u, v]]}
         
    # Construct final paths list
    final_paths = []
    for u, v in combinations(current_branches, 2):
        # Find the path in current_paths
        if (u, v) in current_paths:
            final_paths.append(list(current_paths[(u, v)]))
        elif (v, u) in current_paths:
            final_paths.append(list(current_paths[(v, u)]))
        else:
            # Should not happen if is_valid_configuration passed
            # But just in case, use a direct edge if exists, else error
            # The contract says "one simple path for each unordered pair".
            # If missing, the result is invalid.
            # We assume is_valid_configuration caught this.
            pass
            
    return {
        'branches': list(current_branches),
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
