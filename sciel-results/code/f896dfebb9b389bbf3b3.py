import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    # Build base adjacency and edge set
    adj_base = [0] * n
    edge_set = set()
    for u, v in base_edges:
        adj_base[u] |= (1 << v)
        adj_base[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Generate candidate edges: Only Even-Odd pairs not in base
    # This ensures triangle-free property (bipartite graph)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in edge_set:
                candidates.append((u, v))
                
    # Sort candidates by "length" (difference in index) descending
    # Longer edges are more likely to reduce diameter significantly
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    def get_diameter(adj_mask):
        """Compute diameter using bitmask BFS. Returns diameter or -1 if disconnected."""
        max_dist = 0
        full_mask = (1 << n) - 1
        for start in range(n):
            visited = 1 << start
            frontier = 1 << start
            dist = 0
            while frontier:
                dist += 1
                new_frontier = 0
                f = frontier
                while f:
                    lb = f & (-f)
                    f ^= lb
                    v = lb.bit_length() - 1
                    nbrs = adj_mask[v] & ~visited
                    new_frontier |= nbrs
                if not new_frontier:
                    break
                visited |= new_frontier
                frontier = new_frontier
            if visited != full_mask:
                return -1
            max_dist = max(max_dist, dist - 1)
        return max_dist

    def is_valid(added_edges_set):
        """Check if the set of added edges results in a valid graph."""
        adj_test = adj_base[:]
        for u, v in added_edges_set:
            adj_test[u] |= (1 << v)
            adj_test[v] |= (1 << u)
        
        # Triangle-free is guaranteed by construction (Even-Odd only)
        # Check diameter
        diam = get_diameter(adj_test)
        return diam <= 4

    # We will try to find a solution with k edges, starting from small k
    # k=8, 9, 10, 11, 12
    # We use a randomized local search / hill climbing approach
    
    best_solution = None
    
    for k in range(8, 13):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts for this k
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Initialize with k random candidates, preferring longer ones
            # We pick a mix: some top candidates, some random
            current_edges = set()
            
            # Add some "strong" candidates first to ensure connectivity/diameter reduction
            if k > 0:
                # Pick 1-2 from the top
                top_count = min(2, k)
                for i in range(top_count):
                    if i < len(candidates):
                        current_edges.add(candidates[i])
                        
            # Fill the rest randomly
            while len(current_edges) < k and len(candidates) > 0:
                idx = random.randint(0, len(candidates) - 1)
                current_edges.add(candidates[idx])
                
            if len(current_edges) < k:
                continue
                
            # Local Search: Try to swap edges to improve diameter
            # We don't have a direct "score" for diameter < 4, so we just check validity.
            # If valid, we are done.
            # If not, we try to swap an edge with another candidate to see if it becomes valid.
            # This is a blind search, but effective for small N.
            
            if is_valid(current_edges):
                best_solution = list(current_edges)
                found = True
                break
                
            # If not valid, try some swaps
            # Limit swaps per restart to keep it fast
            for swap_attempt in range(50):
                if time.time() > deadline:
                    break
                    
                # Pick a random edge to remove
                e_to_remove = random.choice(list(current_edges))
                current_edges.remove(e_to_remove)
                
                # Pick a random candidate to add (not already in set)
                # To make it smarter, pick a candidate that is "long" or connects isolated parts
                # For now, random from candidates
                e_to_add = random.choice(candidates)
                
                if e_to_add in current_edges:
                    continue
                    
                current_edges.add(e_to_add)
                
                if is_valid(current_edges):
                    best_solution = list(current_edges)
                    found = True
                    break
                    
            if found:
                break
                
        if found:
            break
            
    if best_solution is None:
        # Fallback: The known 12-edge solution
        # Hub 11 connected to all Evens
        fallback = set()
        hub = 11
        for u in range(n):
            if u % 2 == 0 and (u, hub) not in edge_set:
                fallback.add((min(u, hub), max(u, hub)))
        best_solution = list(fallback)
        
    # Ensure types are correct
    result = [[int(u), int(v)] for u, v in best_solution]
    return result

