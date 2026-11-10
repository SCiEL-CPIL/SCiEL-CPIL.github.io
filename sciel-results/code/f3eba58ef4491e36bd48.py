import itertools
import random
import time

def construct():
    deadline = time.time() + 50  # 50s limit, leaving 10s for safety
    best_edges = None
    best_score = float('inf')
    
    # Candidate 1: Seed solution (2 * K3,3)
    # Vertices 0-5 form K3,3, 6-11 form K3,3
    # K3,3 edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    # Shifted by 6 for the second copy
    seed_edges = []
    for i in range(3):
        for j in range(3):
            seed_edges.append([i, j+3])
            seed_edges.append([i+6, j+9])
            
    # Evaluate seed
    score, valid = evaluate(seed_edges)
    if valid:
        best_score = score
        best_edges = seed_edges

    # Search for better solution
    # We want to find a graph with score < best_score
    # If best_score is 2, we look for 1.
    
    # Simple random search for 3-regular graphs
    # Generating random 3-regular graphs on 12 vertices
    # We can use a random permutation of a known 3-regular graph or random configuration model
    
    # Configuration model for 3-regular graph on 12 vertices
    # 12 vertices, each degree 3 -> 36 stubs
    # Pair stubs randomly
    
    def generate_random_3reg():
        stubs = []
        for i in range(12):
            for _ in range(3):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        for i in range(0, 36, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Loop, invalid
            if (min(u,v), max(u,v)) in edges:
                return None # Multi-edge, invalid
            edges.add((min(u,v), max(u,v)))
        if len(edges) != 18: # 12*3/2 = 18 edges
            return None
        return [list(e) for e in edges]

    def evaluate(edges):
        # Check degrees
        deg = [0]*12
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < 3:
            return float('inf'), False
            
        # Check triangle-free
        # Build adjacency list
        adj = {i: set() for i in range(12)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check triangles
        for u in range(12):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return float('inf'), False # Triangle found
        
        # Find all simple cycles
        # Use DFS or backtracking
        # Since n=12, this is feasible
        cycles = set()
        
        def dfs(start, curr, visited, path):
            if len(path) > 1 and curr == start:
                if len(path) >= 4: # Triangle-free already checked, but cycle length >= 4
                    cycles.add(tuple(sorted(path)))
                return
            if len(path) > 12:
                return
                
            for next_node in adj[curr]:
                if next_node == start and len(path) >= 4:
                    cycles.add(tuple(sorted(path)))
                elif next_node not in visited:
                    visited.add(next_node)
                    path.append(next_node)
                    dfs(start, next_node, visited, path)
                    path.pop()
                    visited.remove(next_node)
                    
        # To avoid duplicates, we only start cycles from the smallest node in the cycle
        # And we ensure the path is sorted? No, path is ordered.
        # We can canonicalize the cycle by rotating it to start with the smallest node
        # and taking the lexicographically smallest representation
        
        def canonicalize(cycle):
            # cycle is a tuple of nodes
            min_idx = cycle.index(min(cycle))
            # Rotate
            rotated = cycle[min_idx:] + cycle[:min_idx]
            # Check reverse
            reversed_rot = tuple(reversed(rotated))
            if reversed_rot < rotated:
                return reversed_rot
            return rotated

        unique_cycles = set()
        for i in range(12):
            # Only start from i if i is the smallest in the potential cycle?
            # We can just run DFS from all nodes and canonicalize
            # But to save time, we can restrict
            # Let's just run from all and use set of canonicalized cycles
            visited = {i}
            path = [i]
            dfs(i, i, visited, path)
            
        for c in cycles:
            unique_cycles.add(canonicalize(c))
            
        if not unique_cycles:
            return float('inf'), False # Must contain a cycle
            
        lengths = set(len(c) for c in unique_cycles)
        return len(lengths), True

    # Run search
    attempts = 0
    while time.time() < deadline:
        edges = generate_random_3reg()
        if edges is None:
            continue
            
        score, valid = evaluate(edges)
        if valid:
            if score < best_score:
                best_score = score
                best_edges = edges
                if best_score == 1:
                    break # Can't do better than 1
            
        attempts += 1
        if attempts > 10000: # Safety break
            break

    if best_edges is None:
        # Fallback to seed
        return seed_edges
        
    return best_edges
