import time
import random
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 10s margin
    
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Convert to lists for faster iteration if needed, but sets are good for membership
    # We need to handle "forbidden" edges dynamically.
    
    def find_ham_cycle(forbidden):
        """
        Finds a Hamiltonian cycle in the graph defined by adj, avoiding edges in 'forbidden'.
        Returns a list of vertices [v0, v1, ..., v(n-1)] such that (vi, vi+1) and (vn-1, v0) are edges not in forbidden.
        Returns None if not found.
        """
        # Start at 0 to fix rotation
        start = 0
        path = [start]
        visited = {start}
        
        def dfs(current, path, visited):
            if len(path) == n:
                # Check if edge back to start exists and is not forbidden
                end = current
                if start in adj[end] and (min(start, end), max(start, end)) not in forbidden:
                    return list(path)
                return None
            
            # Get neighbors
            neighbors = list(adj[current])
            
            # Filter out visited and forbidden
            candidates = []
            for nb in neighbors:
                if nb in visited:
                    continue
                if (min(current, nb), max(current, nb)) in forbidden:
                    continue
                candidates.append(nb)
                
            if not candidates:
                return None
                
            # Heuristic: Warnsdorff's rule
            # Sort by number of unvisited neighbors
            def heuristic_score(nb):
                count = 0
                for n2 in adj[nb]:
                    if n2 not in visited and (min(nb, n2), max(nb, n2)) not in forbidden:
                        count += 1
                return count
            
            candidates.sort(key=heuristic_score)
            
            # Randomize ties to allow exploration
            # Group by score? Or just shuffle the whole list if we want more randomness?
            # Let's shuffle candidates with the same score, or just fully shuffle if we want random walk.
            # Warnsdorff is usually greedy. Let's add some randomness:
            # Pick from top K? Or just random choice from sorted list?
            # To be safe, let's try a few random permutations of the sorted list?
            # No, that's expensive.
            # Let's just iterate through the sorted list.
            
            for nb in candidates:
                path.append(nb)
                visited.add(nb)
                result = dfs(nb, path, visited)
                if result:
                    return result
                path.pop()
                visited.remove(nb)
                
            return None
            
        return dfs(start, path, visited)

    forbidden = set()
    cycles = []
    
    # We want to find up to 3 cycles.
    # We will try multiple times to find the next cycle.
    # If we fail, we might restart the whole process with a different seed?
    # Or just keep trying with different random seeds for the DFS?
    # The DFS above is deterministic given the sort. We need to introduce randomness.
    
    def find_ham_cycle_randomized(forbidden, seed):
        random.seed(seed)
        # We can modify the heuristic to be random-ish
        # Or just shuffle the candidates list before sorting? No, sorting is key.
        # Let's shuffle the candidates, then sort by score, but break ties randomly?
        # Actually, simpler: just shuffle the candidates list completely, then try them in that order.
        # This is a random DFS. With N=21, it might be slow if not guided.
        # Let's combine: Sort by score, but if scores are equal, random order.
        
        def dfs_rand(current, path, visited):
            if len(path) == n:
                end = current
                if start in adj[end] and (min(start, end), max(start, end)) not in forbidden:
                    return list(path)
                return None
            
            neighbors = list(adj[current])
            candidates = []
            for nb in neighbors:
                if nb in visited:
                    continue
                if (min(current, nb), max(current, nb)) in forbidden:
                    continue
                candidates.append(nb)
                
            if not candidates:
                return None
                
            def heuristic_score(nb):
                count = 0
                for n2 in adj[nb]:
                    if n2 not in visited and (min(nb, n2), max(nb, n2)) not in forbidden:
                        count += 1
                return count
            
            # Calculate scores
            scored = [(heuristic_score(nb), nb) for nb in candidates]
            # Shuffle to randomize ties and general order?
            # If we just sort, it's deterministic.
            # Let's shuffle, then stable sort? No.
            # Let's just shuffle the candidates list, and then sort by score?
            # That keeps the greedy nature but randomizes the initial order.
            random.shuffle(candidates)
            # Re-evaluate? No, just sort the shuffled list by score.
            # Wait, if I shuffle then sort by score, the order of equal scores is random.
            # This is good.
            
            scored = [(heuristic_score(nb), nb) for nb in candidates]
            scored.sort(key=lambda x: x[0])
            
            for score, nb in scored:
                path.append(nb)
                visited.add(nb)
                result = dfs_rand(nb, path, visited)
                if result:
                    return result
                path.pop()
                visited.remove(nb)
                
            return None

        start = 0
        path = [start]
        visited = {start}
        return dfs_rand(start, path, visited)

    for i in range(3):
        found = False
        # Try many random seeds
        for attempt in range(100):
            if time.time() > deadline:
                break
            seed = random.randint(0, 1000000)
            cycle = find_ham_cycle_randomized(forbidden, seed)
            if cycle:
                # Validate cycle
                valid = True
                # Check distinct vertices
                if len(set(cycle)) != n:
                    valid = False
                # Check edges
                for j in range(n):
                    u = cycle[j]
                    v = cycle[(j+1)%n]
                    if (min(u,v), max(u,v)) not in [(min(e[0], e[1]), max(e[0], e[1])) for e in edges]:
                        valid = False
                    if (min(u,v), max(u,v)) in forbidden:
                        valid = False
                if valid:
                    cycles.append(cycle)
                    # Add edges to forbidden
                    for j in range(n):
                        u = cycle[j]
                        v = cycle[(j+1)%n]
                        forbidden.add((min(u,v), max(u,v)))
                    found = True
                    break
        if not found:
            # If we can't find the next cycle, break
            break
            
    return cycles
