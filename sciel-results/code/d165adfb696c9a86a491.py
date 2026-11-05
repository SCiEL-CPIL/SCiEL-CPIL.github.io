import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    if not components:
        return [0]
        
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    
    def greedy_path(start):
        """
        Greedy path construction using minimum degree heuristic.
        """
        if start not in comp_set:
            return []
        
        path = [start]
        seen = {start}
        
        while True:
            current = path[-1]
            neighbors = [x for x in adj[current] if x not in seen and x in comp_set]
            
            if not neighbors:
                break
                
            # Choose neighbor with fewest unvisited neighbors (Warnsdorff's rule variant)
            next_node = min(neighbors, key=lambda x: len([y for y in adj[x] if y not in seen and y in comp_set]))
            
            path.append(next_node)
            seen.add(next_node)
            
        return path

    def local_search(path, time_limit):
        """
        Improves the path using insertion and 2-opt-like moves.
        """
        if not path:
            return path
            
        path_set = set(path)
        candidates = [x for x in comp_set if x not in path_set]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to insert a candidate
            # Sort candidates by degree in path to prioritize likely insertions?
            # Or just iterate.
            # Optimization: Check if candidate has any neighbors in path.
            
            # We can speed this up by precomputing neighbors in path for each candidate?
            # For now, simple iteration.
            
            for u in candidates[:]: # Iterate over a copy
                if not (adj[u] & path_set):
                    continue
                
                # Check for direct insertion: (v_k, u, v_{k+1})
                inserted = False
                for i in range(len(path) - 1):
                    if u in adj[path[i]] and u in adj[path[i+1]]:
                        path.insert(i + 1, u)
                        path_set.add(u)
                        candidates.remove(u)
                        improved = True
                        inserted = True
                        break
                if inserted:
                    break
            
            if improved:
                continue
            
            # 2. If no insertion, try to extend from ends
            # This is less effective than insertion but can help if path is short
            # Actually, greedy already did this.
            
            # 3. Try 2-opt move?
            # A 2-opt move in a path context:
            # Reverse a segment v[i..j] to v[j..i].
            # This is valid if edge (v[i-1], v[j]) and (v[i], v[j+1]) exist.
            # This does NOT change the number of vertices, but might enable future insertions.
            # However, for maximizing length, we primarily care about adding vertices.
            # If we are stuck, we might need to rearrange to allow insertion.
            
            # Let's try a few random 2-opt moves if stuck.
            if len(path) > 2:
                moved = False
                for _ in range(10): # Limit attempts
                    i = random.randint(1, len(path) - 2)
                    j = random.randint(i + 1, len(path) - 1)
                    
                    # Check validity of 2-opt move
                    # Original: ... v[i-1] - v[i] ... v[j] - v[j+1] ...
                    # New: ... v[i-1] - v[j] ... v[i] - v[j+1] ...
                    
                    left_node = path[i-1]
                    right_node = path[j+1] if j < len(path) else None
                    
                    # If j is the last node, we are just reversing the tail?
                    # No, 2-opt usually requires two cuts.
                    # Cut 1: between i-1 and i
                    # Cut 2: between j and j+1
                    # Connect i-1 to j, and i to j+1.
                    
                    if j == len(path) - 1:
                        # Only one cut? No, that's just reversal from i to end.
                        # This is valid if edge (i-1, j) exists.
                        # The other end is free.
                        if left_node in adj[path[j]]:
                            path[i:] = path[i:][::-1]
                            moved = True
                            break
                    else:
                        if left_node in adj[path[j]] and path[i] in adj[right_node]:
                            path[i:j+1] = path[i:j+1][::-1]
                            moved = True
                            break
                            
                if moved:
                    improved = True
                    continue
                    
            # If no improvement found in this round, break to avoid infinite loop
            # Or we could randomize the order of candidates?
            # For now, break.
            break
            
        return path

    best_path = []
    
    # Time limit guard
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Run greedy from each start node
    for start in largest_comp:
        if time.time() - start_time > time_limit:
            break
        p = greedy_path(start)
        if len(p) > len(best_path):
            best_path = p
            
    # 2. Run local search on the best path
    if best_path:
        refined = local_search(best_path, time_limit)
        if len(refined) > len(best_path):
            best_path = refined
            
    # 3. Random restarts with local search
    rng = random.Random(748)
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        
        start = rng.choice(largest_comp)
        p = greedy_path(start)
        
        # Also try a random walk initialization
        # p_random = [start]
        # seen = {start}
        # while True:
        #     cur = p_random[-1]
        #     neigh = [x for x in adj[cur] if x not in seen and x in comp_set]
        #     if not neigh: break
        #     nxt = rng.choice(neigh)
        #     p_random.append(nxt)
        #     seen.add(nxt)
        # if len(p_random) > len(p): p = p_random
        
        p = local_search(p, time_limit)
        
        if len(p) > len(best_path):
            best_path = p
            # If we found a very long path, maybe stop early?
            # But let's keep going to be safe.
            
    if not best_path:
        return [0]
        
    return best_path
