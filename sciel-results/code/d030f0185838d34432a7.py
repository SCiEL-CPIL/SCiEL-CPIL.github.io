import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def path_is_valid(path):
        if not path:
            return False
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True

    def greedy_warnsdorff(start, randomize, rng=None):
        path = [start]
        visited = {start}
        while True:
            current = path[-1]
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            scored = []
            for n in neighbors:
                unvisited_count = len(adj[n] - visited)
                scored.append((unvisited_count, n))
            
            scored.sort()
            
            if randomize and rng:
                top_k = scored[:3]
                choice = rng.choice(top_k)
                next_node = choice[1]
            else:
                next_node = scored[0][1]
                
            path.append(next_node)
            visited.add(next_node)
            
        return path

    def extend_path(path):
        if not path:
            return path
        visited = set(path)
        changed = True
        while changed:
            changed = False
            # Extend start
            start_node = path[0]
            neighbors = [n for n in adj[start_node] if n not in visited]
            if neighbors:
                scored = [(len(adj[n] - visited), n) for n in neighbors]
                scored.sort()
                next_node = scored[0][1]
                path.insert(0, next_node)
                visited.add(next_node)
                changed = True
            
            # Extend end
            end_node = path[-1]
            neighbors = [n for n in adj[end_node] if n not in visited]
            if neighbors:
                scored = [(len(adj[n] - visited), n) for n in neighbors]
                scored.sort()
                next_node = scored[0][1]
                path.append(next_node)
                visited.add(next_node)
                changed = True
        return path

    def try_insert_vertex(path):
        """Try to insert unvisited vertices into the path at positions where both neighbors are adjacent."""
        if not path:
            return path
        visited = set(path)
        unvisited = [v for v in range(n) if v not in visited]
        
        inserted = True
        while inserted:
            inserted = False
            for v in unvisited[:]:
                if v in visited:
                    continue
                # Try to insert v between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    if v in adj[path[i]] and v in adj[path[i+1]]:
                        path.insert(i + 1, v)
                        visited.add(v)
                        unvisited.remove(v)
                        inserted = True
                        break
        return path

    def local_search_improve(path, deadline):
        """Optimized 2-opt style local search: check only affected edges for validity."""
        if not path:
            return path
        
        best_path = list(path)
        best_len = len(best_path)
        n = len(best_path)
        
        while time.time() < deadline:
            improved = False
            # We only need to check if the reversal creates valid connections at the boundaries
            # and then try to extend.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Check validity of reversal without creating new list yet
                    # Left connection: path[i-1] to path[j]
                    # Right connection: path[i] to path[j+1]
                    
                    valid = True
                    if i > 0:
                        if best_path[j] not in adj[best_path[i-1]]:
                            valid = False
                    if valid and j < n - 1:
                        if best_path[i] not in adj[best_path[j+1]]:
                            valid = False
                    
                    if not valid:
                        continue
                    
                    # If valid, construct new path and try to extend
                    new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                    extended = extend_path(new_path)
                    if len(extended) > best_len:
                        best_path = extended
                        best_len = len(best_path)
                        improved = True
                        n = len(best_path)
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_path

    def beam_search(start, beam_width=10, deadline=None):
        """Beam search for longest path."""
        if deadline is None:
            deadline = time.time() + 5.0
        
        # Each beam entry: (path, visited_set)
        beams = [([start], {start})]
        
        while time.time() < deadline:
            new_beams = []
            for path, visited in beams:
                current = path[-1]
                neighbors = [n for n in adj[current] if n not in visited]
                if not neighbors:
                    continue
                
                scored = []
                for n in neighbors:
                    unvisited_count = len(adj[n] - visited)
                    scored.append((unvisited_count, n))
                scored.sort()
                
                # Take top k neighbors
                for _, next_node in scored[:beam_width]:
                    new_path = path + [next_node]
                    new_visited = visited | {next_node}
                    new_beams.append((new_path, new_visited))
            
            if not new_beams:
                break
            
            # Keep only top beam_width paths by length
            new_beams.sort(key=lambda x: -len(x[0]))
            beams = new_beams[:beam_width]
            
            # Check if any beam is complete (can't extend)
            all_complete = all(len(path) == len(set(path)) and 
                             not any(n not in visited for n in adj[path[-1]]) 
                             for path, visited in beams)
            if all_complete:
                break
        
        # Return the longest path from beams
        if beams:
            longest = max(beams, key=lambda x: len(x[0]))
            return longest[0]
        return [start]

    def random_walk_with_backtracking(start, max_steps=200, deadline=None):
        """Random walk with backtracking to find long paths."""
        if deadline is None:
            deadline = time.time() + 5.0
        
        rng = random.Random(start)
        best_path = [start]
        current_path = [start]
        visited = {start}
        stack = [(start, list(adj[start]))]
        
        while time.time() < deadline and len(current_path) < n:
            if not stack:
                break
            
            node, neighbors = stack[-1]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                # Backtrack
                stack.pop()
                if stack:
                    prev_node, _ = stack[-1]
                    visited.discard(node)
                    current_path.pop()
                continue
            
            # Pick a random unvisited neighbor
            next_node = rng.choice(unvisited_neighbors)
            current_path.append(next_node)
            visited.add(next_node)
            stack.append((next_node, list(adj[next_node])))
            
            if len(current_path) > len(best_path):
                best_path = list(current_path)
        
        return best_path

    best_path = []
    # Use a safe deadline to ensure we return before the 60s limit (accounting for overhead)
    deadline = time.time() + 50.0
    
    # Phase 1: Deterministic from all starts
    for start in range(n):
        if time.time() > deadline - 10:
            break
        path = greedy_warnsdorff(start, False)
        path = extend_path(path)
        path = try_insert_vertex(path)
        if len(path) > len(best_path):
            best_path = path

    # Phase 2: Local search improvement on best path found so far
    if best_path:
        improved = local_search_improve(best_path, deadline - 5)
        if len(improved) > len(best_path):
            best_path = improved

    # Phase 3: Randomized restarts with beam search and random walks
    rng = random.Random(42)
    while time.time() < deadline - 5:
        start = rng.randint(0, n - 1)
        
        # Try greedy
        path = greedy_warnsdorff(start, True, rng)
        path = extend_path(path)
        path = try_insert_vertex(path)
        if len(path) > len(best_path):
            best_path = path
        
        # Try beam search (limited time)
        if time.time() < deadline - 10:
            beam_path = beam_search(start, beam_width=8, deadline=deadline - 10)
            beam_path = extend_path(beam_path)
            beam_path = try_insert_vertex(beam_path)
            if len(beam_path) > len(best_path):
                best_path = beam_path
        
        # Try random walk with backtracking (limited time)
        if time.time() < deadline - 15:
            walk_path = random_walk_with_backtracking(start, deadline=deadline - 15)
            walk_path = extend_path(walk_path)
            walk_path = try_insert_vertex(walk_path)
            if len(walk_path) > len(best_path):
                best_path = walk_path
    
    # Final polish: local search on the best path
    if best_path and time.time() < deadline - 2:
        improved = local_search_improve(best_path, deadline - 2)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Ensure validity
    if not path_is_valid(best_path):
        # Fallback: find any valid path
        for start in range(n):
            path = greedy_warnsdorff(start, False)
            if path_is_valid(path) and len(path) > len(best_path):
                best_path = path
                break
    
    return best_path
