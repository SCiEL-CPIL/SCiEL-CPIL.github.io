import time
import random
from collections import deque

def construct():
    n = 24
    path_edges = set()
    for i in range(n-1):
        path_edges.add((i, i+1))
        path_edges.add((i+1, i))
    
    # Generate valid candidates: opposite parity, not in path, length > 2
    # Length > 2 ensures we are not adding edges that are redundant for local structure
    # and helps in reducing diameter significantly.
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if u % 2 == v % 2:
                continue
            if (u, v) in path_edges:
                continue
            if v - u <= 2:
                continue
            candidates.append((u, v))
            
    # Sort candidates by length descending to prioritize long bridges
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    def get_diameter(edge_set):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in path_edges:
            if u < v:
                adj[u].add(v)
                adj[v].add(u)
        
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        max_d = 0
        # BFS from each node
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        q.append(v)
        return max_d

    deadline = time.time() + 50  # 50s limit to be safe
    
    # 1. Greedy construction to get an upper bound
    greedy_edges = set()
    current_diam = get_diameter(greedy_edges)
    while current_diam > 4 and time.time() < deadline:
        best_edge = None
        best_new_diam = current_diam
        for c in candidates:
            if c in greedy_edges:
                continue
            greedy_edges.add(c)
            new_diam = get_diameter(greedy_edges)
            greedy_edges.remove(c)
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = c
        if best_edge is None:
            break
        greedy_edges.add(best_edge)
        current_diam = best_new_diam
        
    upper_bound = len(greedy_edges)
    best_solution = list(greedy_edges)
    
    # 2. Try to find a smaller solution
    for k in range(1, upper_bound):
        # Time budget for each k
        k_deadline = time.time() + 5
        if time.time() > deadline:
            break
            
        found = False
        # Randomized Local Search
        # Number of restarts and steps adjusted by k
        restarts = 200
        steps_per_restart = 500
        
        for _ in range(restarts):
            if time.time() > k_deadline or time.time() > deadline:
                break
                
            # Initialize with random k edges from top candidates
            # Using top candidates increases chance of finding a good solution
            init_pool = candidates[:min(50, len(candidates))]
            current_edges = set(random.sample(init_pool, k))
            current_diam = get_diameter(current_edges)
            
            if current_diam <= 4:
                best_solution = list(current_edges)
                found = True
                break
                
            for _ in range(steps_per_restart):
                if time.time() > k_deadline or time.time() > deadline:
                    break
                    
                # Swap a random edge with another random edge
                if not current_edges:
                    break
                    
                e_rm = random.choice(list(current_edges))
                current_edges.remove(e_rm)
                
                e_add = random.choice(candidates)
                current_edges.add(e_add)
                
                new_diam = get_diameter(current_edges)
                
                # Accept move if it improves diameter or is equal (to explore)
                # For finding ANY solution <= 4, we can just check
                if new_diam <= 4:
                    best_solution = list(current_edges)
                    found = True
                    break
                    
                # Heuristic: If new_diam is worse, maybe revert?
                # Simple hill climbing: only accept if new_diam <= current_diam
                # But we want to escape local minima.
                # Let's use a simple random walk with bias towards improvement.
                if new_diam > current_diam:
                    # Revert with high probability
                    if random.random() < 0.8:
                        current_edges.remove(e_add)
                        current_edges.add(e_rm)
                    else:
                        current_diam = new_diam
                else:
                    current_diam = new_diam
                    
            if found:
                break
                
        if found:
            break
            
    # Ensure return type is correct
    # Convert tuples to lists
    return [list(e) for e in best_solution]

