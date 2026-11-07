import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    # Best result found
    best_edges = None
    best_count = float('inf')
    
    # Known good: two K_{3,3} -> {4,6} -> count=2
    def make_k33():
        edges = []
        for offset in (0, 6):
            part_a = [offset, offset+1, offset+2]
            part_b = [offset+3, offset+4, offset+5]
            for u in part_a:
                for v in part_b:
                    edges.append([u, v])
        return edges
    
    best_edges = make_k33()
    best_count = 2
    
    # Function to count distinct cycle lengths
    def count_distinct_cycle_lengths(edges):
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # Use DFS to find all simple cycles
        # To avoid duplicates, only start cycles from the smallest vertex
        # and only visit vertices greater than start in the path
        # except for the closing edge to start
        
        def dfs_cycle(current, start, path, visited):
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    cycle_lengths.add(len(path))
                elif neighbor not in visited and neighbor > start:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_cycle(neighbor, start, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(n):
            visited = {start}
            path = [start]
            dfs_cycle(start, start, path, visited)
        
        return len(cycle_lengths)

    # Local search from the known good graph
    random.seed(42)
    
    # Convert best_edges to a set for easy modification
    current_edges = set(tuple(sorted(e)) for e in best_edges)
    
    # Partitions for bipartite graph
    part_a = set(range(6))
    part_b = set(range(6, 12))
    
    # Ensure current_edges is bipartite (it is)
    
    for attempt in range(200000):
        if time.time() > deadline:
            break
        
        # Pick two edges to swap
        # e1 = (a1, b1), e2 = (a2, b2)
        # New edges: (a1, b2), (a2, b1)
        # This preserves 3-regularity and bipartiteness
        
        edge_list = list(current_edges)
        if len(edge_list) < 2:
            break
            
        e1, e2 = random.sample(edge_list, 2)
        
        # Check if they share a vertex?
        # If they share a vertex, the swap might not be valid or might create loops?
        # (a1, b1) and (a1, b2) -> (a1, b2) and (a1, b1) -> no change.
        # (a1, b1) and (a2, b1) -> (a1, b1) and (a2, b1) -> no change.
        # So they must be disjoint.
        
        v1, v2 = e1
        v3, v4 = e2
        if len({v1, v2, v3, v4}) < 4:
            continue
            
        # Identify a-part and b-part vertices
        # v1 in A, v2 in B. v3 in A, v4 in B.
        # Check which is which
        if v1 in part_a:
            a1, b1 = v1, v2
        else:
            a1, b1 = v2, v1
            
        if v3 in part_a:
            a2, b2 = v3, v4
        else:
            a2, b2 = v4, v3
            
        # New edges
        new_e1 = (min(a1, b2), max(a1, b2))
        new_e2 = (min(a2, b1), max(a2, b1))
        
        # Check if new edges already exist
        if new_e1 in current_edges or new_e2 in current_edges:
            continue
            
        # Apply swap
        current_edges.remove(e1)
        current_edges.remove(e2)
        current_edges.add(new_e1)
        current_edges.add(new_e2)
        
        edge_list = [[u, v] for u, v in current_edges]
        
        # Count distinct cycle lengths
        cnt = count_distinct_cycle_lengths(edge_list)
        
        if cnt < best_count:
            best_count = cnt
            best_edges = edge_list
            if best_count == 1:
                break
        else:
            # Revert swap (random walk)
            current_edges.remove(new_e1)
            current_edges.remove(new_e2)
            current_edges.add(e1)
            current_edges.add(e2)
            
            # With some probability, keep the change even if not better (simulated annealing style)
            # But for simplicity, just revert.
            pass

    # Ensure types are correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
