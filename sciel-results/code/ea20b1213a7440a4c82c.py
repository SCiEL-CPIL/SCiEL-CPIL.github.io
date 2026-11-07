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
    
    # Fast cycle counting for bipartite graphs using BFS from each vertex
    # For a bipartite graph, all cycles are even-length
    # We use the standard approach: for each vertex as root, do BFS and count paths
    # that form cycles back to root
    
    def count_distinct_cycle_lengths_fast(adj):
        n = 12
        cycle_lengths = set()
        
        # For each vertex, find all simple cycles containing it
        # by doing DFS with the constraint that we only go to vertices > start
        # (except for closing back to start)
        
        def dfs(current, start, depth, visited_mask):
            for neighbor in adj[current]:
                if neighbor == start:
                    if depth >= 2:  # cycle length = depth + 1 >= 3, but bipartite so >= 4
                        cycle_lengths.add(depth + 1)
                elif neighbor > start and not (visited_mask >> neighbor) & 1:
                    dfs(neighbor, start, depth + 1, visited_mask | (1 << neighbor))
        
        for start in range(n):
            visited_mask = 1 << start
            dfs(start, start, 0, visited_mask)
        
        return len(cycle_lengths)
    
    # Build adjacency from edge set
    def build_adj(edge_set):
        adj = [set() for _ in range(12)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Local search from the known good graph
    random.seed(42)
    
    # Convert best_edges to a set for easy modification
    current_edges = set(tuple(sorted(e)) for e in best_edges)
    
    # Partitions for bipartite graph
    part_a = frozenset(range(6))
    part_b = frozenset(range(6, 12))
    
    # Precompute adjacency for current graph
    current_adj = build_adj(current_edges)
    
    for attempt in range(500000):
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
        
        # Check if they share a vertex
        v1, v2 = e1
        v3, v4 = e2
        if len({v1, v2, v3, v4}) < 4:
            continue
            
        # Identify a-part and b-part vertices
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
            
        # Apply swap to edge set
        current_edges.remove(e1)
        current_edges.remove(e2)
        current_edges.add(new_e1)
        current_edges.add(new_e2)
        
        # Update adjacency incrementally
        adj = [set(s) for s in current_adj]  # copy
        adj[a1].discard(b1)
        adj[b1].discard(a1)
        adj[a2].discard(b2)
        adj[b2].discard(a2)
        adj[a1].add(b2)
        adj[b2].add(a1)
        adj[a2].add(b1)
        adj[b1].add(a2)
        
        # Count distinct cycle lengths
        cnt = count_distinct_cycle_lengths_fast(adj)
        
        if cnt < best_count:
            best_count = cnt
            best_edges = [[u, v] for u, v in current_edges]
            current_adj = adj
            if best_count == 1:
                break
        else:
            # Revert swap
            current_edges.remove(new_e1)
            current_edges.remove(new_e2)
            current_edges.add(e1)
            current_edges.add(e2)
            current_adj = build_adj(current_edges)

    # Ensure types are correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
