"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _get_adj(edges, n=12):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_cycle(adj, n=12):
    visited = [False] * n
    for start in range(n):
        if not visited[start]:
            queue = deque([(start, -1)])
            visited[start] = True
            while queue:
                node, parent = queue.popleft()
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, node))
                    elif neighbor != parent:
                        return True
    return False

def _count_distinct_cycle_lengths(adj, n=12):
    """Count distinct simple cycle lengths using DFS from each vertex as start."""
    lengths = set()
    
    for start in range(n):
        # DFS finding cycles that include 'start' as the smallest vertex
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    lengths.add(len(path))
                elif neighbor > start and neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    
    return len(lengths) if lengths else 0

def construct():
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Fallback: two disjoint K_{3,3} -> cycle lengths {4, 6}, raw value 2
    def make_two_k33():
        edges = []
        for u in [0,1,2]:
            for v in [3,4,5]:
                edges.append([u, v])
        for u in [6,7,8]:
            for v in [9,10,11]:
                edges.append([u, v])
        return edges
    
    fallback_edges = make_two_k33()
    fallback_adj = _get_adj(fallback_edges)
    fallback_distinct = _count_distinct_cycle_lengths(fallback_adj)
    
    best_edges = fallback_edges
    best_distinct = fallback_distinct
    
    # Search for bipartite graphs with all cycles same length
    # Key: in bipartite graphs, all cycles are even length.
    # Target: all cycles length 4 (raw value 1).
    
    import random
    random.seed(12345)
    
    # Try targeted constructions first
    # Construction: 4-cycles arranged so no 6-cycles form
    # Use bipartition A={0,1,2,3}, B={4,5,6,7,8,9,10,11}
    # Each A vertex connects to exactly 3 B vertices (degree 3)
    # Each B vertex connects to at least 3 A vertices
    # Total edges = 4*3 = 12, but need B vertices to have deg >= 3
    # 12 edges among 8 B vertices: avg degree 12*2/8 = 3. OK if uniform.
    
    # Try: A={0,1,2,3}, B={4,5,6,7,8,9,10,11}
    # Edges: 0-{4,5,6}, 1-{5,6,7}, 2-{6,7,8}, 3-{7,8,9}
    # B degrees: 4:1, 5:2, 6:3, 7:3, 8:2, 9:1, 10:0, 11:0 -> fails
    
    # Need better distribution. Let me try 4 A vertices each with degree 4:
    # 16 edges, B avg degree = 16*2/8 = 4
    # 0-{4,5,6,7}, 1-{4,5,8,9}, 2-{6,7,10,11}, 3-{8,9,10,11}
    # B degrees: 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:2 -> all < 3, fails
    
    # Try 5 A vertices, 7 B vertices:
    # 5*3=15 min edges, 7*3=21 min edges -> need 21 edges
    # Max edges = 5*7 = 35. OK.
    # But this is getting complex. Let me just do random search.
    
    trials = 0
    while time.time() < deadline and trials < 20000:
        trials += 1
        
        a_size = random.choice([3, 4, 5, 6])
        b_size = 12 - a_size
        
        if max(a_size, b_size) * 3 > a_size * b_size:
            continue
        
        part_a = list(range(a_size))
        part_b = list(range(a_size, 12))
        
        # Generate random bipartite graph with controlled edge density
        # Target: each vertex degree around 3-5
        target_edges = max(a_size * 3, b_size * 3) + random.randint(0, 5)
        target_edges = min(target_edges, a_size * b_size)
        
        # Random edge selection
        all_possible = [(u, v) for u in part_a for v in part_b]
        random.shuffle(all_possible)
        edges = [[u, v] for u, v in all_possible[:target_edges]]
        
        adj = _get_adj(edges)
        
        # Check min degree
        if any(len(adj[i]) < 3 for i in range(12)):
            continue
        
        # Check has cycle
        if not _has_cycle(adj):
            continue
        
        # Count distinct cycle lengths
        distinct = _count_distinct_cycle_lengths(adj)
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_edges = edges[:]
            if distinct == 1:
                break
    
    # Final validation and format
    adj_final = _get_adj(best_edges)
    
    # Ensure types are correct
    result = []
    seen = set()
    for u, v in best_edges:
        u_int = int(u)
        v_int = int(v)
        key = (min(u_int, v_int), max(u_int, v_int))
        if key not in seen:
            seen.add(key)
            result.append([u_int, v_int])
    
    result.sort()
    return result

# EVOLVE-BLOCK-END
