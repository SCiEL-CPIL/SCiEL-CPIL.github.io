import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
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
        
        # Find all simple cycles using DFS
        # This is exponential, so we need to be careful
        # Use a smarter approach: for small graphs, enumerate cycles
        
        cycle_lengths = set()
        
        # Use networkx-style cycle enumeration, but we can't rely on it
        # Let's use a simple DFS-based approach for small graphs
        
        # Actually, for 12 vertices, we can use a bitset-based approach
        # or just enumerate paths
        
        # Simpler: use the fact that the graph is small
        # Find all cycles by checking all subsets? No, too slow.
        
        # Use DFS from each vertex, tracking visited
        # To avoid duplicates, only start cycles from the smallest vertex
        
        def find_cycles(start):
            cycles = []
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 3:
                        cycles.append(len(path))
                    elif neighbor not in visited and neighbor > start:
                        # Only visit vertices greater than start to avoid duplicates
                        # Actually, this doesn't work correctly for all cycles
                        # Let me use a different approach
                        pass
            return cycles
        
        # Better approach: use the fact that we can enumerate all simple cycles
        # using a standard algorithm. For 12 vertices, this is feasible.
        
        # Let's use a recursive DFS that finds all simple cycles
        # starting from each vertex, only counting cycles where the start
        # vertex is the smallest in the cycle.
        
        all_cycles = []
        
        def dfs_cycle(current, start, path, visited):
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    all_cycles.append(len(path))
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
        
        return len(set(all_cycles))
    
    # Random search for better graphs
    random.seed(42)
    
    for attempt in range(100000):
        if time.time() > deadline:
            break
        
        # Generate random 3-regular bipartite graph on 6+6 vertices
        part_a = list(range(6))
        part_b = list(range(6, 12))
        
        # Each vertex in part_a needs degree 3, each in part_b needs degree 3
        # This is a 3-regular bipartite graph, equivalent to a 3-regular
        # multigraph... no, it's a 3-regular bipartite simple graph.
        
        # Use configuration model: each vertex has 3 stubs
        # Part A stubs: 3*6 = 18
        # Part B stubs: 3*6 = 18
        # Pair them randomly
        
        a_stubs = [v for v in part_a for _ in range(3)]
        b_stubs = [v for v in part_b for _ in range(3)]
        
        random.shuffle(b_stubs)
        
        edges = set()
        valid = True
        for a, b in zip(a_stubs, b_stubs):
            if a == b:
                valid = False
                break
            pair = (min(a,b), max(a,b))
            if pair in edges:
                valid = False
                break
            edges.add(pair)
        
        if not valid or len(edges) != 18:
            continue
        
        edge_list = [[u, v] for u, v in edges]
        
        # Count distinct cycle lengths
        cnt = count_distinct_cycle_lengths(edge_list)
        
        if cnt < best_count:
            best_count = cnt
            best_edges = edge_list
            if best_count == 1:
                break
    
    # Ensure types are correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
