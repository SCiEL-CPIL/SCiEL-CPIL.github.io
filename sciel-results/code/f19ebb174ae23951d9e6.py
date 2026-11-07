import random
import time

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
    
    # Function to count distinct cycle lengths - fixed and optimized
    def count_distinct_cycle_lengths(edges):
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all simple cycles where start is the minimum vertex
        all_cycle_lengths = set()
        
        def dfs_cycle(current, start, path_len, visited):
            for neighbor in adj[current]:
                if neighbor == start and path_len >= 3:
                    all_cycle_lengths.add(path_len)
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    dfs_cycle(neighbor, start, path_len + 1, visited)
                    visited.remove(neighbor)
        
        for start in range(n):
            visited = {start}
            dfs_cycle(start, start, 1, visited)
        
        return len(all_cycle_lengths)
    
    # Strategy: Try Cartesian Product C3 x K4 (Icosahedron)
    # This is 5-regular, triangle-free.
    # Vertices (i, j) mapped to i * 4 + j
    def make_c3_k4():
        edges = []
        # C3 edges: 0-1, 1-2, 2-0
        c3_edges = [(0, 1), (1, 2), (2, 0)]
        # K4 edges: all pairs in 0..3
        k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        
        for i in range(3):
            for j in range(4):
                u = i * 4 + j
                # Connect via C3
                for k in (i+1)%3, (i-1)%3:
                    v = k * 4 + j
                    if u < v:
                        edges.append([u, v])
                    # Note: adding duplicates is handled by set later or check
                # Connect via K4
                for l in range(4):
                    if l != j:
                        v = i * 4 + l
                        if u < v:
                            edges.append([u, v])
        
        # Remove duplicates
        unique_edges = set()
        for e in edges:
            unique_edges.add((min(e[0], e[1]), max(e[0], e[1])))
        return [[u, v] for u, v in unique_edges]
    
    ico_edges = make_c3_k4()
    ico_count = count_distinct_cycle_lengths(ico_edges)
    if ico_count < best_count:
        best_count = ico_count
        best_edges = ico_edges

    # Random search for better graphs
    random.seed(42)
    
    # Strategy 1: Random 3-regular bipartite graphs on 6+6
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        a_stubs = [v for v in range(6) for _ in range(3)]
        b_stubs = [v for v in range(6, 12) for _ in range(3)]
        
        random.shuffle(b_stubs)
        
        edges = set()
        valid = True
        for a, b in zip(a_stubs, b_stubs):
            pair = (min(a,b), max(a,b))
            if pair in edges:
                valid = False
                break
            edges.add(pair)
        
        if not valid or len(edges) != 18:
            continue
        
        edge_list = [[u, v] for u, v in edges]
        
        # Quick degree check
        deg = [0] * 12
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < 3:
            continue
        
        cnt = count_distinct_cycle_lengths(edge_list)
        
        if cnt < best_count:
            best_count = cnt
            best_edges = edge_list
            if best_count == 1:
                break
    
    # Strategy 2: Non-bipartite triangle-free 3-regular graphs
    if best_count > 1 and time.time() < deadline - 10:
        for attempt in range(20000):
            if time.time() > deadline - 5:
                break
            
            stubs = [v for v in range(12) for _ in range(3)]
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, 36, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                pair = (min(u,v), max(u,v))
                if pair in edges:
                    valid = False
                    break
                edges.add(pair)
            
            if not valid or len(edges) != 18:
                continue
            
            edge_list = [[u, v] for u, v in edges]
            
            # Check triangle-free
            adj = [set() for _ in range(12)]
            for u, v in edge_list:
                adj[u].add(v)
                adj[v].add(u)
            
            has_triangle = False
            for u in range(12):
                if has_triangle:
                    break
                for v in adj[u]:
                    if v > u:
                        common = adj[u] & adj[v]
                        if any(w > v for w in common):
                            has_triangle = True
                            break
            
            if has_triangle:
                continue
            
            deg = [len(adj[u]) for u in range(12)]
            if min(deg) < 3:
                continue
            
            cnt = count_distinct_cycle_lengths(edge_list)
            
            if cnt < best_count:
                best_count = cnt
                best_edges = edge_list
                if best_count == 1:
                    break
    
    # Strategy 3: Bipartite with varying degrees
    if best_count > 1 and time.time() < deadline - 5:
        for attempt in range(15000):
            if time.time() > deadline - 2:
                break
            
            part_a = list(range(6))
            part_b = list(range(6, 12))
            
            # Start with each vertex having degree 3, then add extra edges
            edges = set()
            deg_a = [0] * 6
            deg_b = [0] * 6
            
            for i in range(6):
                targets = random.sample(range(6), 3)
                for t in targets:
                    edges.add((i, 6+t))
                    deg_a[i] += 1
                    deg_b[t] += 1
            
            # Add random extra edges
            extra = random.randint(0, 8)
            for _ in range(extra):
                i = random.randint(0, 5)
                t = random.randint(0, 5)
                if (i, 6+t) not in edges:
                    edges.add((i, 6+t))
                    deg_a[i] += 1
                    deg_b[t] += 1
            
            if min(deg_a) < 3 or min(deg_b) < 3:
                continue
            
            edge_list = [[u, v] for u, v in edges]
            
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
