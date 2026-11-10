import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 12
    
    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([offset + i, offset + 3 + j])
        return edges
    
    # Fallback: two K3,3 components (score 2)
    best_edges = make_k33(0) + make_k33(6)
    best_score = 2
    
    def get_cycle_lengths(edges, n=12):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        def find_cycles_from(start):
            found = []
            def dfs(node, path, visited):
                for neighbor in sorted(adj[node]):
                    if neighbor == start and len(path) >= 4:
                        found.append(len(path))
                    elif neighbor not in visited and neighbor > start:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            dfs(start, [start], {start})
            return found
        
        for v in range(n):
            for c in find_cycles_from(v):
                cycle_lengths.add(c)
        
        return cycle_lengths
    
    def is_valid(edges, n=12):
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d < 3:
                return False
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # Check has cycle
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return True
            parent[px] = py
            return False
        
        has_cycle = False
        for u, v in edges:
            if union(u, v):
                has_cycle = True
                break
        
        return has_cycle
    
    # Enumerate all 3-regular bipartite graphs on 12 vertices (6+6)
    # Using 6x6 0-1 matrices with row/col sums 3
    # Generate via backtracking
    
    import itertools
    
    def generate_3reg_bipartite():
        """Generate all 6x6 0-1 matrices with row/col sums 3."""
        # Use backtracking on rows
        results = []
        
        def backtrack(row, col_sums, matrix):
            if row == 6:
                if all(c == 3 for c in col_sums):
                    results.append([row[:] for row in matrix])
                return
            
            # Try all combinations of 3 columns for this row
            for cols in itertools.combinations(range(6), 3):
                new_col_sums = col_sums[:]
                valid = True
                for c in cols:
                    new_col_sums[c] += 1
                    if new_col_sums[c] > 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Pruning: check if remaining rows can satisfy remaining col sums
                remaining_rows = 6 - row - 1
                can_complete = True
                for c in range(6):
                    remaining_needed = 3 - new_col_sums[c]
                    if remaining_needed < 0 or remaining_needed > remaining_rows:
                        can_complete = False
                        break
                if not can_complete:
                    continue
                
                matrix.append(list(cols))
                backtrack(row + 1, new_col_sums, matrix)
                matrix.pop()
        
        backtrack(0, [0]*6, [])
        return results
    
    # Try to find a graph with 1 distinct cycle length
    # First, try systematic enumeration of 3-regular bipartite graphs
    # But there might be too many, so use a smarter approach
    
    # Key insight: try specific structured graphs first
    # 1. K3,3 + K3,3 (already have, score 2)
    # 2. Try the "twisted" 3-regular bipartite graphs
    
    # Let's try a different structural approach:
    # Build graphs from 4-cycles and check if we can avoid 6-cycles
    
    # Actually, let's try: can we make a 3-regular bipartite graph where
    # all cycles are length 4? This would require the graph to be a union
    # of "4-cycle chains" but 3-regular means each vertex is in multiple cycles.
    
    # A 3-regular bipartite graph on 12 vertices with only 4-cycles would
    # need each vertex to be in exactly 1.5 cycles of length 4, which is
    # impossible. So we can't have ONLY 4-cycles.
    
    # What about ONLY 6-cycles? A 3-regular bipartite graph where all
    # cycles are length 6... The K3,3 has 4-cycles, so it doesn't work.
    
    # Let's try: is there a 3-regular bipartite graph on 12 vertices
    # with only 6-cycles? That would be a "6-cage" type structure.
    
    # Actually, let's just do a focused search:
    # Generate random 3-regular bipartite graphs and use local search
    
    import random
    random.seed(12345)
    
    def random_3reg_bipartite():
        """Generate a random 3-regular bipartite graph using configuration model."""
        # A vertices: 0-5, B vertices: 6-11
        # Each has 3 stubs
        a_stubs = [(i, s) for i in range(6) for s in range(3)]
        b_stubs = [(6+i, s) for i in range(6) for s in range(3)]
        
        random.shuffle(b_stubs)
        
        edges = set()
        valid = True
        for (a, _), (b, _) in zip(a_stubs, b_stubs):
            key = (min(a, b), max(a, b))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if not valid:
            return None
        
        return [list(e) for e in edges]
    
    def local_search(edges, max_iterations=10000):
        """Try to reduce distinct cycle lengths via edge swaps."""
        if not is_valid(edges, n):
            return edges, best_score
        
        cl = get_cycle_lengths(edges, n)
        current_score = len(cl)
        if current_score <= 1:
            return edges, current_score
        
        best_local = edges
        best_local_score = current_score
        
        # Build adjacency for quick checks
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick two random edges and try to swap
            if len(edges) < 4:
                break
            
            e1_idx = random.randint(0, len(edges) - 1)
            e2_idx = random.randint(0, len(edges) - 1)
            if e1_idx == e2_idx:
                continue
            
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Try swapping: remove (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)
            # or (u1,u2) and (v1,v2) - but must maintain bipartiteness
            # A = {0-5}, B = {6-11}
            
            def is_a(v):
                return v < 6
            
            # Only valid if new edges are between A and B
            candidates = []
            
            # Swap 1: (u1,v2) and (u2,v1)
            if is_a(u1) != is_a(v2) and is_a(u2) != is_a(v1):
                if u1 != v2 and u2 != v1:
                    candidates.append([(u1, v2), (u2, v1)])
            
            # Swap 2: (u1,u2) and (v1,v2) - only if one from A, one from B
            if is_a(u1) != is_a(u2) and is_a(v1) != is_a(v2):
                if u1 != u2 and v1 != v2:
                    candidates.append([(u1, u2), (v1, v2)])
            
            for new_edges_pair in candidates:
                # Check no duplicates
                edge_set = set(tuple(sorted(e)) for e in edges)
                new_edge_set = set()
                valid = True
                for ne in new_edges_pair:
                    key = tuple(sorted(ne))
                    if key in new_edge_set:
                        valid = False
                        break
                    new_edge_set.add(key)
                
                if not valid:
                    continue
                
                # Check not already in graph
                for ne in new_edges_pair:
                    key = tuple(sorted(ne))
                    if key in edge_set:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Build new edge list
                new_edges = [e for i, e in enumerate(edges) if i not in (e1_idx, e2_idx)]
                for ne in new_edges_pair:
                    new_edges.append(list(ne))
                
                if not is_valid(new_edges, n):
                    continue
                
                new_cl = get_cycle_lengths(new_edges, n)
                new_score = len(new_cl)
                
                if new_score < best_local_score:
                    best_local = new_edges
                    best_local_score = new_score
                    edges = new_edges
                    if new_score == 1:
                        return edges, new_score
        
        return best_local, best_local_score
    
    # Phase 1: Try random 3-regular bipartite graphs with local search
    for attempt in range(500):
        if time.time() > deadline - 15:
            break
        
        edges = random_3reg_bipartite()
        if edges is None:
            continue
        
        if not is_valid(edges, n):
            continue
        
        cl = get_cycle_lengths(edges, n)
        if not cl:
            continue
        
        score = len(cl)
        if score < best_score:
            best_score = score
            best_edges = edges
            if score == 1:
                break
        
        # Local search to improve
        improved_edges, improved_score = local_search(edges, max_iterations=5000)
        if improved_score < best_score:
            best_score = improved_score
            best_edges = improved_edges
            if score == 1:
                break
    
    # Phase 2: If still at 2, try systematic enumeration of specific structures
    if best_score > 1 and time.time() < deadline - 20:
        # Try all 3-regular bipartite graphs via backtracking (limited)
        # Generate a subset by fixing first few rows
        
        import itertools
        
        # Try specific structured graphs:
        # Graph where A[i] connects to B[i], B[i+1], B[i+2] mod 6 (cyclic)
        # This is a specific 3-regular bipartite graph
        
        structured_graphs = []
        
        # Cyclic graph: A[i] -> B[i], B[i+1], B[i+2] mod 6
        edges_cyclic = []
        for i in range(6):
            for j in range(3):
                edges_cyclic.append([i, 6 + (i + j) % 6])
        if is_valid(edges_cyclic, n):
            cl = get_cycle_lengths(edges_cyclic, n)
            if cl:
                score = len(cl)
                if score < best_score:
                    best_score = score
                    best_edges = edges_cyclic
        
        # Try all permutations of the cyclic pattern
        for perm in itertools.permutations(range(6)):
            if time.time() > deadline - 10:
                break
            
            edges_perm = []
            edge_set = set()
            valid = True
            for i in range(6):
                for j in range(3):
                    a = i
                    b = 6 + perm[(i + j) % 6]
                    key = (min(a, b), max(a, b))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                    edges_perm.append([a, b])
                if not valid:
                    break
            
            if not valid:
                continue
            
            if not is_valid(edges_perm, n):
                continue
            
            cl = get_cycle_lengths(edges_perm, n)
            if not cl:
                continue
            
            score = len(cl)
            if score < best_score:
                best_score = score
                best_edges = edges_perm
                if score == 1:
                    break
    
    # Final: ensure types are correct
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
    
    return result
