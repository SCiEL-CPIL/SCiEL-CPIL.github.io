import random
import time

def construct():
    def min_degree_from_edges(edges, n):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg)

    def check_c4_and_min_deg(edges, n, target_deg):
        # Check C4-freeness and min degree
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree
        md = min(len(adj[i]) for i in range(n))
        if md < target_deg:
            return False, "min_deg"
        
        # Check C4
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False, "c4"
        return True, "ok"

    def try_dense_start(n, time_limit=25):
        target_deg = n - 39
        if target_deg <= 0:
            return {'n': n, 'edges': []}
        
        # Start with complete graph
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                edges.add((i, j))
        
        start_time = time.time()
        random.seed(n * 1000 + 42)
        
        while time.time() - start_time < time_limit:
            # Check if valid
            valid, reason = check_c4_and_min_deg(list(edges), n, target_deg)
            if valid:
                return {'n': n, 'edges': [list(e) for e in edges]}
            
            if reason == "min_deg":
                # Min degree too low, can't fix by removing edges
                break
            
            # Find a C4 and remove an edge
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            c4_found = False
            for u in range(n):
                for v in adj[u]:
                    if v <= u:
                        continue
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        common_list = list(common)
                        random.shuffle(common_list)
                        w1, w2 = common_list[0], common_list[1]
                        # C4: u-w1-v-w2-u
                        candidates = [(u, w1), (w1, v), (v, w2), (w2, u)]
                        # Normalize to sorted tuples
                        candidates = [tuple(sorted(c)) for c in candidates]
                        candidates = [c for c in candidates if c in edges]
                        if candidates:
                            e_to_remove = random.choice(candidates)
                            edges.remove(e_to_remove)
                            c4_found = True
                            break
                if c4_found:
                    break
            
            if not c4_found:
                break
        
        return None

    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        result = try_dense_start(n, time_limit=25)
        if result:
            return result
    
    # Fallback: PG(4,4) incidence graph, n=42
    q = 4
    # Generate points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # Generate lines (normals)
    lines = []
    for a in range(q):
        for b in range(q):
            lines.append((a, b, 1))
    for a in range(q):
        lines.append((a, 1, 0))
    lines.append((1, 0, 0))
    
    num_points = len(points)
    edges = []
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            if (p[0]*l[0] + p[1]*l[1] + p[2]*l[2]) % q == 0:
                edges.append([i, num_points + j])
    
    n = 2 * num_points
    return {'n': n, 'edges': edges}
