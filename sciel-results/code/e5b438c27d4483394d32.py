import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_degrees(edges):
        degs = defaultdict(int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        return degs

    def eulerian_path(edges, start=None):
        if not edges:
            return []
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        if start is None:
            odd_vertices = [v for v, d in get_degrees(edges).items() if d % 2 != 0]
            if odd_vertices:
                start = odd_vertices[0]
            else:
                start = next(iter(adj))
        
        path = [start]
        stack = [start]
        remaining = set(edges)
        
        while stack:
            v = stack[-1]
            next_v = None
            while adj[v]:
                u = adj[v].pop()
                e = tuple(sorted((v, u)))
                if e in remaining:
                    remaining.remove(e)
                    adj[u].remove(v)
                    next_v = u
                    break
            if next_v is not None:
                stack.append(next_v)
                path.append(next_v)
            else:
                stack.pop()
                
        return path

    def decompose_graph(edges):
        if not edges:
            return []
            
        degs = get_degrees(edges)
        odd_vertices = [v for v, d in degs.items() if d % 2 != 0]
        K = max(1, len(odd_vertices) // 2)
        
        if K == 1:
            return [eulerian_path(edges)]
            
        best_paths = None
        best_count = float('inf')
        
        rng = random.Random(42)
        
        for attempt in range(300):
            if time.time() > deadline:
                break
                
            current_adj = defaultdict(list)
            for u, v in edges:
                current_adj[u].append(v)
                current_adj[v].append(u)
                
            remaining_edges = set(edges)
            paths = []
            
            while remaining_edges:
                active_vertices = set()
                for u, v in remaining_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                    
                if not active_vertices:
                    break
                    
                start = rng.choice(list(active_vertices))
                path = [start]
                current = start
                visited = {start}
                
                while True:
                    avail = [nb for nb in current_adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in visited]
                    
                    if not avail:
                        break
                    
                    # Heuristic: Pick neighbor with highest remaining degree
                    avail.sort(key=lambda x: len([e for e in remaining_edges if x in e]), reverse=True)
                    
                    if len(avail) > 1 and rng.random() < 0.2:
                        next_v = rng.choice(avail[:3])
                    else:
                        next_v = avail[0]
                    
                    edge = tuple(sorted((current, next_v)))
                    remaining_edges.remove(edge)
                    
                    current_adj[current].remove(next_v)
                    current_adj[next_v].remove(current)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v
                    
                paths.append(path)
                
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                
                if best_count == K:
                    break
                    
        if best_paths is None:
            return [list(e) for e in edges]
            
        return best_paths

    def try_improve(paths, edges, max_iters=50):
        """Try to merge paths to reduce count."""
        if len(paths) <= 1:
            return paths
        
        # Build edge set for quick lookup
        edge_set = set(tuple(sorted(p)) for p in edges)
        
        # Build path index: for each path, store its edges and endpoints
        path_info = []
        for p in paths:
            if len(p) < 2:
                continue
            p_edges = set()
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                p_edges.add(e)
            path_info.append({
                'path': p,
                'edges': p_edges,
                'start': p[0],
                'end': p[-1],
                'len': len(p)
            })
        
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            merged = False
            for i in range(len(path_info)):
                for j in range(i + 1, len(path_info)):
                    if merged:
                        break
                    pi, pj = path_info[i], path_info[j]
                    # Check if we can merge: end of one matches start of other
                    # or start of one matches end of other, and no vertex overlap
                    # Simple case: pi.end == pj.start or pi.start == pj.end
                    # and no shared vertices
                    if set(pi['path']) & set(pj['path']):
                        continue
                    
                    new_path = None
                    if pi['end'] == pj['start']:
                        new_path = pi['path'] + pj['path'][1:]
                    elif pi['start'] == pj['end']:
                        new_path = list(reversed(pj['path'])) + pi['path'][1:]
                    elif pi['end'] == pj['end']:
                        new_path = pi['path'] + list(reversed(pj['path']))[1:]
                    elif pi['start'] == pj['start']:
                        new_path = list(reversed(pj['path'])) + pi['path'][1:]
                    
                    if new_path is None:
                        continue
                    
                    # Verify all edges exist and path is simple
                    if len(set(new_path)) != len(new_path):
                        continue
                    
                    valid = True
                    for k in range(len(new_path) - 1):
                        e = tuple(sorted((new_path[k], new_path[k+1])))
                        if e not in edge_set:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Merge
                    new_edges = set()
                    for k in range(len(new_path) - 1):
                        new_edges.add(tuple(sorted((new_path[k], new_path[k+1]))))
                    
                    path_info[i]['path'] = new_path
                    path_info[i]['edges'] = new_edges
                    path_info[i]['start'] = new_path[0]
                    path_info[i]['end'] = new_path[-1]
                    path_info[i]['len'] = len(new_path)
                    path_info.pop(j)
                    merged = True
                    improved = True
                    break
                
                if merged:
                    break
            
            if not merged:
                break
        
        if not improved:
            return paths
        
        return [p['path'] for p in path_info]

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose_graph(edges)
        # Try to improve by merging
        paths = try_improve(paths, edges)
        results.append(paths)
        
    return results
