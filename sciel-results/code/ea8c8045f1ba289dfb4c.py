"""Find dense triangle-free 4-chromatic graph on 16 vertices."""
import time
import random

def construct():
    deadline = time.time() + 100  # safety margin
    
    N = 16
    
    def time_ok():
        return time.time() < deadline
    
    def build_adj(edges):
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(N):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    def is_3_colorable(adj):
        """Backtracking 3-coloring with vertex ordering by degree."""
        color = [-1] * N
        order = sorted(range(N), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Collect forbidden colors
            forbidden = set()
            for u in adj[v]:
                if color[u] != -1:
                    forbidden.add(color[u])
            for c in range(3):
                if c not in forbidden:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_set_from_list(edges):
        s = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            s.add((u, v))
        return s
    
    def make_edge_list(eset):
        return [[u, v] for u, v in eset]
    
    # Start with Grötzsch graph on vertices 0-10
    # Grötzsch: Mycielski of C5
    # C5: 0-1-2-3-4-0
    # Mycielski adds vertices 5-9 (copies) and vertex 10 (apex)
    grötzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [0, 6], [1, 5], [2, 6], [3, 7], [4, 8],  # cross edges (vi, wj)
        [1, 6], [2, 7], [3, 8], [4, 9], [0, 5],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10],  # apex to w-vertices
    ]
    
    # Verify Grötzsch is triangle-free and 4-chromatic
    g_adj = build_adj(grötzsch_edges)
    # Extend to 16 vertices: 11-15 start isolated
    best_edges = set()
    for e in grötzsch_edges:
        u, v = e
        if u > v:
            u, v = v, u
        best_edges.add((u, v))
    
    best_count = len(best_edges)
    best_result = make_edge_list(best_edges)
    
    # Verify initial state
    adj = build_adj(best_result)
    assert is_triangle_free(adj), "Initial graph not triangle-free"
    assert not is_3_colorable(adj), "Initial graph is 3-colorable"
    
    # Local search: try to add edges
    def try_add_edges(edge_set, max_time=80):
        """Greedy + randomized edge addition."""
        edges = list(edge_set)
        adj = build_adj(edges)
        count = len(edges)
        start_time = time.time()
        
        while time.time() < start_time + max_time:
            # Get all missing edges
            all_possible = [(u, v) for u in range(N) for v in range(u+1, N)]
            missing = [(u, v) for (u, v) in all_possible if (u, v) not in edge_set]
            
            if not missing:
                break
            
            random.shuffle(missing)
            
            added_any = False
            for u, v in missing:
                if time.time() >= start_time + max_time:
                    break
                # Check triangle-free: no common neighbor of u and v
                common = adj[u] & adj[v]
                if common:
                    continue
                # Try adding edge
                adj[u].add(v)
                adj[v].add(u)
                # Check 4-chromaticity
                if not is_3_colorable(adj):
                    # Good, keep it
                    edge_set.add((u, v))
                    count += 1
                    added_any = True
                else:
                    # Remove it
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            if not added_any:
                # Stuck: try removing a random edge and re-adding
                if edges:
                    random.shuffle(edges)
                    for rem_u, rem_v in edges[:3]:  # try removing up to 3
                        if time.time() >= start_time + max_time:
                            break
                        key = (min(rem_u, rem_v), max(rem_u, rem_v))
                        if key not in edge_set:
                            continue
                        edge_set.discard(key)
                        adj[rem_u].discard(rem_v)
                        adj[rem_v].discard(rem_u)
                        count -= 1
                        # Now try re-adding all missing edges
                        missing2 = [(a, b) for a in range(N) for b in range(a+1, N) if (a, b) not in edge_set]
                        random.shuffle(missing2)
                        for a, b in missing2:
                            if time.time() >= start_time + max_time:
                                break
                            if adj[a] & adj[b]:
                                continue
                            adj[a].add(b)
                            adj[b].add(a)
                            if not is_3_colorable(adj):
                                edge_set.add((a, b))
                                count += 1
                            else:
                                adj[a].discard(b)
                                adj[b].discard(a)
                        break  # Only try one removal per iteration
        
        return edge_set, count
    
    # Multiple restarts with different initial extensions
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        # Start fresh from Grötzsch
        cur_edges = set(best_edges)
        
        # Randomly connect isolated vertices 11-15 to some vertices
        # Try different random connections
        for vi in range(11, 16):
            if time.time() >= deadline:
                break
            # Try connecting to a random subset of non-adjacent vertices
            candidates = [x for x in range(11) if x != vi]
            random.shuffle(candidates)
            for target in candidates:
                if time.time() >= deadline:
                    break
                key = (min(vi, target), max(vi, target))
                if key in cur_edges:
                    continue
                # Check triangle-free
                adj_tmp = build_adj(make_edge_list(cur_edges))
                if adj_tmp[vi] & adj_tmp[target]:
                    continue
                adj_tmp[vi].add(target)
                adj_tmp[target].add(vi)
                if not is_3_colorable(adj_tmp):
                    cur_edges.add(key)
        
        # Now run local search
        cur_edges, cur_count = try_add_edges(cur_edges, max_time=15)
        
        if cur_count > best_count:
            best_count = cur_count
            best_result = make_edge_list(cur_edges)
    
    # Final verification
    final_adj = build_adj(best_result)
    if not is_triangle_free(final_adj):
        # Fallback to Grötzsch
        return make_edge_list(best_edges)
    if is_3_colorable(final_adj):
        return make_edge_list(best_edges)
    
    return best_result
