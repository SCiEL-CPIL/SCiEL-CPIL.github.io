import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Try n=39 first
    for n in [39, 38, 37, 36, 35, 30]:
        if time.time() > deadline:
            break
            
        # Try a few random instances for this n
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            random.seed(42 + attempt * 100 + n)
            
            adj = [set() for _ in range(n)]
            edges = set()
            
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                # Check C4
                # Intersect neighbors
                # Use the smaller set for iteration
                if len(adj[u]) < len(adj[v]):
                    common = False
                    for w in adj[u]:
                        if w in adj[v]:
                            common = True
                            break
                    if not common:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.add((u, v))
                else:
                    common = False
                    for w in adj[v]:
                        if w in adj[u]:
                            common = True
                            break
                    if not common:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.add((u, v))
                        
            # Check alpha < 11
            # We need to verify NO independent set of size 11
            # Use DFS with pruning
            
            def has_indep_set_k(adj, n, k):
                # adj is list of sets
                # Returns True if there is an independent set of size k
                
                # Greedy coloring bound
                def greedy_color(candidates):
                    # candidates is a set of vertices
                    # Returns number of colors
                    # Simple greedy
                    colors = {}
                    # Order by degree in candidates to improve bound?
                    # Sort by degree descending
                    verts = list(candidates)
                    verts.sort(key=lambda x: len(adj[x] & candidates), reverse=True)
                    for v in verts:
                        used = set()
                        for u in adj[v]:
                            if u in candidates and u in colors:
                                used.add(colors[u])
                        c = 0
                        while c in used:
                            c += 1
                        colors[v] = c
                    return max(colors.values()) + 1 if colors else 0

                def dfs(current_size, candidates):
                    if current_size >= k:
                        return True
                    if not candidates:
                        return False
                    if current_size + len(candidates) < k:
                        return False
                    
                    c = greedy_color(candidates)
                    if current_size + c < k:
                        return False
                    
                    # Pick vertex
                    # Heuristic: max degree
                    v = max(candidates, key=lambda x: len(adj[x] & candidates))
                    
                    # Include v
                    new_cand_incl = candidates - {v} - adj[v]
                    if dfs(current_size + 1, new_cand_incl):
                        return True
                    
                    # Exclude v
                    new_cand_excl = candidates - {v}
                    if dfs(current_size, new_cand_excl):
                        return True
                    
                    return False

                return dfs(0, set(range(n)))

            if not has_indep_set_k(adj, n, 11):
                # Valid
                edge_list = [[u, v] for u, v in edges]
                return {'n': n, 'edges': edge_list}
            
    # Fallback
    edges_seed = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges_seed += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges_seed}
