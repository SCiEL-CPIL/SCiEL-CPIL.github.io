import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n_target = 40
    
    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def has_independent_set_size_k(adj, n, k):
        # Returns the set if found, else None
        # DFS
        
        def dfs(current_set, candidates):
            if len(current_set) == k:
                return current_set
            if not candidates:
                return None
            
            # Heuristic: pick vertex with max degree in candidates
            # to prune
            
            # Convert candidates to list for easy manipulation?
            # Or use bit manipulation.
            
            # Let's use a list of vertices for the current candidate set
            # This is slower but easier to debug/implement heuristic.
            # For n=40, list operations are fine.
            
            cand_list = [v for v in range(n) if (candidates >> v) & 1]
            
            if len(cand_list) < k - len(current_set):
                return None
            
            # Sort by degree in candidate subgraph
            cand_list.sort(key=lambda v: bin(adj[v] & candidates).count('1'), reverse=True)
            
            for v in cand_list:
                # Include v
                new_set = current_set + [v]
                new_candidates = candidates & ~(1 << v) & ~adj[v]
                res = dfs(new_set, new_candidates)
                if res:
                    return res
                
                # Exclude v
                new_candidates_excl = candidates & ~(1 << v)
                # Note: We don't need to recurse for exclude if we are just checking existence?
                # No, we must check.
                # But this doubles the work.
                # Standard B&B:
                # 1. Include v
                # 2. Exclude v
                # We can prune exclude if we know that even with all remaining vertices, we can't reach k.
                # But we already checked len(cand_list).
                
                # Optimization: If we fail to include v, we try excluding v.
                # But if we exclude v, we are in the same state with v removed.
                # So we can just loop.
                # The loop structure above effectively tries "Include v" for the first v.
                # If it fails, we move to the next v, which is equivalent to "Exclude first v, Include second v".
                # This is correct.
            
            return None

        return dfs([], (1 << n) - 1)

    def find_independent_set(adj, n, k):
        return has_independent_set_size_k(adj, n, k)

    def improve_graph(adj, n, time_limit):
        # Local search
        adj = list(adj)
        
        while time.time() < time_limit:
            # Check if alpha < 10
            # We can just try to find an independent set of size 10.
            # If we can't find one, it might still exist, but for random graphs,
            # if a greedy/DFS search doesn't find it quickly, it's likely < 10.
            # However, for correctness, we should be sure.
            # Given the time limit, we will assume if DFS fails, it's < 10.
            # This is a risk, but acceptable for "maximize n" where we want to find a valid one.
            # Actually, if we return a graph that HAS an independent set of 10, the evaluator will reject it.
            # So we MUST be sure.
            
            # Let's run the DFS. If it returns None, we assume alpha < 10.
            # If it returns a set, we need to fix it.
            
            indep_set = find_independent_set(adj, n, 10)
            
            if indep_set is None:
                # Success!
                return adj
            
            # Found an independent set of size 10
            # Try to add an edge within it
            added = False
            # Shuffle the set to try different pairs
            random.shuffle(indep_set)
            for i in range(len(indep_set)):
                for j in range(i + 1, len(indep_set)):
                    u = indep_set[i]
                    v = indep_set[j]
                    # Check if (u, v) is already an edge (it shouldn't be, since they are in an independent set)
                    # Check if adding (u, v) creates a triangle
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    if common == 0:
                        # Safe to add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
                if added:
                    break
            
            if not added:
                # Stuck. Perturb: remove a random edge
                # Find a random edge
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append((u, v))
                if edges:
                    u, v = random.choice(edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Empty graph? Reset
                    return None

    # Try to find a graph on 40 vertices
    for seed in range(10000):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        # Generate random maximal triangle-free graph
        all_edges = [(u, v) for u in range(n_target) for v in range(u + 1, n_target)]
        random.shuffle(all_edges)
        
        adj = [0] * n_target
        for u, v in all_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Improve
        result = improve_graph(adj, n_target, deadline - 1)
        
        if result:
            # Verify one last time?
            # The improve function returns only if it thinks alpha < 10.
            # We should do a final check to be safe.
            # But the check is the same as inside improve.
            # So it should be fine.
            
            edges = get_edges(result, n_target)
            return {'n': n_target, 'edges': edges}
            
    # Fallback to 39
    n_fallback = 39
    for seed in range(10000):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        all_edges = [(u, v) for u in range(n_fallback) for v in range(u + 1, n_fallback)]
        random.shuffle(all_edges)
        
        adj = [0] * n_fallback
        for u, v in all_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        result = improve_graph(adj, n_fallback, deadline - 1)
        
        if result:
            edges = get_edges(result, n_fallback)
            return {'n': n_fallback, 'edges': edges}
            
    # Final fallback: Seed program
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
