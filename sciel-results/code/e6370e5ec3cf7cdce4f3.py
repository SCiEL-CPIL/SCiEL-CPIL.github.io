import random
import time

def construct():
    target_n = 35
    time_limit = 115
    start_time = time.time()
    
    def random_graph(n):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def find_k4(adj, n):
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    if len(common) >= 2:
                        c_list = list(common)
                        return (i, j, c_list[0], c_list[1])
        return None

    def find_ind_6(adj, n):
        # Find independent set of size 6
        # Equivalent to K6 in complement
        def backtrack(candidates, current):
            if len(current) == 6:
                return current
            if len(candidates) < 6 - len(current):
                return None
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Check if v can be added to current
            # v must not be connected to any node in current in adj
            can_add = True
            for u in current:
                if u in adj[v]:
                    can_add = False
                    break
            
            if can_add:
                # Filter candidates: must not be connected to v
                new_cands = [u for u in rest if u not in adj[v]]
                res = backtrack(new_cands, current + [v])
                if res:
                    return res
            
            res = backtrack(rest, current)
            if res:
                return res
            return None

        return backtrack(list(range(n)), [])

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    best_result = None
    
    # Try to find a graph on target_n
    for restart in range(500):
        if time.time() - start_time > time_limit:
            break
            
        adj = random_graph(target_n)
        
        # Local search
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
                
            k4 = find_k4(adj, target_n)
            if k4:
                i, j, k, l = k4
                # Remove random edge from the K4
                edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                u, v = random.choice(edges_in_k4)
                adj[u].discard(v)
                adj[v].discard(u)
                continue
                
            ind6 = find_ind_6(adj, target_n)
            if ind6:
                # Add random edge to break independence
                u, v = random.sample(ind6, 2)
                adj[u].add(v)
                adj[v].add(u)
                continue
                
            # No violations found
            # Verify to be sure?
            # For now, assume valid
            edges = get_edges(adj, target_n)
            if best_result is None:
                best_result = {'n': target_n, 'edges': edges}
            break
        if best_result is not None:
            break

    if best_result is not None:
        return best_result
        
    # Fallback: Smaller n
    # Try n=20
    for n_fb in [20, 15, 10]:
        if time.time() - start_time > time_limit:
            break
        adj = random_graph(n_fb)
        # Quick check for small n
        valid = True
        # Check K4
        if find_k4(adj, n_fb):
            valid = False
        # Check Ind 6 (for n=20, ind 6 is possible)
        if valid and find_ind_6(adj, n_fb):
            valid = False
            
        if valid:
            return {'n': n_fb, 'edges': get_edges(adj, n_fb)}
            
    # Last resort: n=1
    return {'n': 1, 'edges': []}
