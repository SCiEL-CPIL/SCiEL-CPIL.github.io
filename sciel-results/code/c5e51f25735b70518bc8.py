import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def max_clique_size(comp_adj):
        """
        Finds the size of the maximum clique in a graph represented by adjacency sets.
        Uses a branch and bound approach with greedy coloring for upper bounds.
        """
        n = len(comp_adj)
        if n == 0:
            return 0
        
        # Order vertices by degree (descending) to improve pruning
        # We work with indices 0..n-1
        # To make coloring easier, we can reorder vertices.
        # But reordering adj sets is expensive.
        # We can just pick vertices in a heuristic order.
        
        best = 0
        
        # P: set of candidate vertices
        # X: set of vertices not in P (already processed or pruned)
        # size: current clique size
        
        def color_sort(P):
            """
            Greedily color P and return (order, color_bounds).
            order: list of vertices in P, sorted by color.
            color_bounds: list where color_bounds[i] is the max color for vertices in order[:i+1].
            """
            if not P:
                return [], []
            
            # Simple greedy coloring
            # We want to find an upper bound on clique size in P.
            # A valid coloring of P gives an upper bound on clique size.
            # We want a "tight" coloring.
            
            # Standard greedy: pick vertex, assign smallest color not used by neighbors in P.
            # To make it tight, we should pick vertices with high degree in P first?
            # Or just iterate.
            
            colors = {}
            color_count = 0
            order = []
            
            # Sort P by degree in P descending for better coloring?
            P_list = sorted(P, key=lambda x: len(comp_adj[x] & P), reverse=True)
            
            for v in P_list:
                # Find smallest color
                used = set()
                for u in P_list:
                    if u in comp_adj[v] and u in colors:
                        used.add(colors[u])
                        
                c = 0
                while c in used:
                    c += 1
                colors[v] = c
                if c + 1 > color_count:
                    color_count = c + 1
                order.append(v)
                
            # Now, we need to sort order such that colors are non-decreasing?
            # Tomita's algorithm uses a specific ordering.
            # The bound for a vertex at index i in the ordered list is colors[order[i]].
            # We want to process vertices with high color bounds first?
            # Actually, the standard implementation processes vertices in reverse order of coloring.
            
            # Let's just return the max color as a simple bound for the whole set?
            # No, we need per-vertex bounds for pruning.
            
            # Simplified: Just return the max color as the bound for the set.
            # This is weaker but simpler.
            return color_count

        def dfs(P, size):
            nonlocal best
            if size + len(P) <= best:
                return
            if size >= 11: # We only care if >= 11
                best = 11
                return
            if not P:
                return
                
            # Upper bound: size + len(P)
            # Better bound: size + max_clique_in_P
            # Use greedy coloring bound
            
            # For speed, let's just use len(P) as bound for now, 
            # and rely on the fact that n=43 is small.
            # If this is too slow, I'll add coloring.
            
            # Pick a vertex v from P
            # Heuristic: pick v with max degree in P
            v = max(P, key=lambda x: len(comp_adj[x] & P))
            
            new_P = P & comp_adj[v]
            
            dfs(new_P, size + 1)
            
            P.remove(v)
            if size + len(P) <= best:
                return
            
            # Recurse on remaining P without v
            dfs(P, size)

        dfs(set(range(n)), 0)
        return best

    def has_independent_set_of_size_k(adj, k):
        n = len(adj)
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
        
        return max_clique_size(comp_adj) >= k

    def generate_c4_free(n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            safe = True
            for w in adj[v]:
                if adj[u] & adj[w]:
                    safe = False
                    break
            if not safe:
                continue
            
            for w in adj[u]:
                if adj[v] & adj[w]:
                    safe = False
                    break
            if not safe:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
        return adj, edges

    def is_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return False
        return True

    # Try to find a valid graph
    for n in [43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        for seed in range(20):
            if time.time() > deadline:
                break
                
            adj, edges = generate_c4_free(n, seed)
            
            if not is_c4_free(adj):
                continue
                
            if not has_independent_set_of_size_k(adj, 11):
                return {'n': n, 'edges': edges}

    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
