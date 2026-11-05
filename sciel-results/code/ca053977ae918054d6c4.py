import time
import random

def construct():
    start_time = time.time()
    time_limit = 120.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_vc(edges, n):
        if not edges:
            return set()
        
        # Greedy initial solution
        def greedy_vc(edges):
            cover = set()
            rem = set(edges)
            while rem:
                deg = {}
                for u, v in rem:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                if not deg: break
                best_v = max(deg, key=lambda x: deg[x])
                cover.add(best_v)
                rem = {(u, v) for u, v in rem if u != best_v and v != best_v}
            return cover

        best_cover = greedy_vc(edges)
        best_size = len(best_cover)
        
        # B&B
        def bnb(rem_edges, current_cover):
            nonlocal best_size, best_cover
            
            if time.time() - start_time > time_limit:
                return
                
            if not rem_edges:
                if len(current_cover) < best_size:
                    best_size = len(current_cover)
                    best_cover = set(current_cover)
                return
                
            if len(current_cover) >= best_size:
                return
            
            # Check if max degree <= 1
            deg = {}
            for u, v in rem_edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if deg and max(deg.values()) <= 1:
                # Graph is a matching
                # We must pick one vertex from each edge
                # To minimize size, we just pick one.
                # But we need to update current_cover.
                # Since edges are disjoint, we can just pick one endpoint arbitrarily.
                # This adds len(rem_edges) to the size.
                new_size = len(current_cover) + len(rem_edges)
                if new_size < best_size:
                    # Construct the cover
                    new_cover = set(current_cover)
                    for u, v in rem_edges:
                        new_cover.add(u) # Arbitrary choice
                    best_size = new_size
                    best_cover = new_cover
                return

            # Pick edge with max degree sum
            edge_list = list(rem_edges)
            best_edge = max(edge_list, key=lambda e: deg.get(e[0], 0) + deg.get(e[1], 0))
            u, v = best_edge
            
            # Branch u
            new_rem_1 = {(a, b) for a, b in rem_edges if a != u and b != u}
            bnb(new_rem_1, current_cover | {u})
            
            # Branch v
            new_rem_2 = {(a, b) for a, b in rem_edges if a != v and b != v}
            bnb(new_rem_2, current_cover | {v})

        try:
            bnb(set(edges), set())
        except Exception:
            pass
            
        return best_cover

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > time_limit - 5:
            edges = random_graph(n, p, seed)
            # Fallback to greedy
            cover = set()
            rem = set(edges)
            while rem:
                deg = {}
                for u, v in rem:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                if not deg: break
                best_v = max(deg, key=lambda x: deg[x])
                cover.add(best_v)
                rem = {(u, v) for u, v in rem if u != best_v and v != best_v}
            results.append(sorted(cover))
            continue
        
        edges = random_graph(n, p, seed)
        cover = solve_vc(edges, n)
        results.append(sorted(cover))
        
    return results
