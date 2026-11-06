import random
import time
import sys
import math
import numpy as np

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    def is_k4_free(graph_mask):
        for u in range(N):
            for v in range(u+1, N):
                if (graph_mask[u] >> v) & 1:
                    common = graph_mask[u] & graph_mask[v]
                    c = common
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            return False
        return True
    
    def greedy_triangle_free_upper_bound(graph_mask):
        """Fast greedy upper bound on max triangle-free induced subgraph size.
        Returns an upper bound (>= true max) that is much faster to compute."""
        # Greedy: repeatedly pick the vertex that minimizes triangles added
        # This gives a lower bound on the max triangle-free set, not upper.
        # For upper bound: use the fact that any triangle-free set can have at most
        # vertices where each vertex has limited neighbors in the set.
        
        # Better: use a simple greedy to find a large triangle-free set (lower bound),
        # and use remaining vertices as upper bound heuristic.
        
        # Actually, for pruning in local search, we want a quick estimate.
        # Let's use: start with all vertices, remove vertices from triangles greedily.
        # The number removed + remaining gives an upper bound on triangle-free size? No.
        
        # Simplest fast bound: the max triangle-free size is at most the size of
        # the largest independent set in the "triangle conflict" graph, but that's complex.
        
        # For now, use a quick greedy to find a triangle-free set (lower bound),
        # and use N - (min vertex cover of triangles) as a rough upper bound.
        
        # Quick approach: find max triangle-free set greedily (this is a lower bound)
        # and also try to find a small triangle transversal (upper bound = N - transversal_size)
        
        # Find a triangle transversal (set of vertices hitting all triangles) greedily
        # Upper bound on triangle-free size = N - min_transversal_size
        
        # Precompute all triangles
        triangles = []
        for a in range(N):
            for b in range(a+1, N):
                if (graph_mask[a] >> b) & 1:
                    common = graph_mask[a] & graph_mask[b] & ~((1 << a) | (1 << b))
                    c = common
                    while c:
                        lsb = c & (-c)
                        d = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[a] >> d) & 1 and (graph_mask[b] >> d) & 1:
                            triangles.append((a, b, d))
        
        if not triangles:
            return N
        
        # Greedy set cover for triangles
        vertex_triangle_count = [0] * N
        for tri in triangles:
            for v in tri:
                vertex_triangle_count[v] += 1
        
        transversal = 0
        remaining_tris = set(triangles)
        while remaining_tris:
            # Pick vertex that hits most remaining triangles
            best_v = -1
            best_count = -1
            for v in range(N):
                if (transversal >> v) & 1:
                    continue
                count = 0
                for tri in remaining_tris:
                    if v in tri:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            transversal |= (1 << best_v)
            new_remaining = set()
            for tri in remaining_tris:
                if best_v not in tri:
                    new_remaining.add(tri)
            remaining_tris = new_remaining
        
        # Upper bound: N - transversal_size
        return N - transversal.bit_count()
    
    def max_triangle_free_size(graph_mask, use_greedy_prune=True):
        # Find the largest subset of vertices that induces a triangle-free graph.
        # Use branch and bound with bitmasks.
        
        # Precompute common neighbors for all pairs
        common_neighbors = [[0]*N for _ in range(N)]
        for u in range(N):
            for v in range(u+1, N):
                if (graph_mask[u] >> v) & 1:
                    common_neighbors[u][v] = graph_mask[u] & graph_mask[v]
                    common_neighbors[v][u] = common_neighbors[u][v]
        
        best = 0
        
        # Get a quick lower bound from greedy
        def greedy_lower_bound():
            # Greedy: start with empty set, add vertices that don't create triangles
            current = 0
            remaining = (1 << N) - 1
            while remaining:
                # Pick vertex with minimum degree in remaining
                best_v = -1
                best_deg = N + 1
                c = remaining
                while c:
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = (graph_mask[v] & remaining).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                # Check if adding best_v keeps triangle-free
                valid = True
                c2 = current
                while c2:
                    lsb2 = c2 & (-c2)
                    u = lsb2.bit_length() - 1
                    c2 ^= lsb2
                    if (graph_mask[u] >> best_v) & 1:
                        if common_neighbors[u][best_v] & current:
                            valid = False
                            break
                if valid:
                    current |= (1 << best_v)
                    remaining &= ~(1 << best_v)
                else:
                    remaining &= ~(1 << best_v)
            return current.bit_count()
        
        best = greedy_lower_bound()
        
        def bt(remaining, current):
            nonlocal best
            if not remaining:
                if current.bit_count() > best:
                    best = current.bit_count()
                return
            
            # Bound
            if current.bit_count() + remaining.bit_count() <= best:
                return
            
            # Choose the vertex in remaining with highest degree (more likely to conflict)
            best_v = -1
            best_deg = -1
            c = remaining
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (graph_mask[v] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            
            # Check if current + {v} is triangle-free
            valid = True
            c = current
            while c:
                lsb2 = c & (-c)
                u = lsb2.bit_length() - 1
                c ^= lsb2
                if (graph_mask[u] >> v) & 1:
                    if common_neighbors[u][v] & current:
                        valid = False
                        break
            
            if valid:
                # Compute forbidden vertices
                forbidden = 0
                c = current
                while c:
                    lsb2 = c & (-c)
                    u = lsb2.bit_length() - 1
                    c ^= lsb2
                    if (graph_mask[u] >> v) & 1:
                        forbidden |= common_neighbors[u][v]
                
                new_remaining = remaining & ~forbidden & ~(1 << v)
                bt(new_remaining, current | (1 << v))
            
            # Try skipping v
            bt(remaining & ~(1 << v), current)
        
        full_mask = (1 << N) - 1
        bt(full_mask, 0)
        return best
    
    def random_k4_free_graph(edge_prob=0.5):
        graph_mask = [0] * N
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < edge_prob:
                    common = graph_mask[u] & graph_mask[v]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if not has_edge:
                        graph_mask[u] |= (1 << v)
                        graph_mask[v] |= (1 << u)
                        edges.append([u, v])
        return edges, graph_mask
    
    def local_search(edges, graph_mask, max_iters=100):
        current_score = max_triangle_free_size(graph_mask)
        best_local_score = current_score
        best_local_edges = edges[:]
        best_local_mask = graph_mask[:]
        
        # Temperature for simulated annealing
        temp = 1.0
        cooling = 0.95
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding edges
            for u in range(N):
                for v in range(u+1, N):
                    if (graph_mask[u] >> v) & 1:
                        continue
                    
                    # Check K4-free
                    common = graph_mask[u] & graph_mask[v]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if has_edge:
                        continue
                    
                    # Quick greedy check before expensive exact eval
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    
                    greedy_ub = greedy_triangle_free_upper_bound(graph_mask)
                    if greedy_ub > best_local_score + 2:
                        # Likely won't improve, revert
                        graph_mask[u] &= ~(1 << v)
                        graph_mask[v] &= ~(1 << u)
                        continue
                    
                    new_score = max_triangle_free_size(graph_mask)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        edges.append([u, v])
                        if current_score < best_local_score:
                            best_local_score = current_score
                            best_local_edges = edges[:]
                            best_local_mask = graph_mask[:]
                    else:
                        # Simulated annealing: accept with probability exp(-delta/temp)
                        delta = new_score - current_score
                        if random.random() < math.exp(-delta / max(temp, 0.01)):
                            current_score = new_score
                            edges.append([u, v])
                            if current_score < best_local_score:
                                best_local_score = current_score
                                best_local_edges = edges[:]
                                best_local_mask = graph_mask[:]
                        else:
                            graph_mask[u] &= ~(1 << v)
                            graph_mask[v] &= ~(1 << u)
            
            # Try removing edges
            for i in range(len(edges)-1, -1, -1):
                if time.time() > deadline - 5:
                    break
                u, v = edges[i]
                graph_mask[u] &= ~(1 << v)
                graph_mask[v] &= ~(1 << u)
                edges.pop(i)
                
                greedy_ub = greedy_triangle_free_upper_bound(graph_mask)
                if greedy_ub > best_local_score + 2:
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    edges.insert(i, [u, v])
                    continue
                
                new_score = max_triangle_free_size(graph_mask)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_edges = edges[:]
                        best_local_mask = graph_mask[:]
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(-delta / max(temp, 0.01)):
                        current_score = new_score
                        if current_score < best_local_score:
                            best_local_score = current_score
                            best_local_edges = edges[:]
                            best_local_mask = graph_mask[:]
                    else:
                        graph_mask[u] |= (1 << v)
                        graph_mask[v] |= (1 << u)
                        edges.insert(i, [u, v])
            
            temp *= cooling
            if not improved and temp < 0.1:
                break
        
        return best_local_edges, best_local_score, best_local_mask
    
    best_edges = None
    best_score = N + 1
    best_mask = None
    
    # Start with complete tripartite
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    tripartite_edges = []
    tripartite_mask = [0] * N
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    tripartite_edges.append([a, b])
                    tripartite_mask[a] |= (1 << b)
                    tripartite_mask[b] |= (1 << a)
    
    tripartite_score = max_triangle_free_size(tripartite_mask)
    if tripartite_score < best_score:
        best_score = tripartite_score
        best_edges = tripartite_edges[:]
        best_mask = tripartite_mask[:]
    
    # Local search from tripartite
    ls_edges, ls_score, ls_mask = local_search(tripartite_edges[:], tripartite_mask[:])
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
        best_mask = ls_mask[:]
    
    # Random restarts with varying edge probabilities
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        edge_prob = random.uniform(0.3, 0.7)
        edges, mask = random_k4_free_graph(edge_prob)
        score = max_triangle_free_size(mask)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
            best_mask = mask[:]
        
        # Local search
        ls_edges, ls_score, ls_mask = local_search(edges[:], mask[:])
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges[:]
            best_mask = ls_mask[:]
        
        if time.time() > deadline - 25:
            break
    
    # Final polish on best solution
    if best_edges is not None and time.time() < deadline - 10:
        final_mask = best_mask[:]
        final_edges = best_edges[:]
        final_score = best_score
        
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding edges
            for u in range(N):
                for v in range(u+1, N):
                    if (final_mask[u] >> v) & 1:
                        continue
                    common = final_mask[u] & final_mask[v]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (final_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if has_edge:
                        continue
                    
                    final_mask[u] |= (1 << v)
                    final_mask[v] |= (1 << u)
                    new_score = max_triangle_free_size(final_mask)
                    if new_score < final_score:
                        final_score = new_score
                        improved = True
                        final_edges.append([u, v])
                    else:
                        final_mask[u] &= ~(1 << v)
                        final_mask[v] &= ~(1 << u)
            
            # Try removing edges
            for i in range(len(final_edges)-1, -1, -1):
                if time.time() > deadline - 5:
                    break
                u, v = final_edges[i]
                final_mask[u] &= ~(1 << v)
                final_mask[v] &= ~(1 << u)
                final_edges.pop(i)
                
                new_score = max_triangle_free_size(final_mask)
                if new_score < final_score:
                    final_score = new_score
                    improved = True
                else:
                    final_mask[u] |= (1 << v)
                    final_mask[v] |= (1 << u)
                    final_edges.insert(i, [u, v])
            
            if not improved:
                break
        
        if final_score < best_score:
            best_score = final_score
            best_edges = final_edges[:]
            best_mask = final_mask[:]
    
    return best_edges if best_edges else []

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with {len(result)} edges")
