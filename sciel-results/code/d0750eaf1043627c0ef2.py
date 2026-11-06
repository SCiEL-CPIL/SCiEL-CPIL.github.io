import random
import time
import sys
import numpy as np

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Precompute all triangles as bitmasks
    # A triangle is a set of 3 vertices {a,b,c} with edges (a,b),(b,c),(a,c)
    # We need to ensure no K4, i.e., no 4 vertices all pairwise connected.
    
    # Strategy: Use bitmask representation of the graph.
    # graph_mask is a list of N integers, where graph_mask[u] has bit v set if (u,v) is an edge.
    
    def is_k4_free(graph_mask):
        # For each pair of adjacent vertices (u,v), check if they share two common neighbors
        # that are also adjacent to each other (forming K4 with u,v).
        # Actually, K4 exists if there's a triangle {a,b,c} and a vertex d connected to all three.
        # More efficiently: for each edge (u,v), let common = graph_mask[u] & graph_mask[v].
        # If common contains an edge (w,x) where w,x are both in common, then {u,v,w,x} is K4.
        # Wait, that's not quite right. If w and x are both neighbors of u and v, and w-x is an edge,
        # then {u,v,w,x} has all 6 edges? No. u-w, u-x, v-w, v-x, w-x. We need u-v too (yes, it's an edge).
        # So yes, if there's an edge within the common neighborhood of (u,v), we have a K4.
        for u in range(N):
            for v in range(u+1, N):
                if (graph_mask[u] >> v) & 1:
                    common = graph_mask[u] & graph_mask[v]
                    # Check if common induces any edge
                    # For each w in common, check if graph_mask[w] & common != 0 (excluding w itself, but w isn't in common's bit representation of w)
                    # Actually, we just need to check if there's any edge within 'common'.
                    # Iterate through bits of common
                    c = common
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other vertex in common
                        if (graph_mask[w] & common) & (common ^ lsb):
                            return False
        return True

    def max_triangle_free_size(graph_mask):
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
        
        def bt(remaining, current):
            nonlocal best
            if not remaining:
                if current.bit_count() > best:
                    best = current.bit_count()
                return
            
            # Bound
            if current.bit_count() + remaining.bit_count() <= best:
                return
            
            # Choose the first vertex in remaining
            lsb = remaining & (-remaining)
            v = lsb.bit_length() - 1
            
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
    
    # Initial graph: start with a random K4-free graph
    # Or start with a complete tripartite and perturb
    
    def random_k4_free_graph():
        # Start with empty graph, add random edges, ensure no K4
        graph_mask = [0] * N
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < 0.5:
                    # Check if adding (u,v) creates a K4
                    # A K4 would be formed if there exist w,x such that all of u,v,w,x are mutually connected
                    # After adding (u,v), check if there's a triangle in the common neighborhood of u and v
                    # that already exists, OR if u and v share two neighbors w,x with w-x edge.
                    common = graph_mask[u] & graph_mask[v]
                    # Check if common contains an edge
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
    
    # Local search with spectral optimization
    def local_search(edges, graph_mask):
        # Use spectral optimization to guide edge modifications
        # Objective: minimize max_triangle_free_size while maintaining K4-free
        
        current_score = max_triangle_free_size(graph_mask)
        
        # Build adjacency matrix for spectral analysis
        A = np.zeros((N, N))
        for u in range(N):
            for v in range(u+1, N):
                if (graph_mask[u] >> v) & 1:
                    A[u, v] = 1
                    A[v, u] = 1
        
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Compute spectral radius
            eigenvalues = np.linalg.eigvalsh(A)
            spectral_radius = abs(eigenvalues[0])
            
            # Try adding edges that reduce spectral radius while maintaining K4-free
            improved = False
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
                    
                    # Add edge
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    A[u, v] = 1
                    A[v, u] = 1
                    
                    new_score = max_triangle_free_size(graph_mask)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        edges.append([u, v])
                    else:
                        # Revert
                        graph_mask[u] &= ~(1 << v)
                        graph_mask[v] &= ~(1 << u)
                        A[u, v] = 0
                        A[v, u] = 0
            
            # Try removing edges that increase spectral radius (which might help)
            for i in range(len(edges)-1, -1, -1):
                if time.time() > deadline:
                    break
                u, v = edges[i]
                # Remove edge
                graph_mask[u] &= ~(1 << v)
                graph_mask[v] &= ~(1 << u)
                A[u, v] = 0
                A[v, u] = 0
                edges.pop(i)
                
                new_score = max_triangle_free_size(graph_mask)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.insert(i, [u, v])
            
            if not improved:
                break
        
        return edges, current_score
    
    # Start with a few random restarts
    best_edges = None
    best_score = N + 1
    
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
    
    # Local search from tripartite
    ls_edges, ls_score = local_search(tripartite_edges[:], tripartite_mask[:])
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 30:  # Leave 30s margin for final polish
        restarts += 1
        edges, mask = random_k4_free_graph()
        score = max_triangle_free_size(mask)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
        
        # Local search
        ls_edges, ls_score = local_search(edges[:], mask[:])
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges[:]
        
        if time.time() > deadline - 40:
            break
    
    # Final high-precision polish: more aggressive local search on best solution
    if best_edges is not None and time.time() < deadline - 10:
        # Reconstruct mask from best_edges
        final_mask = [0] * N
        for u, v in best_edges:
            final_mask[u] |= (1 << v)
            final_mask[v] |= (1 << u)
        
        # Run additional local search iterations
        final_score = max_triangle_free_size(final_mask)
        for _ in range(50):
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
                        best_edges.append([u, v])
                    else:
                        final_mask[u] &= ~(1 << v)
                        final_mask[v] &= ~(1 << u)
            
            # Try removing edges
            for i in range(len(best_edges)-1, -1, -1):
                if time.time() > deadline - 5:
                    break
                u, v = best_edges[i]
                final_mask[u] &= ~(1 << v)
                final_mask[v] &= ~(1 << u)
                best_edges.pop(i)
                
                new_score = max_triangle_free_size(final_mask)
                if new_score < final_score:
                    final_score = new_score
                    improved = True
                else:
                    final_mask[u] |= (1 << v)
                    final_mask[v] |= (1 << u)
                    best_edges.insert(i, [u, v])
            
            if not improved:
                break
        
        if final_score < best_score:
            best_score = final_score
    
    return best_edges if best_edges else []

# Ensure the function is called correctly
if __name__ == "__main__":
    result = construct()
    print(f"Found graph with {len(result)} edges")
