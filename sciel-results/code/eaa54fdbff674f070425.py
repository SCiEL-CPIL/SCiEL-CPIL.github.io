import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 16
    target_edges = 65  # floor(16^2/4) + 1

    # Build initial graph: K_{6,10} + a perfect matching within the part of size 10
    # Part A: vertices 0..5 (size 6), Part B: vertices 6..15 (size 10)
    # Edges: all cross edges (6*10=60) + 5 matching edges within B = 65
    edges = set()
    for u in range(6):
        for v in range(6, 16):
            edges.add((min(u,v), max(u,v)))
    # Matching within B: (6,7), (8,9), (10,11), (12,13), (14,15)
    for a, b in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
        edges.add((min(a,b), max(a,b)))
    
    assert len(edges) == 65, f"Expected 65 edges, got {len(edges)}"

    def build_bitmasks(edge_set):
        masks = [0] * n
        for (u, v) in edge_set:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    def compute_score(edge_set):
        masks = build_bitmasks(edge_set)
        deg = [bin(m).count('1') for m in masks]
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if (masks[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = masks[u] & masks[v]
                    # Only count w > v to avoid duplicates
                    common &= ~((1 << (v+1)) - 1)
                    while common:
                        lsb = common & (-common)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    best_edges = set(edges)
    best_score = compute_score(best_edges)
    current_edges = set(edges)
    current_score = best_score

    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_non_edges(edge_set):
        return [p for p in all_pairs if p not in edge_set]

    # Local search: try edge swaps (remove one edge, add one non-edge)
    # Focus on reducing the max triangle degree sum
    random.seed(42)
    
    iter_count = 0
    stall_count = 0
    max_stall = 500
    
    while time.time() < deadline:
        iter_count += 1
        
        # Get current non-edges
        non_edges = get_non_edges(current_edges)
        
        if not non_edges:
            break
        
        # Find the worst triangle(s) to target
        masks = build_bitmasks(current_edges)
        deg = [bin(m).count('1') for m in masks]
        
        # Find all triangles and their degree sums
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if (masks[u] >> v) & 1:
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v+1)) - 1)
                    while common:
                        lsb = common & (-common)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        triangles.append((s, u, v, w))
                        common ^= lsb
        
        if not triangles:
            break
        
        # Sort by degree sum descending
        triangles.sort(reverse=True)
        
        # Try to improve by swapping edges
        improved = False
        # Sample a subset of non-edges for speed
        if len(non_edges) > 200:
            sample_non_edges = random.sample(non_edges, 200)
        else:
            sample_non_edges = non_edges
        
        # Try removing edges that are in worst triangles
        worst_triangles = triangles[:min(10, len(triangles))]
        
        for (ws, u, v, w) in worst_triangles:
            # Edges in this triangle: (u,v), (u,w), (v,w)
            tri_edges = [(min(u,v), max(u,v)), (min(u,w), max(u,w)), (min(v,w), max(v,w))]
            
            for rem_edge in tri_edges:
                # Try adding each non-edge
                for add_edge in sample_non_edges:
                    if add_edge == rem_edge:
                        continue
                    # Check if this creates a valid graph
                    new_edges = set(current_edges)
                    new_edges.discard(rem_edge)
                    new_edges.add(add_edge)
                    
                    if len(new_edges) != target_edges:
                        continue
                    
                    new_score = compute_score(new_edges)
                    
                    if new_score < current_score:
                        current_edges = new_edges
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_edges = set(current_edges)
                            best_score = current_score
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            stall_count += 1
            if stall_count > max_stall:
                # Try random restarts with small perturbations
                if time.time() > deadline - 20:
                    break
                # Perturb: remove 1-2 random edges, add 1-2 random non-edges
                perturbation = 1
                for _ in range(perturbation):
                    rem = random.choice(list(current_edges))
                    current_edges.discard(rem)
                    ne = get_non_edges(current_edges)
                    if ne:
                        add = random.choice(ne)
                        current_edges.add(add)
                current_score = compute_score(current_edges)
                stall_count = 0
        else:
            stall_count = 0
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Return best result
    result_edges = [list(e) for e in best_edges]
    return {'n': n, 'edges': result_edges}
