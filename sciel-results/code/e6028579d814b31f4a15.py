import random
import time
import itertools

def construct():
    n = 18
    # Start with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.add((min(a, b), max(a, b)))
    
    def is_k4_free(edge_set):
        # Check for K4
        # Optimize: if a vertex has degree < 3, it can't be in K4? No.
        # Just check all 4-subsets
        verts = list(range(n))
        # Precompute adjacency matrix for speed
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        
        # Count K4s
        # If we find one, return False
        for c in itertools.combinations(verts, 4):
            u, v, w, x = c
            if adj[u][v] and adj[u][w] and adj[u][x] and adj[v][w] and adj[v][x] and adj[w][x]:
                return False
        return True

    def max_triangle_free_size(edge_set):
        # Find largest subset S such that G[S] is triangle-free
        # Use branch and bound
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
            
        best_size = 0
        # We can prune if current potential size <= best_size
        
        # Simple heuristic: Start with all vertices, remove those involved in triangles?
        # Or use a greedy set packing?
        
        # For n=18, we can try all subsets of size >= 12?
        # There are C(18, 12) = 18564 subsets of size 12.
        # C(18, 13) = 8568.
        # C(18, 14) = 3060.
        # C(18, 15) = 816.
        # C(18, 16) = 153.
        # C(18, 17) = 18.
        # C(18, 18) = 1.
        # Total ~ 30k subsets.
        # Checking if a subset is triangle-free takes O(k^3) ~ 12^3 = 1728 ops.
        # 30k * 1728 ~ 50M ops. Feasible in Python? Maybe 1-2 seconds.
        
        # We want to know if there is a triangle-free induced subgraph of size > best_score.
        # If we find one, we can update best_score.
        # We want to MINIMIZE the score, so we want to know the MAX size.
        
        # To speed up, we check from largest size down.
        # If we find a triangle-free subgraph of size k, the score is at least k.
        # We want to know the exact max size to compare with current best.
        
        # Let's just check if there is a triangle-free subgraph of size 12.
        # If yes, score >= 12.
        # If no, check 11.
        
        # This function returns the max size.
        
        def is_triangle_free(subset):
            verts = list(subset)
            k = len(verts)
            # Check all triples
            for i in range(k):
                for j in range(i+1, k):
                    for l in range(j+1, k):
                        u, v, w = verts[i], verts[j], verts[l]
                        if adj[u][v] and adj[v][w] and adj[w][u]:
                            return False
            return True

        # Check sizes from 18 down to 1
        # But we only care if it's < 12 (to beat K6,6,6)
        # So we check if there is a triangle-free subgraph of size 12.
        # If yes, return 12 (or more).
        # If no, check 11.
        
        # Optimization: If we just want to know if score < 12, 
        # we need to verify that ALL subsets of size 12 contain a triangle.
        
        # Let's compute the exact max size.
        max_sz = 0
        # Iterate sizes from 18 down to 1
        # Stop when we find the first one? No, we want the MAX.
        # So we check 18, 17, ..., 1.
        # First one found is the max.
        
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                if is_triangle_free(subset):
                    return size
        return 0

    # Local Search
    current_edges = edges
    current_score = 12 # Known for K6,6,6
    
    # Try to improve
    start_time = time.time()
    timeout = 100 # Leave margin
    
    while time.time() - start_time < timeout:
        # Pick a random edge to flip
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        e = random.choice(all_possible_edges)
        
        new_edges = set(current_edges)
        if e in new_edges:
            new_edges.remove(e)
        else:
            new_edges.add(e)
        
        # Check K4-free
        if not is_k4_free(new_edges):
            continue
            
        # Check score
        # Only compute if we hope to improve?
        # We can't know without computing.
        # But computing is expensive.
        # Heuristic: If we removed an edge, score likely increases or stays same.
        # If we added an edge, score likely decreases.
        # So prefer adding edges?
        # But adding edges risks K4.
        
        # Let's just compute score for a few random valid moves.
        # To save time, we can sample.
        
        new_score = max_triangle_free_size(new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            if current_score < 12:
                # Found better!
                break
        
    # Return list of edges
    return [list(e) for e in current_edges]
