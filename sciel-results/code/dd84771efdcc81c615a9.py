import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n = 16
    # Precompute all K4 subgraphs
    quads = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    quads.append((i, j, k, l))
    
    # Precompute for each edge, which quads contain it
    edge_to_quads = {}
    for idx, (a, b, c, d) in enumerate(quads):
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        for e in edges:
            if e not in edge_to_quads:
                edge_to_quads[e] = []
            edge_to_quads[e].append(idx)
    
    def count_colors_for_quad(colors, quad):
        i, j, k, l = quad
        return len({colors[i][j], colors[i][k], colors[i][l], colors[j][k], colors[j][l], colors[k][l]})
    
    def compute_quad_colors(colors):
        """Compute color count for each quad"""
        counts = [0] * len(quads)
        for idx, (i, j, k, l) in enumerate(quads):
            counts[idx] = count_colors_for_quad(colors, (i, j, k, l))
        return counts
    
    def compute_violations(quad_counts):
        """Count how many quads have < 5 colors"""
        return sum(1 for c in quad_counts if c < 5)
    
    def compute_min_color_count(quad_counts):
        """Get minimum color count across all quads"""
        return min(quad_counts) if quad_counts else 0
    
    # Try k from 13 to 15
    best_result = None
    best_k = 17
    
    for k in range(13, 16):
        # Multiple restarts
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            colors = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    colors[i][j] = random.randint(0, k-1)
                    colors[j][i] = colors[i][j]
            
            # Compute initial quad colors
            quad_counts = compute_quad_colors(colors)
            current_violations = compute_violations(quad_counts)
            current_min_count = compute_min_color_count(quad_counts)
            
            # Check if already valid
            if current_violations == 0:
                res = []
                for i in range(n):
                    for j in range(i+1, n):
                        res.append(colors[i][j])
                return res
            
            # Local search with incremental updates
            max_iters = 20000
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random edge
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                old_c = colors[i][j]
                
                # Try changing to a random new color
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    continue
                
                # Get affected quads
                affected_quads = edge_to_quads.get((i, j), [])
                
                # Save old counts for affected quads
                old_counts = {}
                for qidx in affected_quads:
                    old_counts[qidx] = quad_counts[qidx]
                
                # Apply change
                colors[i][j] = new_c
                colors[j][i] = new_c
                
                # Update only affected quads
                for qidx in affected_quads:
                    quad_counts[qidx] = count_colors_for_quad(colors, quads[qidx])
                
                # Compute new violations
                new_violations = current_violations
                for qidx in affected_quads:
                    if old_counts[qidx] < 5 and quad_counts[qidx] >= 5:
                        new_violations -= 1
                    elif old_counts[qidx] >= 5 and quad_counts[qidx] < 5:
                        new_violations += 1
                
                # Accept if violations decrease or stay same
                if new_violations <= current_violations:
                    current_violations = new_violations
                    if current_violations == 0:
                        res = []
                        for a in range(n):
                            for b in range(a+1, n):
                                res.append(colors[a][b])
                        return res
                else:
                    # Revert
                    colors[i][j] = old_c
                    colors[j][i] = old_c
                    for qidx in affected_quads:
                        quad_counts[qidx] = old_counts[qidx]
        
    # Fallback: Return the seed solution (17 colors) if no better found
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
