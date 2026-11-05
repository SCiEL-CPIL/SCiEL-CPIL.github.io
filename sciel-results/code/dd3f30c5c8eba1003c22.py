import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 148.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    edge_idx = {e: idx for idx, e in enumerate(edges)}
    k4_edge_indices = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4_edge_indices.append([edge_idx[e] for e in k4_edges])
    
    # edge_to_k4s: for each edge, list of K4 indices that contain it
    edge_to_k4s = [[] for _ in range(m)]
    for k4_idx, edge_indices in enumerate(k4_edge_indices):
        for e_idx in edge_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Vertex-labeling approach:
    # Assign each vertex a label from {0, 1, ..., k-1}
    # Edge (i,j) gets color = label[i] * (k) + label[j] if label[i] <= label[j], else label[j]*k + label[i]
    # But this gives k^2 colors. Instead, use a more compact mapping.
    
    # Better: use a "group" structure. Partition vertices into groups.
    # Edges within same group get one color, edges between groups get colors based on group pair.
    
    # Let's try: partition into g groups, edges within group i get color i,
    # edges between group i and j get color g + (i*j + j*i + min(i,j)*g + max(i,j)) or similar.
    
    # Actually, let's use the direct vertex-label approach with a hash function.
    # label[v] in {0, ..., L-1}
    # color(i,j) = (label[i] + label[j]) % C  where C is the number of colors
    
    # But we want to minimize C. Let's try small C values.
    
    # Alternative: use a "type" assignment where each vertex has a type,
    # and edge color depends on the pair of types.
    
    # Let's try: each vertex gets a label from {0, 1, ..., k-1}
    # Edge (i,j) color = (label[i] * (k+1) + label[j]) if label[i] <= label[j]
    #                    = (label[j] * (k+1) + label[i]) if label[j] < label[i]
    # This gives k*(k+1)/2 colors.
    
    # For k=5: 15 colors. For k=4: 10 colors.
    
    # Let's try to find a valid labeling with few colors.
    
    def count_distinct_colors_from_labels(labels, k):
        """Given vertex labels and group size k, compute edge colors and count distinct."""
        colors = []
        for i, j in edges:
            a, b = labels[i], labels[j]
            if a <= b:
                c = a * (k + 1) + b
            else:
                c = b * (k + 1) + a
            colors.append(c)
        return colors
    
    def is_valid_coloring(col):
        """Check if the entire coloring is valid."""
        for k4_e in k4_edge_indices:
            colors = set(col[i] for i in k4_e)
            if len(colors) < 5:
                return False
        return True
    
    def check_local_validity(col, changed_edge_idx):
        """Check if the coloring is valid, assuming only changed_edge_idx was modified."""
        for k4_idx in edge_to_k4s[changed_edge_idx]:
            e_indices = k4_edge_indices[k4_idx]
            colors = set(col[i] for i in e_indices)
            if len(colors) < 5:
                return False
        return True
    
    # Strategy: try different k values (number of vertex labels)
    # For each k, search for a valid labeling.
    
    best_result = None
    best_num_colors = float('inf')
    
    random.seed(42)
    
    # Try k = 4, 5, 6, 7 vertex labels
    for k in [4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        # Number of possible edge colors: k*(k+1)/2
        max_colors = k * (k + 1) // 2
        
        # If max_colors < 5, skip (can't have 5 distinct colors in any K4)
        if max_colors < 5:
            continue
        
        # Try to find a valid labeling
        # Start with random labels
        for attempt in range(10000):
            if time.time() > deadline:
                break
            
            labels = np.random.randint(0, k, size=n)
            colors = count_distinct_colors_from_labels(labels, k)
            
            if is_valid_coloring(colors):
                num_distinct = len(set(colors))
                if num_distinct < best_num_colors:
                    best_num_colors = num_distinct
                    best_result = colors.copy()
                    # Try to improve by local search on labels
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for v in range(n):
                            if time.time() > deadline:
                                break
                            old_label = labels[v]
                            for new_label in range(k):
                                if new_label == old_label:
                                    continue
                                labels[v] = new_label
                                new_colors = count_distinct_colors_from_labels(labels, k)
                                if is_valid_coloring(new_colors):
                                    new_num_distinct = len(set(new_colors))
                                    if new_num_distinct <= num_distinct:
                                        colors = new_colors
                                        num_distinct = new_num_distinct
                                        improved = True
                                        if num_distinct < best_num_colors:
                                            best_num_colors = num_distinct
                                            best_result = colors.copy()
                                    break  # Keep the first valid new label
                                else:
                                    labels[v] = old_label
                            else:
                                labels[v] = old_label
    
    # If we found a result, remap colors to 0..k-1
    if best_result is not None:
        result = [int(c) for c in best_result]
        unique_colors = sorted(set(result))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        result = [color_map[c] for c in result]
        return result
    
    # Fallback: use the original approach with (i+j) % 17
    coloring = np.array([(i + j) % 17 for i, j in edges], dtype=int)
    if not is_valid_coloring(coloring):
        # Random restart
        while time.time() < deadline:
            coloring = np.random.randint(0, 17, size=m)
            if is_valid_coloring(coloring):
                break
    
    # Try to reduce colors
    best_coloring = coloring.copy()
    best_colors = len(set(coloring))
    
    while time.time() < deadline:
        unique_colors, counts = np.unique(coloring, return_counts=True)
        
        if len(unique_colors) > 13:
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            cand_colors = unique_colors[candidates]
            target_color = cand_colors[np.argmax(cand_colors)]
            
            target_edges = np.where(coloring == target_color)[0]
            if len(target_edges) > 0:
                e_idx = random.choice(target_edges)
                other_colors = [c for c in unique_colors if c != target_color]
                if other_colors:
                    random.shuffle(other_colors)
                    found_valid = False
                    for new_color in other_colors:
                        old_color = coloring[e_idx]
                        coloring[e_idx] = new_color
                        if check_local_validity(coloring, e_idx):
                            new_unique = set(coloring)
                            if len(new_unique) < best_colors:
                                best_colors = len(new_unique)
                                best_coloring = coloring.copy()
                            found_valid = True
                            break
                        else:
                            coloring[e_idx] = old_color
                    if not found_valid:
                        coloring[e_idx] = old_color
        else:
            # Random swaps
            if random.random() < 0.5:
                i1 = random.randint(0, m - 1)
                i2 = random.randint(0, m - 1)
                if i1 != i2:
                    c1, c2 = coloring[i1], coloring[i2]
                    if c1 != c2:
                        coloring[i1], coloring[i2] = c2, c1
                        affected_k4s = set(edge_to_k4s[i1]) | set(edge_to_k4s[i2])
                        valid = True
                        for k4_idx in affected_k4s:
                            e_indices = k4_edge_indices[k4_idx]
                            colors = set(coloring[i] for i in e_indices)
                            if len(colors) < 5:
                                valid = False
                                break
                        if valid:
                            new_unique = set(coloring)
                            if len(new_unique) < best_colors:
                                best_colors = len(new_unique)
                                best_coloring = coloring.copy()
                        else:
                            coloring[i1], coloring[i2] = c1, c2
            else:
                e_idx = random.randint(0, m - 1)
                old_color = coloring[e_idx]
                palette = list(set(coloring))
                if len(palette) > 1:
                    new_color = random.choice(palette)
                    if new_color != old_color:
                        coloring[e_idx] = new_color
                        if check_local_validity(coloring, e_idx):
                            new_unique = set(coloring)
                            if len(new_unique) < best_colors:
                                best_colors = len(new_unique)
                                best_coloring = coloring.copy()
                        else:
                            coloring[e_idx] = old_color
    
    result = [int(c) for c in best_coloring]
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    return result
