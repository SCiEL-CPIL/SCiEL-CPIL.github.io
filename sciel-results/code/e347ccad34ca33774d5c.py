import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    edge_ij = []  # map edge index -> (i, j)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_ij.append((i, j))
            idx += 1
    
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
    
    # Try multiple initial colorings and pick the best
    def try_initial_coloring(colors):
        """Check if coloring is valid and return distinct color count."""
        violations = 0
        for edge_idxs in k4s:
            edge_colors = set(colors[ei] for ei in edge_idxs)
            if len(edge_colors) < 5:
                violations += 1
        if violations > 0:
            return None
        return len(set(colors))
    
    # Candidate initial colorings
    candidates = []
    
    # Additive: (i+j) % p
    for p in [17, 19, 23, 29]:
        colors = [(i + j) % p for i in range(n) for j in range(i + 1, n)]
        candidates.append(colors)
    
    # Multiplicative: (i*j) % p for p prime >= 16
    for p in [17, 19, 23, 29, 31]:
        colors = [(i * j) % p for i in range(n) for j in range(i + 1, n)]
        candidates.append(colors)
    
    # Mixed: (i*a + j*b) % p
    for p in [17, 19, 23]:
        for a in range(1, p):
            for b in range(1, p):
                colors = [(i * a + j * b) % p for i in range(n) for j in range(i + 1, n)]
                candidates.append(colors)
    
    # Find best initial coloring
    best_init_colors = None
    best_init_distinct = float('inf')
    for colors in candidates:
        distinct = try_initial_coloring(colors)
        if distinct is not None and distinct < best_init_distinct:
            best_init_distinct = distinct
            best_init_colors = colors[:]
    
    if best_init_colors is None:
        # Fallback to (i+j) % 17
        colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    else:
        colors = best_init_colors
    
    def count_k4_violations(colors):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for edge_idxs in k4s:
            edge_colors = set(colors[ei] for ei in edge_idxs)
            if len(edge_colors) < 5:
                violations += 1
        return violations
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    current_violations = count_k4_violations(colors)
    current_colors = count_distinct_colors(colors)
    
    best_colors = current_colors
    best_col_list = colors[:]
    
    random.seed(42)
    
    # Local search: try to reduce number of distinct colors
    while time.time() < deadline:
        used_colors = sorted(set(colors))
        if len(used_colors) <= best_colors:
            break
        
        # Pick a color to try to eliminate - prefer colors with fewer edges
        color_counts = {}
        for c in used_colors:
            color_counts[c] = sum(1 for x in colors if x == c)
        target_color = min(used_colors, key=lambda c: color_counts[c])
        
        # Find all edges with this color
        edges_with_color = [i for i in range(120) if colors[i] == target_color]
        
        if not edges_with_color:
            continue
        
        # Try remapping each edge to another color
        improved = False
        for edge_idx in edges_with_color:
            if time.time() > deadline:
                break
            
            # Try changing this edge to each other color
            for new_color in used_colors:
                if new_color == target_color:
                    continue
                
                old_color = colors[edge_idx]
                colors[edge_idx] = new_color
                
                # Check only K4s containing this edge
                new_violations = 0
                for k4_idx in edge_to_k4s[edge_idx]:
                    edge_idxs = k4s[k4_idx]
                    edge_colors = set(colors[ei] for ei in edge_idxs)
                    if len(edge_colors) < 5:
                        new_violations += 1
                
                if new_violations == 0:
                    # Check if target_color is still in use
                    if target_color not in colors:
                        current_colors -= 1
                        if current_colors < best_colors:
                            best_colors = current_colors
                            best_col_list = colors[:]
                        improved = True
                        break
                    else:
                        colors[edge_idx] = old_color
                else:
                    colors[edge_idx] = old_color
        
        if not improved:
            # Try random perturbation to escape local optima
            for _ in range(10):
                if time.time() > deadline:
                    break
                edge_idx = random.randint(0, 119)
                old_color = colors[edge_idx]
                new_color = random.choice(used_colors)
                colors[edge_idx] = new_color
                
                # Check violations
                new_violations = 0
                for k4_idx in edge_to_k4s[edge_idx]:
                    edge_idxs = k4s[k4_idx]
                    edge_colors = set(colors[ei] for ei in edge_idxs)
                    if len(edge_colors) < 5:
                        new_violations += 1
                
                if new_violations > 0:
                    colors[edge_idx] = old_color
    
    # Return the best coloring found
    if count_k4_violations(best_col_list) > 0:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_col_list
