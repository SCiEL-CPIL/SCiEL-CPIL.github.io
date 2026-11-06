"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    n = 16
    edge_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_idx)
    
    # Map edge (i,j) to index
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edge_idx):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
        
    # Precompute all K4s and their edge indices
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in combinations(k4, 2):
            edges.append(edge_to_idx[(a, b)])
        k4_edges.append(edges)
        
    # For each edge, which K4s does it belong to?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)

    # F4^2 geometry: represent points as pairs (a,b) where a,b in F4
    # F4 = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # Represent F4 elements as: 0->0, 1->1, alpha->2, alpha+1->3
    # Addition and multiplication in F4
    F4_add = np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]])
    F4_mul = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]])
    
    # 16 points in F4^2
    points = [(a, b) for a in range(4) for b in range(4)]
    point_to_vertex = {p: i for i, p in enumerate(points)}
    
    # Compute direction (slope) for each pair of points
    # Direction is (db, da) normalized, or use the line's parallel class
    # In F4^2, there are 5 parallel classes (directions)
    # For points p1=(a1,b1), p2=(a2,b2), direction is determined by (a2-a1, b2-b1)
    # Normalize: if da != 0, slope = (db-da) * da^{-1}; if da == 0, slope = infinity
    
    # Actually, let's just compute the parallel class index
    # The 5 directions correspond to the 5 lines through a point
    # Direction vector (da, db) with (da,db) != (0,0), modulo scaling
    
    def get_direction(p1, p2):
        da = F4_add[p1[0], p2[0]]  # p2[0] - p1[0] in F4... wait, need subtraction
        # In F4, subtraction = addition (char 2)
        db = F4_add[p1[1], p2[1]]
        if da == 0 and db == 0:
            return 0  # shouldn't happen
        # Normalize: find representative
        # 5 directions: (1,0), (0,1), (1,1), (1,alpha), (1,alpha+1)
        if da == 0:
            return 1  # vertical
        # da != 0, slope = db * da^{-1}
        # Invert da in F4
        inv = {0:0, 1:1, 2:2, 3:3}  # in F4, each nonzero element is its own inverse? No.
        # F4: 1*1=1, alpha*alpha=alpha+1, (alpha+1)*(alpha+1)=1... let me check
        # alpha^2 = alpha + 1, so alpha * (alpha+1) = alpha^2 + alpha = (alpha+1) + alpha = 1
        # So alpha^{-1} = alpha+1 = 3, and (alpha+1)^{-1} = alpha = 2
        inv_map = {0:0, 1:1, 2:3, 3:2}
        da_inv = inv_map[da]
        slope = F4_mul[db, da_inv]
        # slope in {0, 1, 2, 3}
        # Map to direction index 0-4
        # slope 0 -> horizontal, 1, 2, 3 -> other slopes
        # But we also have vertical (da=0)
        # So: vertical=1, horizontal=0, slope=1->2, slope=2->3, slope=3->4
        if slope == 0:
            return 0  # horizontal
        elif slope == 1:
            return 2
        elif slope == 2:
            return 3
        else:
            return 4
    
    # Build geometry-based coloring: color = direction
    geo_colors = []
    for (i, j) in edge_idx:
        p1 = points[i]
        p2 = points[j]
        d = get_direction(p1, p2)
        geo_colors.append(d)
    
    # Count violations for geo coloring
    def count_violations(colors):
        vcount = 0
        for edges in k4_edges:
            cs = set(colors[e] for e in edges)
            if len(cs) < 5:
                vcount += 1
        return vcount
    
    geo_violations = count_violations(geo_colors)
    
    # Fallback coloring
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    def setup_state(colors):
        k4_color_counts = [{} for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        violations = []
        
        for k4_idx, edges in enumerate(k4_edges):
            counts = {}
            for e in edges:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            k4_distinct[k4_idx] = len(counts)
            if k4_distinct[k4_idx] < 5:
                violations.append(k4_idx)
                
        return k4_color_counts, k4_distinct, violations

    def try_move_safe(colors, k4_color_counts, k4_distinct, violations, num_c):
        if not violations:
            return True
        
        v_idx = random.choice(violations)
        v_k4_edges = k4_edges[v_idx]
        v_counts = k4_color_counts[v_idx]
        
        # Find colors missing in this K4
        missing_colors = [c for c in range(num_c) if c not in v_counts]
        if not missing_colors:
            return False
            
        edges_to_try = v_k4_edges[:]
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            old_c = colors[e]
            count_old = v_counts.get(old_c, 0)
            if count_old == 1:
                continue
                
            new_c = random.choice(missing_colors)
            if new_c == old_c:
                continue
                
            affected_k4s = edge_k4s[e]
            delta_violations = 0
            
            for k_idx in affected_k4s:
                k_counts = k4_color_counts[k_idx]
                k_dist = k4_distinct[k_idx]
                was_violated = k_dist < 5
                
                cnt_old = k_counts.get(old_c, 0)
                if cnt_old == 1:
                    new_dist = k_dist - 1
                else:
                    new_dist = k_dist
                    
                cnt_new = k_counts.get(new_c, 0)
                if cnt_new == 0:
                    new_dist += 1
                    
                is_violated = new_dist < 5
                
                if was_violated and not is_violated:
                    delta_violations -= 1
                elif not was_violated and is_violated:
                    delta_violations += 1
                    
            if delta_violations < 0:
                colors[e] = new_c
                for k_idx in affected_k4s:
                    k_counts = k4_color_counts[k_idx]
                    k_dist = k4_distinct[k_idx]
                    was_violated = k_dist < 5
                    
                    cnt_old = k_counts.get(old_c, 0)
                    if cnt_old == 1:
                        del k_counts[old_c]
                        new_dist = k_dist - 1
                    else:
                        k_counts[old_c] = cnt_old - 1
                        new_dist = k_dist
                        
                    cnt_new = k_counts.get(new_c, 0)
                    if cnt_new == 0:
                        k_counts[new_c] = 1
                        new_dist += 1
                    else:
                        k_counts[new_c] = cnt_new + 1
                        
                    is_violated = new_dist < 5
                    k4_distinct[k_idx] = new_dist
                    
                    if was_violated and not is_violated:
                        try:
                            violations.remove(k_idx)
                        except ValueError:
                            pass
                    elif not was_violated and is_violated:
                        violations.append(k_idx)
                return True
            else:
                continue
                
        return False

    best_result = fallback
    best_colors = 17

    # Try geometry-based initialization first
    init_strategies = []
    if geo_violations < 100:
        init_strategies.append(('geo', geo_colors[:]))
    init_strategies.append(('random13', [random.randint(0, 12) for _ in range(num_edges)]))
    init_strategies.append(('random14', [random.randint(0, 13) for _ in range(num_edges)]))
    init_strategies.append(('fallback', fallback[:]))
    
    for num_c in [13, 14, 15, 16]:
        if time.time() > deadline - 5:
            break
            
        for restart in range(30):
            if time.time() > deadline - 5:
                    break
                    
            if restart == 0 and num_c == 13:
                # Use geometry-based init for first restart at 13 colors
                colors = geo_colors[:]
                # Remap colors to 0..12
                color_map = {}
                next_c = 0
                for i in range(num_edges):
                    c = colors[i]
                    if c not in color_map:
                        color_map[c] = next_c
                        next_c += 1
                    colors[i] = color_map[c]
                # If we used more than 13 colors, truncate
                if next_c > num_c:
                    for i in range(num_edges):
                        if colors[i] >= num_c:
                            colors[i] = random.randint(0, num_c - 1)
            elif restart == 1 and num_c == 13:
                colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            else:
                colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            
            k4_color_counts, k4_distinct, violations = setup_state(colors)
            
            for step in range(50000):
                if time.time() > deadline - 5:
                    break
                    
                if not violations:
                    used = len(set(colors))
                    if used < best_colors:
                        best_colors = used
                        best_result = colors[:]
                    break
                    
                if not try_move_safe(colors, k4_color_counts, k4_distinct, violations, num_c):
                    break

    if best_colors > 17:
        return fallback
        
    return best_result
# EVOLVE-BLOCK-END
