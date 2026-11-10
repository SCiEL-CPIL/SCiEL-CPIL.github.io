import time
import random

def construct():
    t0 = time.time()
    deadline = 130.0  # Safer margin (evaluator kills at 150s)
    
    # Neighbors in triangular lattice (axial coordinates)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def count_edges(pts):
        pset = set(pts)
        edges = 0
        for a, b in pts:
            for da, db in neighbors:
                if (a + da, b + db) in pset:
                    edges += 1
        return edges // 2

    def edge_delta_remove_add(pts_set, remove_pt, add_pt):
        lost = 0
        for da, db in neighbors:
            if (remove_pt[0] + da, remove_pt[1] + db) in pts_set:
                lost += 1
        gained = 0
        for da, db in neighbors:
            nb = (add_pt[0] + da, add_pt[1] + db)
            if nb in pts_set and nb != remove_pt:
                gained += 1
        return gained - lost

    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    # Candidate pool: all points within hex_dist <= 8
    all_candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            if hex_dist(a, b) <= 8:
                all_candidates.append((a, b))
    
    # Try multiple greedy constructions with different tie-breaking
    best_points = None
    best_edges = 0
    
    for tie_break in ['dist', 'lex']:
        current = [(0, 0)]
        current_set = {(0, 0)}
        
        for _ in range(31):
            best_pt = None
            best_score = -1
            best_dist = 999
            best_lex = (999, 999)
            for pt in all_candidates:
                if pt in current_set:
                    continue
                # Count edges this point would add
                score = 0
                for da, db in neighbors:
                    if (pt[0] + da, pt[1] + db) in current_set:
                        score += 1
                d = hex_dist(pt[0], pt[1])
                lex = (abs(pt[0]), abs(pt[1]))
                
                if tie_break == 'dist':
                    if score > best_score or (score == best_score and d < best_dist):
                        best_score = score
                        best_dist = d
                        best_pt = pt
                elif tie_break == 'lex':
                    if score > best_score or (score == best_score and lex < best_lex):
                        best_score = score
                        best_lex = lex
                        best_pt = pt
            if best_pt:
                current.append(best_pt)
                current_set.add(best_pt)
        
        edges = count_edges(current)
        if edges > best_edges:
            best_edges = edges
            best_points = list(current)
        
        if best_edges >= 76:
            break
    
    if best_edges >= 76:
        return [list(p) for p in best_points]
    
    # Local search with simulated annealing
    current_set = set(best_points)
    pool = [p for p in all_candidates if p not in current_set]
    random.seed(42)
    current = list(best_points)
    cur_edges = best_edges
    
    T = 2.0
    T_min = 0.01
    cooling = 0.998
    
    while (time.time() - t0) < deadline:
        if T < T_min:
            T = T_min
        
        i = random.randint(0, 31)
        old = current[i]
        j = random.randint(0, len(pool) - 1)
        new = pool[j]
        
        if new in current_set:
            continue
            
        temp_set = current_set - {old}
        delta = edge_delta_remove_add(temp_set, old, new)
        
        if delta > 0:
            current[i] = new
            current_set.discard(old)
            current_set.add(new)
            cur_edges += delta
            pool[j] = old
            
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_points = list(current)
                if best_edges >= 76:
                    break
        else:
            # Accept with probability exp(delta / T)
            if delta >= -3 and random.random() < (2.718281828 ** (delta / T)):
                current[i] = new
                current_set.discard(old)
                current_set.add(new)
                cur_edges += delta
                pool[j] = old
        
        T *= cooling
    
    return [list(p) for p in best_points]
