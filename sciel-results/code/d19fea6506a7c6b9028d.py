import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for polish and return
    
    # Lattice neighbors for a + b*omega
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]

    def get_edges(points):
        """Calculate number of edges in the set of points."""
        point_set = set(points)
        edges = 0
        for p in points:
            a, b = p
            # Check 3 unique neighbors to avoid double counting
            if (a+1, b) in point_set: edges += 1
            if (a, b+1) in point_set: edges += 1
            if (a-1, b+1) in point_set: edges += 1
        return edges

    # Initial configuration: 16x2 strip
    points = [[a, b] for a in range(16) for b in range(2)]
    current_edges = get_edges(points)
    best_points = [p.copy() for p in points]
    best_edges = current_edges
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    point_set = set(tuple(p) for p in points)
    
    # Run SA
    max_steps = 5000000 # High limit, time will cut it
    
    for step in range(max_steps):
        if time.time() > deadline - 5: # Reserve 5s for polish
            break
        if T < min_T:
            T = min_T
            
        idx = random.randint(0, 31)
        p_old = tuple(points[idx])
        
        # Move selection
        r = random.random()
        if r < 0.6:
            da, db = random.choice(neighbors)
            p_new = (p_old[0] + da, p_old[1] + db)
        elif r < 0.9:
            da = random.randint(-2, 2)
            db = random.randint(-2, 2)
            p_new = (p_old[0] + da, p_old[1] + db)
        else:
            # Larger jump to escape local optima
            da = random.randint(-5, 5)
            db = random.randint(-5, 5)
            p_new = (p_old[0] + da, p_old[1] + db)
            
        if p_new in point_set:
            continue
            
        lost = 0
        for da, db in neighbors:
            if (p_old[0] + da, p_old[1] + db) in point_set:
                lost += 1
                
        gained = 0
        for da, db in neighbors:
            if (p_new[0] + da, p_new[1] + db) in point_set:
                if (p_new[0] + da, p_new[1] + db) != p_old:
                    gained += 1
                    
        delta = gained - lost
        
        if delta > 0 or random.random() < math.exp(delta / T):
            point_set.remove(p_old)
            point_set.add(p_new)
            points[idx] = list(p_new)
            current_edges += delta
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = [list(p) for p in points]
                
        T *= cooling
        
    # Polish phase: Greedy local search on best_points
    polish_deadline = start_time + 118
    
    current_best = [list(p) for p in best_points]
    current_best_edges = best_edges
    best_final_points = current_best
    best_final_edges = current_best_edges
    
    while time.time() < polish_deadline:
        improved = False
        point_set = set(tuple(p) for p in current_best)
        
        for i in range(32):
            if time.time() > polish_deadline:
                break
            p_old = tuple(current_best[i])
            
            # Try all neighbors and some random spots
            candidates = []
            for da, db in neighbors:
                candidates.append((p_old[0] + da, p_old[1] + db))
            # Add a few random spots
            for _ in range(5):
                da = random.randint(-3, 3)
                db = random.randint(-3, 3)
                candidates.append((p_old[0] + da, p_old[1] + db))
                
            best_delta = 0
            best_p_new = None
            
            for p_new in candidates:
                if p_new in point_set:
                    continue
                if p_new == p_old:
                    continue
                    
                lost = 0
                for da, db in neighbors:
                    if (p_old[0] + da, p_old[1] + db) in point_set:
                        lost += 1
                        
                gained = 0
                for da, db in neighbors:
                    if (p_new[0] + da, p_new[1] + db) in point_set:
                        if (p_new[0] + da, p_new[1] + db) != p_old:
                            gained += 1
                            
                delta = gained - lost
                if delta > best_delta:
                    best_delta = delta
                    best_p_new = p_new
                    
            if best_delta > 0 and best_p_new is not None:
                point_set.remove(p_old)
                point_set.add(best_p_new)
                current_best[i] = list(best_p_new)
                current_best_edges += best_delta
                improved = True
                
        if current_best_edges > best_final_edges:
            best_final_edges = current_best_edges
            best_final_points = [list(p) for p in current_best]
            
        if not improved:
            break
            
    return best_final_points
