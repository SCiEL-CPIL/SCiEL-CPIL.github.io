import random
import time
import math

N = 64
POINTS = [(x, y) for x in range(N) for y in range(N)]
M = len(POINTS)

def sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def construct():
    start_time = time.time()
    time_limit = 295.0
    
    best_size = 0
    best_set = []
    
    def greedy_random():
        current_set = []
        dist_sets = []
        candidates = POINTS[:]
        random.shuffle(candidates)
        
        for p in candidates:
            ok = True
            n = len(current_set)
            
            if n >= 2:
                d_p = [sq_dist(p, ex) for ex in current_set]
                
                # Check apex at p
                if len(set(d_p)) < n:
                    ok = False
                else:
                    # Check apex at existing points
                    for i in range(n):
                        di = d_p[i]
                        if di in dist_sets[i]:
                            ok = False
                            break
            elif n == 1:
                d_p = [sq_dist(p, current_set[0])]
            else:
                d_p = []
            
            if ok:
                current_set.append(p)
                for i in range(n):
                    dist_sets[i].add(d_p[i])
                dist_sets.append(set(d_p))
        
        return len(current_set), current_set
    
    def local_search(initial_set):
        current_set = list(initial_set)
        dist_sets = []
        
        # Rebuild dist_sets
        n = len(current_set)
        for i in range(n):
            ds = set()
            for j in range(n):
                if i != j:
                    ds.add(sq_dist(current_set[i], current_set[j]))
            dist_sets.append(ds)
        
        improved = True
        while improved:
            improved = False
            candidates = POINTS[:]
            random.shuffle(candidates)
            
            for p in candidates:
                if p in current_set:
                    continue
                
                ok = True
                n = len(current_set)
                
                if n >= 2:
                    d_p = [sq_dist(p, ex) for ex in current_set]
                    
                    if len(set(d_p)) < n:
                        ok = False
                    else:
                        for i in range(n):
                            di = d_p[i]
                            if di in dist_sets[i]:
                                ok = False
                                break
                elif n == 1:
                    d_p = [sq_dist(p, current_set[0])]
                else:
                    d_p = []
                
                if ok:
                    current_set.append(p)
                    for i in range(n):
                        dist_sets[i].add(d_p[i])
                    dist_sets.append(set(d_p))
                    improved = True
                    break
        
        return len(current_set), current_set
    
    # Run multiple greedy restarts
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        
        s, s_set = greedy_random()
        if s > best_size:
            best_size = s
            best_set = s_set
    
    # Try local search from the best set found
    if best_set and time.time() - start_time < time_limit:
        s, s_set = local_search(best_set)
        if s > best_size:
            best_size = s
            best_set = s_set
    
    return best_set
