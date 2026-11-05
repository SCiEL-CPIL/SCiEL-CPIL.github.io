def construct():
    import time
    import random
    
    deadline = time.time() + 290
    
    N = 64
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_isosceles(p1, p2, p3):
        d12 = sq_dist(p1, p2)
        d13 = sq_dist(p1, p3)
        d23 = sq_dist(p2, p3)
        return d12 == d13 or d12 == d23 or d13 == d23

    def can_add_fast(p, S, dist_sets):
        dists_from_p = []
        for q in S:
            d = sq_dist(p, q)
            if d in dists_from_p:
                return False
            dists_from_p.append(d)
            if d in dist_sets[q]:
                return False
        return True

    def add_point(S, dist_sets, p):
        S.append(p)
        dists = []
        for q in S[:-1]:
            d = sq_dist(p, q)
            dists.append(d)
            dist_sets[q].add(d)
        dist_sets[p] = set(dists)

    def remove_point(S, dist_sets, p):
        S.remove(p)
        for q in S:
            d = sq_dist(p, q)
            if d in dist_sets[q]:
                dist_sets[q].discard(d)
        del dist_sets[p]

    def greedy(order):
        S = []
        dist_sets = {}
        for p in order:
            if time.time() > deadline:
                break
            if p in dist_sets:
                continue
            if can_add_fast(p, S, dist_sets):
                add_point(S, dist_sets, p)
        return S, dist_sets

    def local_search(S, dist_sets):
        """Swap-based local search: try removing each point and adding the best replacement."""
        current_S = list(S)
        current_dist = {k: v.copy() for k, v in dist_sets.items()}
        best_S = list(current_S)
        best_dist = {k: v.copy() for k, v in current_dist.items()}
        best_size = len(current_S)
        
        rng = random.Random(12345)
        
        while time.time() < deadline - 10:
            improved = False
            indices = list(range(len(current_S)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 10:
                    break
                if len(current_S) <= best_size:
                    break
                
                p_remove = current_S[idx]
                
                remove_point(current_S, current_dist, p_remove)
                
                candidates = [p for p in all_points if p not in current_dist]
                rng.shuffle(candidates)
                
                added = False
                for p_add in candidates:
                    if time.time() > deadline - 10:
                        break
                    if can_add_fast(p_add, current_S, current_dist):
                        add_point(current_S, current_dist, p_add)
                        added = True
                        break
                
                if added:
                    if len(current_S) > best_size:
                        best_S = list(current_S)
                        best_dist = {k: v.copy() for k, v in current_dist.items()}
                        best_size = len(current_S)
                        improved = True
                else:
                    add_point(current_S, current_dist, p_remove)
            
            if not improved:
                break
        
        return best_S, best_dist

    best_S = []
    best_dist = {}
    
    rng = random.Random(42)
    
    for i in range(15):
        if time.time() > deadline - 20:
            break
        order = all_points.copy()
        rng.shuffle(order)
        S, dist_sets = greedy(order)
        if len(S) > len(best_S):
            best_S = S
            best_dist = dist_sets
    
    if len(best_S) > 0 and time.time() < deadline - 15:
        best_S, best_dist = local_search(best_S, best_dist)
    
    valid_S = list(best_S)
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        n = len(valid_S)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_isosceles(valid_S[i], valid_S[j], valid_S[k]):
                        valid_S.pop(k)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    return valid_S
