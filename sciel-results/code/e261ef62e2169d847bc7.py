import time
import random

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290

    points = [(x, y) for x in range(N) for y in range(N)]

    def sq_dist(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    def try_add(S, D, p):
        """
        Returns True if p can be added.
        Updates S and D if successful.
        """
        if p in S:
            return False
            
        valid = True
        dists_from_p = set()
        
        for s in S:
            d = sq_dist(p, s)
            
            if d in dists_from_p:
                valid = False
                break
                
            dists_from_p.add(d)
            
            if d in D[s]:
                valid = False
                break
        
        if valid:
            S.add(p)
            D[p] = dists_from_p
            for s in S:
                if s != p:
                    D[s].add(sq_dist(p, s))
            return True
        return False

    def greedy_from(S_init, D_init, candidates, time_limit):
        """
        Greedily add points from candidates to S_init.
        """
        S = set(S_init)
        D = {k: set(v) for k, v in D_init.items()}
        
        # Shuffle candidates to avoid bias
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for p in cand_list:
            if time.time() > time_limit:
                break
            if p not in S:
                try_add(S, D, p)
                
        return S, D

    best_S = set()
    best_size = 0
    
    # Phase 1: Random Greedy Restarts
    # We do this for a while
    t_phase1_end = start_time + 60
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, D = greedy_from(set(), {}, order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            best_D = {k: set(v) for k, v in D.items()}

    # Phase 2: Local Search / Perturbation on the best solution
    # We try to remove some points and re-add better ones
    
    S = best_S.copy()
    D = {k: set(v) for k, v in best_D.items()}
    
    # List of points NOT in S
    # We need to efficiently find candidates.
    # Since M is 4096, we can just iterate all points and check if in S.
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try to add any point
        # Iterate through all points, check if can add
        # This is O(M * |S|). 4096 * 100 = 400k. Fast.
        added_any = False
        # Shuffle to avoid always adding the same "easy" point
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        for p in candidates:
            if try_add(S, D, p):
                improved = True
                added_any = True
                break # Add one, then restart loop to re-evaluate? 
                # Or keep adding?
                # If we keep adding, we might block others.
                # Let's just add one and loop back.
        
        if not improved:
            # Stuck. Perturb.
            if len(S) > 0:
                # Remove 1 to 3 random points
                remove_count = random.randint(1, 3)
                to_remove = random.sample(list(S), min(remove_count, len(S)))
                for r in to_remove:
                    S.discard(r)
                    del D[r]
                    for s in S:
                        D[s].discard(sq_dist(r, s))
                
                no_improve += 1
                if no_improve > max_no_improve:
                    # Full restart
                    order = list(range(M))
                    random.shuffle(order)
                    S, D = greedy_from(set(), {}, order, deadline)
                    no_improve = 0
            else:
                # Empty set, restart
                order = list(range(M))
                random.shuffle(order)
                S, D = greedy_from(set(), {}, order, deadline)
                no_improve = 0
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_D = {k: set(v) for k, v in D.items()}
        else:
            no_improve = 0
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_D = {k: set(v) for k, v in D.items()}

    return [points[i] for i in best_S]
