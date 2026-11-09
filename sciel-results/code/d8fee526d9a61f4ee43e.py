import random
import time

N = 64
M = N * N

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Pre-compute coordinates for fast access
    # points[i] = (x, y)
    
    def sq_dist(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    def try_add(S, D, p):
        """
        Returns True if p can be added to S.
        Updates S and D in place if successful.
        """
        if p in S:
            return False
            
        valid = True
        dists_from_p = set()
        
        for s in S:
            d = sq_dist(p, s)
            
            # Check if p is apex: distance to s must be unique
            if d in dists_from_p:
                valid = False
                break
                
            dists_from_p.add(d)
            
            # Check if s is apex: distance p-s must not exist in s's distance set
            if d in D[s]:
                valid = False
                break
        
        if valid:
            S.add(p)
            D[p] = dists_from_p
            # Update distance sets for all existing points
            for s in S:
                if s != p:
                    D[s].add(sq_dist(p, s))
            return True
        return False

    def greedy_build(order, time_limit):
        S = set()
        D = {}
        for p in order:
            if time.time() > time_limit:
                break
            try_add(S, D, p)
        return S, D

    best_S = set()
    best_size = 0

    # Phase 1: Greedy Restarts
    # We run several greedy restarts to find a good starting point.
    t_phase1_end = time.time() + 10
    restarts = 0
    while time.time() < t_phase1_end and restarts < 50:
        order = list(range(M))
        random.shuffle(order)
        S, D = greedy_build(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            best_D = {k: set(v) for k, v in D.items()}
        restarts += 1

    # Phase 2: Local Search
    S = best_S.copy()
    D = {k: set(v) for k, v in best_D.items()}
    
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 5:
        improved = False
        
        # Try to add any point
        # We iterate through all points not in S
        # To make this faster, we can keep a list of candidates
        # But M=4096 is small, so a full scan is ~4096 * |S| ops.
        # If |S| is 100, that's 400k ops. Very fast.
        
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        for p in candidates:
            if try_add(S, D, p):
                improved = True
                # If we added a point, we break and restart the outer loop
                # because the set of candidates has changed
                break
        
        if not improved:
            # Stuck. Perturb.
            if len(S) > 0:
                # Remove 1 to 3 random points
                remove_count = random.randint(1, 3)
                to_remove = random.sample(list(S), min(remove_count, len(S)))
                for r in to_remove:
                    S.discard(r)
                    if r in D:
                        del D[r]
                    for s in S:
                        if s in D:
                            D[s].discard(sq_dist(r, s))
                
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Full restart to escape deep local minima
                    order = list(range(M))
                    random.shuffle(order)
                    S, D = greedy_build(order, deadline)
                    no_improve_count = 0
            else:
                # Empty set, restart
                order = list(range(M))
                random.shuffle(order)
                S, D = greedy_build(order, deadline)
                no_improve_count = 0
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_D = {k: set(v) for k, v in D.items()}
        else:
            no_improve_count = 0
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_D = {k: set(v) for k, v in D.items()}

    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
