import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit
    
    # Precompute coordinates
    coords = [(i // N, i % N) for i in range(NUM_POINTS)]
    
    # S: list of point indices in the current set
    S = []
    in_set = [False] * NUM_POINTS
    # dist_from[i] is a set of squared distances from i to all other points in S
    dist_from = [set() for _ in range(NUM_POINTS)]
    
    best_S = []
    best_size = 0
    
    def calc_d2(p, q):
        px, py = coords[p]
        qx, qy = coords[q]
        dx = px - qx
        dy = py - qy
        return dx*dx + dy*dy

    def is_valid_add(p, S, dist_from):
        """
        Check if p can be added to S.
        Conditions:
        1. d(p, a) must be distinct for all a in S.
        2. d(p, a) must not be in dist_from[a] for any a in S.
           (i.e., d(p, a) != d(a, b) for any b in S, b != a)
        """
        d2s = set()
        for a in S:
            d2_pa = calc_d2(p, a)
            if d2_pa in d2s:
                return False
            d2s.add(d2_pa)
            
            if d2_pa in dist_from[a]:
                return False
                
        return True

    def do_add(p, S, in_set, dist_from):
        S.append(p)
        in_set[p] = True
        dist_from[p] = set()
        for a in S[:-1]:
            d2 = calc_d2(p, a)
            dist_from[p].add(d2)
            dist_from[a].add(d2)

    def do_remove(p, S, in_set, dist_from):
        # Find index efficiently if S is small, but S.index is O(K).
        # For K ~ 20-30, this is fine.
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        
        for a in S:
            d2 = calc_d2(p, a)
            dist_from[a].discard(d2)
        
        dist_from[p] = set()

    def greedy_init(order):
        S.clear()
        for i in range(NUM_POINTS):
            in_set[i] = False
        for i in range(NUM_POINTS):
            dist_from[i] = set()
            
        for p in order:
            if is_valid_add(p, S, dist_from):
                do_add(p, S, in_set, dist_from)

    # 1. Greedy Initialization with multiple random orders to find a good starting point
    best_init_size = 0
    best_init_S = []
    
    # Try a few random greedy passes
    for _ in range(5):
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        greedy_init(order)
        if len(S) > best_init_size:
            best_init_size = len(S)
            best_init_S = S[:]
            
    # Set current state to best init
    S = best_init_S[:]
    for i in range(NUM_POINTS):
        in_set[i] = False
    for i in range(NUM_POINTS):
        dist_from[i] = set()
    for p in S:
        in_set[p] = True
    # Rebuild dist_from for the set
    for i, p in enumerate(S):
        for j in range(i):
            q = S[j]
            d2 = calc_d2(p, q)
            dist_from[p].add(d2)
            dist_from[q].add(d2)
            
    best_size = len(S)
    best_S = S[:]
    
    # 2. Local Search: Conflict-Driven Swaps
    # We maintain a list of "candidates" that failed to add.
    # Instead of random swaps, we try to resolve conflicts.
    
    # Precompute a list of all points not in S
    # We will iterate through points not in S and try to add them.
    # If they fail, we identify the "blocking" points and try to swap them.
    
    # To make this efficient, we keep a set of points to try.
    # We'll use a priority queue or just a shuffled list of non-S points.
    
    # We'll run this for a fixed number of iterations or until time limit.
    # Each "step" is: pick a random point not in S.
    # If valid, add it.
    # If invalid, find a random point in S that conflicts, and try to swap it with the new point?
    # Actually, a better move:
    # If p is invalid because of conflict with a in S, try to find a p' not in S such that:
    # 1. p' is valid with S \ {a}
    # 2. p' is "better" or just different?
    # This is hard to check directly.
    
    # Alternative: Standard 1-Swap Local Search with Smarter Candidate Selection.
    # We maintain a "frontier" of points that are "close" to being valid.
    
    # Let's stick to a robust 1-Swap SA but with a twist:
    # We don't just pick random p_in. We pick p_in from a small neighborhood
    # of points that have high "potential" (e.g., low distance to existing set? No, that increases conflict).
    # Actually, points far away might have unique distances.
    
    # Let's try: Random restarts + Greedy + 1-Swap SA.
    # The key improvement over the previous code:
    # 1. Faster is_valid_add (already done).
    # 2. Better initialization (multiple greedy passes).
    # 3. In SA, if a swap is rejected, we don't just cool down. We keep the state.
    # 4. We add a "kick" mechanism: if no improvement for X steps, remove a random point to allow re-entry.
    
    T = 1.0
    alpha = 0.9995
    min_T = 0.001
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and T > min_T:
        # 1. Try to add a random point
        if random.random() < 0.4:
            p = random.randint(0, NUM_POINTS - 1)
            if not in_set[p]:
                if is_valid_add(p, S, dist_from):
                    do_add(p, S, in_set, dist_from)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        else:
            # 2. Try a swap
            if S:
                p_out = random.choice(S)
                p_in = random.randint(0, NUM_POINTS - 1)
                
                if in_set[p_in]:
                    continue
                
                # Check if p_in is valid with S \ {p_out}
                # This is the expensive part.
                # We can shortcut: if p_in is valid with S, it's valid with S\{p_out}.
                # But p_in is likely NOT valid with S (that's why we swap).
                
                # Efficient Check:
                d2s = set()
                valid = True
                for a in S:
                    if a == p_out:
                        continue
                    d2 = calc_d2(p_in, a)
                    if d2 in d2s:
                        valid = False
                        break
                    d2s.add(d2)
                    
                    # Check if d2 is in dist_from[a]
                    # But we must exclude the distance to p_out.
                    d2_out = calc_d2(p_out, a)
                    if d2 in dist_from[a] and d2 != d2_out:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    # Remove p_out
                    idx = S.index(p_out)
                    S.pop(idx)
                    in_set[p_out] = False
                    for a in S:
                        d2 = calc_d2(p_out, a)
                        dist_from[a].discard(d2)
                    dist_from[p_out] = set()
                    
                    # Add p_in
                    S.append(p_in)
                    in_set[p_in] = True
                    dist_from[p_in] = set()
                    for a in S[:-1]:
                        d2 = calc_d2(p_in, a)
                        dist_from[p_in].add(d2)
                        dist_from[a].add(d2)
                    
                    no_improve_count += 1
                else:
                    no_improve_count += 1
        
        # 3. Kick mechanism: If stuck, remove a random point to break symmetry
        if no_improve_count > max_no_improve:
            if S:
                p = random.choice(S)
                do_remove(p, S, in_set, dist_from)
                no_improve_count = 0
                # Don't update best here as size decreased
        
        T *= alpha
        
        if time.time() > deadline - 5:
            break

    return [coords[i] for i in best_S]
