import numpy as np
import random
import time
import math

N = 64
TOTAL_POINTS = N * N

def construct():
    # Precompute coordinates
    coords = np.array([(x, y) for x in range(N) for y in range(N)])
    
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Compute dist2 as int32
    dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
    for i in range(TOTAL_POINTS):
        dx = x_coords[i] - x_coords
        dy = y_coords[i] - y_coords
        dist2[i] = dx*dx + dy*dy

    random.seed(42)
    
    # Simulated Annealing
    # State: in_set (boolean array), S (list of indices in set)
    # For each i in S, dist_counts[i] is a dict: distance -> count
    
    best_set = []
    best_score = -1
    
    # Initial state: Random small set or empty
    # Let's start with empty and grow
    S = []
    in_set = np.zeros(TOTAL_POINTS, dtype=bool)
    dist_counts = {} # dict from index to dict of dist counts
    
    def get_score(S, dist_counts, energy):
        # We need to track energy globally or compute it?
        # Better to track energy incrementally.
        pass
        
    # Let's maintain global energy E
    # And size K
    K = 0
    E = 0
    
    # Helper to add point
    def try_add(p):
        nonlocal K, E
        if in_set[p]:
            return False, 0
        
        # Calculate change in energy
        delta_E = 0
        
        # Distances from p to all in S
        d_vals = dist2[p, S]
        
        # 1. Conflicts within p's new distances
        # Count duplicates in d_vals
        if len(d_vals) > 0:
            counts = {}
            for d in d_vals:
                counts[d] = counts.get(d, 0) + 1
            for c in counts.values():
                if c > 1:
                    delta_E += c - 1
        
        # 2. Conflicts added to existing points in S
        for i, q in enumerate(S):
            d = d_vals[i]
            # Check if q already has distance d
            # dist_counts[q] is the dict
            c_q = dist_counts[q].get(d, 0)
            if c_q >= 1:
                delta_E += 1
        
        return True, delta_E

    # Helper to remove point
    def try_remove(p):
        nonlocal K, E
        if not in_set[p]:
            return False, 0
            
        delta_E = 0
        S_copy = S[:]
        S_copy.remove(p)
        
        # Distances from p to all in S_copy
        # Note: S in the closure is the current S including p
        # We need distances to S \ {p}
        # d_vals = dist2[p, [x for x in S if x != p]]
        
        # 1. Remove p's internal conflicts
        # We need the counts of distances from p to other points in S
        d_vals = dist2[p, [x for x in S if x != p]]
        if len(d_vals) > 0:
            counts = {}
            for d in d_vals:
                counts[d] = counts.get(d, 0) + 1
            for c in counts.values():
                if c > 1:
                    delta_E -= (c - 1)
        
        # 2. Remove conflicts from existing points
        for q in S:
            if q == p: continue
            d = dist2[p, q]
            c_q = dist_counts[q].get(d, 0)
            # c_q is the count BEFORE removal (including p)
            # If c_q >= 2, then removing p reduces the count from c_q to c_q-1
            # Old term: max(0, c_q - 1)
            # New term: max(0, c_q - 2)
            # Diff: max(0, c_q-2) - max(0, c_q-1)
            # If c_q == 1: 0 - 0 = 0
            # If c_q >= 2: (c_q-2) - (c_q-1) = -1
            if c_q >= 2:
                delta_E -= 1
                
        return True, delta_E

    # Actual Add
    def do_add(p):
        nonlocal K, E
        in_set[p] = True
        S.append(p)
        K += 1
        
        d_vals = dist2[p, S[:-1]] # Distances to others in S (excluding p)
        
        # Update E
        # 1. p's internal conflicts
        if len(d_vals) > 0:
            counts = {}
            for d in d_vals:
                counts[d] = counts.get(d, 0) + 1
            for c in counts.values():
                if c > 1:
                    E += c - 1
        
        # 2. Update dist_counts for existing points and E
        for i, q in enumerate(S[:-1]):
            d = d_vals[i]
            c_q = dist_counts[q].get(d, 0)
            if c_q >= 1:
                E += 1
            dist_counts[q][d] = c_q + 1
        
        # Initialize dist_counts for p
        dist_counts[p] = {}
        for d in d_vals:
            dist_counts[p][d] = dist_counts[p].get(d, 0) + 1

    # Actual Remove
    def do_remove(p):
        nonlocal K, E
        in_set[p] = False
        S.remove(p)
        K -= 1
        
        # Get distances to remaining S
        others = [x for x in S]
        d_vals = dist2[p, others] if others else np.array([])
        
        # 1. Remove p's internal conflicts
        if len(d_vals) > 0:
            counts = {}
            for d in d_vals:
                counts[d] = counts.get(d, 0) + 1
            for c in counts.values():
                if c > 1:
                    E -= (c - 1)
        
        # 2. Update dist_counts for remaining points and E
        for q in others:
            d = dist2[p, q]
            c_q = dist_counts[q].get(d, 0)
            # c_q is current count (including p)
            if c_q >= 2:
                E -= 1
            dist_counts[q][d] = c_q - 1
            if dist_counts[q][d] == 0:
                del dist_counts[q][d]
        
        # Remove p's dist_counts
        del dist_counts[p]

    # Simulated Annealing Loop
    start_time = time.time()
    time_limit = 280
    
    # Initialize with a random greedy construction to get a decent start
    # Greedy: add points that increase E by 0
    random.shuffle(range(TOTAL_POINTS))
    for p in range(TOTAL_POINTS):
        if time.time() - start_time > time_limit - 60: break
        can_add, dE = try_add(p)
        if can_add and dE == 0:
            do_add(p)
            
    current_score = K - 1000 * E
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        T = max(T * cooling, T_min)
        
        # Choose a move
        move_type = random.random()
        if move_type < 0.4:
            # Add
            if K < TOTAL_POINTS:
                # Pick random candidate not in set
                # Optimization: Pick from a pool? Just random int
                p = random.randint(0, TOTAL_POINTS - 1)
                if not in_set[p]:
                    can_add, dE = try_add(p)
                    if can_add:
                        old_score = K - 1000 * E
                        do_add(p)
                        new_score = K - 1000 * E
                        delta = new_score - old_score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            if new_score > current_score:
                                current_score = new_score
                                if E == 0:
                                    best_set = S[:]
                                    best_score = K
                        else:
                            do_remove(p)
        elif move_type < 0.7:
            # Remove
            if K > 0:
                p = random.choice(S)
                can_remove, dE = try_remove(p)
                if can_remove:
                    old_score = K - 1000 * E
                    do_remove(p)
                    new_score = K - 1000 * E
                    delta = new_score - old_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        if new_score > current_score:
                            current_score = new_score
                            if E == 0:
                                best_set = S[:]
                                best_score = K
                    else:
                        do_add(p)
        else:
            # Swap
            if K > 0 and K < TOTAL_POINTS:
                p_in = random.choice(S)
                p_out = random.randint(0, TOTAL_POINTS - 1)
                while in_set[p_out]:
                    p_out = random.randint(0, TOTAL_POINTS - 1)
                
                # Remove p_in, Add p_out
                # Calculate delta for remove
                can_remove, dE_remove = try_remove(p_in)
                if can_remove:
                    do_remove(p_in)
                    
                    # Calculate delta for add
                    can_add, dE_add = try_add(p_out)
                    if can_add:
                        do_add(p_out)
                        # Score change:
                        # K is same. E changed by dE_remove + dE_add?
                        # try_remove and try_add calculate dE based on state BEFORE the op.
                        # So total dE = dE_remove + dE_add (approx, since state changed after remove)
                        # Actually, try_add uses current S (after remove). So it is correct.
                        # Score change = (K - 1000*(E + dE_remove + dE_add)) - (K - 1000*E) = -1000*(dE_remove + dE_add)
                        # Wait, try_remove returns dE which is the change in E.
                        # So new E = E + dE_remove + dE_add.
                        delta = -1000 * (dE_remove + dE_add)
                        
                        if delta > 0 or random.random() < math.exp(delta / T):
                            if current_score < K - 1000 * E:
                                current_score = K - 1000 * E
                                if E == 0:
                                    best_set = S[:]
                                    best_score = K
                        else:
                            # Revert: Remove p_out, Add p_in
                            do_remove(p_out)
                            do_add(p_in)
                    else:
                        # Revert remove
                        do_add(p_in)

    return [tuple(coords[i]) for i in best_set]
