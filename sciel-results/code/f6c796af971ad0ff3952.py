import random
import time
from itertools import combinations
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a valid 2-coloring of triples for n=34 such that:
    - No red K4^(3) (4 vertices, all 4 triples red)
    - No blue K5^(3) (5 vertices, all 10 triples blue)
    Returns {'n': n, 'colors': list of 0s and 1s}
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 34
    # Generate all triples (a, b, c) with a < b < c
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute for each triple, which 4-sets and 5-sets it belongs to
    # A triple (a,b,c) is part of a 4-set (a,b,c,d) where d is any other vertex
    # A triple (a,b,c) is part of a 5-set (a,b,c,d,e) where d,e are any other two vertices
    
    # For efficiency, we will maintain:
    # red_counts_4: array of size C(n,4), counts red triples in each 4-set
    # blue_counts_5: array of size C(n,5), counts blue triples in each 5-set
    
    # Generate 4-sets and 5-sets
    four_sets = list(combinations(range(n), 4))
    five_sets = list(combinations(range(n), 5))
    num_four_sets = len(four_sets)
    num_five_sets = len(five_sets)
    
    four_set_to_idx = {s: i for i, s in enumerate(four_sets)}
    five_set_to_idx = {s: i for i, s in enumerate(five_sets)}
    
    # For each triple index, store the indices of the 4-sets and 5-sets it affects
    # And whether the triple is the "extra" element or part of the base?
    # Actually, for a 4-set {a,b,c,d}, the triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d).
    # If we flip triple (a,b,c), it affects the 4-set {a,b,c,d}.
    
    # Let's build adjacency:
    # triple_4sets[i] = list of (four_set_idx, sign) where sign is +1 if adding red increases count, etc.
    # Actually, we just need to update the count.
    # If triple i is flipped:
    # - If it was red and becomes blue: red_count[4set] decreases by 1.
    # - If it was blue and becomes red: red_count[4set] increases by 1.
    
    # Similarly for blue counts in 5-sets.
    
    # Precomputation
    triple_4sets = [[] for _ in range(num_triples)]
    triple_5sets = [[] for _ in range(num_triples)]
    
    # This precomputation might be slow if done naively.
    # C(34,3) = 5984 triples.
    # Each triple is in C(31,1)=31 four-sets and C(31,2)=465 five-sets.
    # Total links: 5984 * (31 + 465) ~ 3M links. This is manageable.
    
    # To speed up, we can iterate over 4-sets and 5-sets and add to triples.
    # But building the reverse map is O(1) per link if we have the set-to-idx map.
    
    for i, s4 in enumerate(four_sets):
        # Triples in this 4-set
        for t in combinations(s4, 3):
            idx = triple_to_idx[t]
            triple_4sets[idx].append(i)
            
    for i, s5 in enumerate(five_sets):
        # Triples in this 5-set
        for t in combinations(s5, 3):
            idx = triple_to_idx[t]
            triple_5sets[idx].append(i)
            
    # State: colors array of 0/1 (0=red, 1=blue)
    # Counts:
    red_counts_4 = [0] * num_four_sets
    blue_counts_5 = [0] * num_five_sets
    
    # Violations:
    # Red violation if red_counts_4[i] == 4
    # Blue violation if blue_counts_5[i] == 10
    
    # We want to minimize total violations.
    
    def init_state():
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        red_counts_4 = [0] * num_four_sets
        blue_counts_5 = [0] * num_five_sets
        
        for i, c in enumerate(colors):
            if c == 0: # Red
                for fs_idx in triple_4sets[i]:
                    red_counts_4[fs_idx] += 1
            else: # Blue
                for fs_idx in triple_5sets[i]:
                    blue_counts_5[fs_idx] += 1
                    
        # Count violations
        red_viol = sum(1 for c in red_counts_4 if c == 4)
        blue_viol = sum(1 for c in blue_counts_5 if c == 10)
        return colors, red_counts_4, blue_counts_5, red_viol + blue_viol

    def flip(triple_idx, colors, red_counts_4, blue_counts_5, red_viol, blue_viol):
        c = colors[triple_idx]
        # Update counts
        if c == 0: # Was Red, becomes Blue
            # Red counts decrease
            for fs_idx in triple_4sets[triple_idx]:
                old = red_counts_4[fs_idx]
                if old == 4:
                    red_viol -= 1
                red_counts_4[fs_idx] = old - 1
                # No new red violation created by decreasing
            # Blue counts increase
            for fs_idx in triple_5sets[triple_idx]:
                old = blue_counts_5[fs_idx]
                blue_counts_5[fs_idx] = old + 1
                if old + 1 == 10:
                    blue_viol += 1
            colors[triple_idx] = 1
        else: # Was Blue, becomes Red
            # Blue counts decrease
            for fs_idx in triple_5sets[triple_idx]:
                old = blue_counts_5[fs_idx]
                if old == 10:
                    blue_viol -= 1
                blue_counts_5[fs_idx] = old - 1
                # No new blue violation created by decreasing
            # Red counts increase
            for fs_idx in triple_4sets[triple_idx]:
                old = red_counts_4[fs_idx]
                red_counts_4[fs_idx] = old + 1
                if old + 1 == 4:
                    red_viol += 1
            colors[triple_idx] = 0
            
        return red_viol, blue_viol

    best_viol = float('inf')
    best_colors = None
    
    max_restarts = 5000
    max_steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        colors, red_counts_4, blue_counts_5, total_viol = init_state()
        
        if total_viol < best_viol:
            best_viol = total_viol
            best_colors = colors[:]
            if best_viol == 0:
                break
                
        steps = 0
        while steps < max_steps_per_restart and total_viol > 0:
            if time.time() > deadline:
                break
                
            # Find a triple to flip that reduces violations
            # We can pick a random violation and try to fix it.
            # Or pick a random triple and see if flipping helps.
            
            # Heuristic: Pick a random red violation 4-set, pick a random triple in it, flip it.
            # Or pick a random blue violation 5-set, pick a random triple in it, flip it.
            
            # Let's try: Pick a random triple. Calculate delta violations. If delta < 0, flip.
            # To speed up, we can sample a few triples.
            
            improved = False
            for _ in range(10): # Try 10 random triples
                t_idx = random.randint(0, num_triples - 1)
                c = colors[t_idx]
                
                delta = 0
                if c == 0: # Red -> Blue
                    # Red 4-sets decrease
                    for fs_idx in triple_4sets[t_idx]:
                        if red_counts_4[fs_idx] == 4:
                            delta -= 1
                    # Blue 5-sets increase
                    for fs_idx in triple_5sets[t_idx]:
                        if blue_counts_5[fs_idx] == 9: # Becomes 10
                            delta += 1
                else: # Blue -> Red
                    # Blue 5-sets decrease
                    for fs_idx in triple_5sets[t_idx]:
                        if blue_counts_5[fs_idx] == 10:
                            delta -= 1
                    # Red 4-sets increase
                    for fs_idx in triple_4sets[t_idx]:
                        if red_counts_4[fs_idx] == 3: # Becomes 4
                            delta += 1
                
                if delta < 0:
                    red_viol, blue_viol = flip(t_idx, colors, red_counts_4, blue_counts_5, red_viol, blue_viol)
                    total_viol += delta
                    improved = True
                    break
            if not improved:
                # Stuck, restart or random walk?
                # Let's do a few random flips to escape local minimum
                for _ in range(5):
                    t_idx = random.randint(0, num_triples - 1)
                    red_viol, blue_viol = flip(t_idx, colors, red_counts_4, blue_counts_5, red_viol, blue_viol)
                    # Recalculate total_viol approximately or exactly?
                    # For safety, recalculate exactly if we care, but delta tracking is better.
                    # Since we don't track delta for random flips, we should recalculate or track carefully.
                    # Let's just break and restart if we can't improve.
                break
            
            steps += 1
            
            if total_viol < best_viol:
                best_viol = total_viol
                best_colors = colors[:]
                if best_viol == 0:
                    break
    
    if best_colors is None:
        # Fallback to seed if no better found (should not happen)
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
    
    return {'n': n, 'colors': best_colors}

# EVOLVE-BLOCK-END
