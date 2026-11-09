"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def _solve_for_n(n, time_limit):
        """
        Attempt to find a valid coloring for a given n using local search.
        Returns (colors, success)
        """
        # Precompute triple indices
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple (a,b,c) to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute for each triple index, the list of 4-sets and 5-sets it belongs to
        # This is crucial for incremental updates
        # A 4-set containing triple (a,b,c) is formed by adding one vertex d.
        # Since a<b<c, d can be any vertex not in {a,b,c}.
        # To maintain canonical form for 4-sets and 5-sets, we just need to identify the set.
        
        # For 4-sets:
        # A 4-set is a tuple of 4 sorted vertices.
        # If we add d to (a,b,c), the sorted set is sorted((a,b,c,d)).
        
        # For 5-sets:
        # A 5-set is a tuple of 5 sorted vertices.
        # If we add d, e to (a,b,c), the sorted set is sorted((a,b,c,d,e)).
        
        # To make updates fast, we store for each triple index:
        # - list of 4-set indices it belongs to
        # - list of 5-set indices it belongs to
        
        # First, we need to index all 4-sets and 5-sets.
        all_4sets = list(combinations(range(n), 4))
        all_5sets = list(combinations(range(n), 5))
        
        num_4sets = len(all_4sets)
        num_5sets = len(all_5sets)
        
        # Map 4-set tuple to index
        set4_to_idx = {s: i for i, s in enumerate(all_4sets)}
        # Map 5-set tuple to index
        set5_to_idx = {s: i for i, s in enumerate(all_5sets)}
        
        # Build incidence lists: triple_idx -> list of 4set_idxs, list of 5set_idxs
        triple_to_4sets = [[] for _ in range(num_triples)]
        triple_to_5sets = [[] for _ in range(num_triples)]
        
        # Iterate over all 4-sets and add them to their constituent triples
        for s4_idx, s4 in enumerate(all_4sets):
            # 4 triples in a 4-set
            for t in combinations(s4, 3):
                t_idx = triple_to_idx[t]
                triple_to_4sets[t_idx].append(s4_idx)
                
        # Iterate over all 5-sets and add them to their constituent triples
        for s5_idx, s5 in enumerate(all_5sets):
            # 10 triples in a 5-set
            for t in combinations(s5, 3):
                t_idx = triple_to_idx[t]
                triple_to_5sets[t_idx].append(s5_idx)
                
        # State variables
        # colors: 0 for Red, 1 for Blue
        # red_count_4[s4_idx]: number of red triples in this 4-set
        # blue_count_5[s5_idx]: number of blue triples in this 5-set
        
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Initialize counts
        red_count_4 = np.zeros(num_4sets, dtype=np.int16)
        blue_count_5 = np.zeros(num_5sets, dtype=np.int16)
        
        # Initial calculation
        for t_idx, t in enumerate(triples):
            c = colors[t_idx]
            for s4_idx in triple_to_4sets[t_idx]:
                if c == 0:
                    red_count_4[s4_idx] += 1
            for s5_idx in triple_to_5sets[t_idx]:
                if c == 1:
                    blue_count_5[s5_idx] += 1
                    
        # Calculate initial violations
        # Violation if red_count_4 == 4 or blue_count_5 == 10
        total_violations = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
        
        best_violations = total_violations
        best_colors = colors.copy()
        
        # Local Search
        # We want to minimize total_violations.
        # If we flip a triple t from 0 to 1:
        #   - For each 4-set containing t: red_count_4 decreases by 1.
        #     - If it was 4, it becomes 3 (violation removed).
        #     - If it was 3, it becomes 2 (no change in violation status).
        #     - If it was <3, no change.
        #     - Note: It cannot increase red count.
        #   - For each 5-set containing t: blue_count_5 increases by 1.
        #     - If it was 10, it becomes 11 (violation removed? No, 10 is violation. 11 is not a violation in the sense of "all blue", wait.
        #       The condition is "all 10 triples blue". So if count is 10, it is a violation.
        #       If we flip a blue triple to red, blue_count decreases.
        #       If we flip a red triple to blue, blue_count increases.
        #       
        # Let's re-evaluate the delta calculation carefully.
        # Current state: colors[t] is either 0 or 1.
        # New state: colors[t] is 1 or 0.
        
        # Case 1: Flip 0 (Red) -> 1 (Blue)
        #   - Red counts in 4-sets containing t: decrease by 1.
        #     - Violation (Red K4) is removed if count was 4. (4->3)
        #     - Violation (Red K4) is created if count was 3? No, 3 is not a violation.
        #     - So, Delta_R = - (number of 4-sets with count 4)
        #   - Blue counts in 5-sets containing t: increase by 1.
        #     - Violation (Blue K5) is created if count was 9. (9->10)
        #     - Violation (Blue K5) is removed if count was 10? No, if it was 10, it's already a violation.
        #       Wait, if count is 10, all are blue. If we flip one to red, count becomes 9. Violation removed.
        #       So, if count was 10, Delta_B = -1 (violation removed).
        #       If count was 9, Delta_B = +1 (violation created).
        #       If count was <9, Delta_B = 0.
        
        # Case 2: Flip 1 (Blue) -> 0 (Red)
        #   - Red counts in 4-sets containing t: increase by 1.
        #     - If count was 3, Delta_R = +1 (violation created).
        #     - If count was 4, Delta_R = -1 (violation removed).
        #   - Blue counts in 5-sets containing t: decrease by 1.
        #     - If count was 9, Delta_B = +1 (violation created? No, 9 is not violation. 10 is. So 9->8 is no change).
        #       Wait. If count is 9, it's not a violation. If we flip blue to red, count becomes 8. Still not violation.
        #       If count is 10, it is a violation. If we flip blue to red, count becomes 9. Violation removed.
        #       So, if count was 10, Delta_B = -1.
        #       If count was <10, Delta_B = 0.
        
        # Let's formalize the delta calculation for a flip of triple t_idx.
        
        def calc_delta(t_idx, current_color):
            delta = 0
            # 4-sets
            for s4_idx in triple_to_4sets[t_idx]:
                cnt = red_count_4[s4_idx]
                if current_color == 0: # Flipping to Blue (Red count decreases)
                    if cnt == 4:
                        delta -= 1 # Violation removed
                    # If cnt < 4, no violation change
                else: # Flipping to Red (Red count increases)
                    if cnt == 3:
                        delta += 1 # Violation created
                    # If cnt == 4, it was already a violation. Becomes 5? No, max is 4.
                    # Wait, if cnt is 4, all 4 are red. We are flipping one to red? No, we are flipping FROM blue TO red.
                    # So current_color is 1. The triple was Blue.
                    # If we flip it to Red, red_count increases.
                    # If red_count was 3, it becomes 4. Violation created.
                    # If red_count was 4, it means all 4 were already red. But this triple was Blue. Contradiction.
                    # So if current_color is 1, this triple is NOT red. So it is not included in red_count_4.
                    # So red_count_4[s4_idx] counts the OTHER 3 triples.
                    # Max count for other 3 is 3.
                    # So if cnt == 3, delta += 1.
            
            # 5-sets
            for s5_idx in triple_to_5sets[t_idx]:
                cnt = blue_count_5[s5_idx]
                if current_color == 0: # Flipping to Blue (Blue count increases)
                    # This triple was Red, so not in blue_count.
                    # If we flip to Blue, blue_count increases by 1.
                    if cnt == 9:
                        delta += 1 # Violation created (9->10)
                    # If cnt == 10, impossible because this triple was Red.
                else: # Flipping to Red (Blue count decreases)
                    # This triple was Blue, so in blue_count.
                    # If we flip to Red, blue_count decreases by 1.
                    if cnt == 10:
                        delta -= 1 # Violation removed (10->9)
                    # If cnt < 10, no violation change (since <10 is not violation)
                    
            return delta

        # Main Loop
        # We use a simple hill climbing with random restarts if stuck.
        # To be faster, we can try to pick triples that are in many violations.
        # But random is often good enough with incremental updates.
        
        stuck_count = 0
        max_stuck = 5000
        num_restarts = 0
        max_restarts = 50
        
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors.tolist(), True
            
            # Try to find an improving move
            improved = False
            # Pick a random triple
            # We can bias towards triples in violating sets, but random is fine for now.
            t_idx = random.randint(0, num_triples - 1)
            current_color = colors[t_idx]
            
            d = calc_delta(t_idx, current_color)
            
            if d < 0:
                # Apply flip
                colors[t_idx] = 1 - current_color
                new_color = 1 - current_color
                
                # Update counts
                for s4_idx in triple_to_4sets[t_idx]:
                    if current_color == 0: # Was Red, now Blue
                        red_count_4[s4_idx] -= 1
                    else: # Was Blue, now Red
                        red_count_4[s4_idx] += 1
                        
                for s5_idx in triple_to_5sets[t_idx]:
                    if current_color == 0: # Was Red, now Blue
                        blue_count_5[s5_idx] += 1
                    else: # Was Blue, now Red
                        blue_count_5[s5_idx] -= 1
                        
                total_violations += d
                improved = True
                stuck_count = 0
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    # Restart with a new random coloring
                    num_restarts += 1
                    if num_restarts > max_restarts:
                        break
                    colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                    red_count_4 = np.zeros(num_4sets, dtype=np.int16)
                    blue_count_5 = np.zeros(num_5sets, dtype=np.int16)
                    for t_idx, t in enumerate(triples):
                        c = colors[t_idx]
                        for s4_idx in triple_to_4sets[t_idx]:
                            if c == 0:
                                red_count_4[s4_idx] += 1
                        for s5_idx in triple_to_5sets[t_idx]:
                            if c == 1:
                                blue_count_5[s5_idx] += 1
                    total_violations = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
                    stuck_count = 0

        return best_colors.tolist(), False

    # Try to solve for n=34 first
    # If that fails, try 33, 32, etc.
    for n in range(34, 8, -1):
        if time.time() > deadline - 5:
            break
            
        print(f"Attempting n={n}")
        time_left = deadline - time.time()
        # Allocate most of the time to the largest n
        if n == 34:
            t_alloc = time_left
        else:
            t_alloc = min(time_left, 10) # Give smaller n less time
            
        colors, success = _solve_for_n(n, t_alloc)
        
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a small valid coloring if we didn't find a large one
    # n=9 with mod 3 is a known small example?
    # Let's just return n=1 to be safe if search fails completely
    # Or try a very quick random search for small n
    for n in range(9, 4, -1):
        if time.time() > deadline - 1:
            break
        # Quick check for small n
        # Just return n=1 if nothing else works
        pass
        
    return {'n': 1, 'colors': [0]}

# EVOLVE-BLOCK-END
