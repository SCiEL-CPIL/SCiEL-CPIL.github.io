import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def precompute(n):
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        # Map each triple to the indices of the 4-sets and 5-sets it belongs to
        # A 4-set (a,b,c,d) contains 4 triples
        # A 5-set (a,b,c,d,e) contains 10 triples
        triple_to_4sets = [[] for _ in range(len(triples))]
        triple_to_5sets = [[] for _ in range(len(triples))]
        
        for i, s4 in enumerate(four_sets):
            a, b, c, d = s4
            for t in combinations(s4, 3):
                triple_to_4sets[triple_index[t]].append(i)
                
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def solve(n, time_limit):
        if n < 4:
            return [0] * 0
            
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # 0 = Red, 1 = Blue
        # Strategy: Start with all Blue (1). This creates many Red violations? No, Red K4 requires all Red.
        # If all Blue, Red K4 is impossible (0 red triples). Blue K5 is possible (10 blue triples).
        # So we need to break Blue K5s by turning some triples Red.
        # Conversely, if we start all Red, we have Red K4s.
        
        # Let's try a balanced random start, then repair.
        # Actually, a structured start might be better.
        # Let's try: Color triple (a,b,c) Red if (a+b+c) % 2 == 0 else Blue?
        # Or just random.
        
        # We will use a "Conflict-Driven" repair.
        # State: colors array.
        # Metrics: red_count_4[i] (0..4), blue_count_5[j] (0..10).
        # Violation: red_count_4[i] == 4 OR blue_count_5[j] == 10.
        
        # We want to minimize the number of violations.
        
        # Initialize
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        red_count_4 = [0] * num_4sets
        blue_count_5 = [0] * num_5sets
        
        # Init counts
        for t_idx, t in enumerate(triples):
            is_red = (colors[t_idx] == 0)
            for s4_idx in triple_to_4sets[t_idx]:
                if is_red:
                    red_count_4[s4_idx] += 1
            for s5_idx in triple_to_5sets[t_idx]:
                if not is_red:
                    blue_count_5[s5_idx] += 1
                    
        def get_violations():
            v4 = [i for i, c in enumerate(red_count_4) if c == 4]
            v5 = [i for i, c in enumerate(blue_count_5) if c == 10]
            return v4, v5
            
        v4, v5 = get_violations()
        if not v4 and not v5:
            return colors
            
        best_colors = colors.copy()
        best_v_count = len(v4) + len(v5)
        
        # Greedy Repair Loop
        # We pick a random violated set (4-set or 5-set).
        # We find the best triple to flip to reduce violations.
        
        while time.time() < time_limit:
            if not v4 and not v5:
                return colors
                
            # Pick a target violation
            # If we have both, pick one randomly or the one with more "potential"?
            # Let's just pick a random violated index from the union of indices?
            # Indices are different spaces (4-sets vs 5-sets).
            
            all_violated = []
            for i in v4:
                all_violated.append(('4', i))
            for i in v5:
                all_violated.append(('5', i))
                
            if not all_violated:
                return colors
                
            target_type, target_idx = random.choice(all_violated)
            
            # Determine which triples are in this target set
            # If target is 4-set i: triples are the 4 triples of that 4-set.
            # If target is 5-set i: triples are the 10 triples of that 5-set.
            
            if target_type == '4':
                # Get the 4 triples of four_sets[target_idx]
                s4 = four_sets[target_idx]
                involved_triples = [triple_index[t] for t in combinations(s4, 3)]
                # To fix a Red K4 (all 4 red), we MUST flip one of these 4 triples to Blue.
                # Flipping one to Blue reduces red_count_4 of this set to 3 (fixes it).
                # It might create new violations elsewhere.
                
                best_delta = 0
                best_flip = -1
                best_new_v4_count = len(v4)
                best_new_v5_count = len(v5)
                
                for t_idx in involved_triples:
                    # Simulate flip t_idx: Red -> Blue
                    # Update counts
                    # red_count_4 decreases for all 4-sets containing t_idx
                    # blue_count_5 increases for all 5-sets containing t_idx
                    
                    # We need to calculate the change in number of violations.
                    # Current violations:
                    # - target_idx is a violation (red_count_4 == 4). Flipping reduces to 3. So -1 violation.
                    # - Other 4-sets containing t_idx:
                    #   - If they were violations (count==4), they become count==3. Fix. (-1)
                    #   - If they were count==3, they become count==2. No change.
                    #   - If they were count<3, no change.
                    #   - Can they become violations? No, count decreases.
                    # - Other 5-sets containing t_idx:
                    #   - If they were violations (count==10), they become count==11? No, max is 10.
                    #     Wait, if a 5-set has 10 blue triples, and we flip one to Red, it becomes 9. Fix. (-1)
                    #   - If they were count==9, they become count==10. New violation! (+1)
                    #   - If count<9, no change.
                    
                    delta_v = -1 # The target 4-set is fixed
                    
                    # Check other 4-sets containing t_idx
                    for s4_idx in triple_to_4sets[t_idx]:
                        if s4_idx == target_idx:
                            continue
                        c = red_count_4[s4_idx]
                        if c == 4:
                            delta_v -= 1 # Fixed another violation
                        # c == 3 -> 2, no change
                        # c < 3 -> no change
                        
                    # Check 5-sets containing t_idx
                    for s5_idx in triple_to_5sets[t_idx]:
                        c = blue_count_5[s5_idx]
                        if c == 10:
                            delta_v -= 1 # Fixed a blue violation
                        elif c == 9:
                            delta_v += 1 # Created a new blue violation
                        
                    if delta_v < best_delta:
                        best_delta = delta_v
                        best_flip = t_idx
                        # We don't strictly need to track new counts for all, just the delta.
                        # But we need to apply the flip if accepted.
                        
                if best_flip != -1:
                    # Apply flip
                    t_idx = best_flip
                    colors[t_idx] = 1 # Blue
                    
                    for s4_idx in triple_to_4sets[t_idx]:
                        red_count_4[s4_idx] -= 1
                    for s5_idx in triple_to_5sets[t_idx]:
                        blue_count_5[s5_idx] += 1
                        
                    v4, v5 = get_violations()
                    cur_v_count = len(v4) + len(v5)
                    if cur_v_count < best_v_count:
                        best_v_count = cur_v_count
                        best_colors = colors.copy()
                        
            else: # target_type == '5'
                s5 = five_sets[target_idx]
                involved_triples = [triple_index[t] for t in combinations(s5, 3)]
                # To fix a Blue K5 (all 10 blue), we MUST flip one of these 10 triples to Red.
                
                best_delta = 0
                best_flip = -1
                
                for t_idx in involved_triples:
                    # Simulate flip t_idx: Blue -> Red
                    # red_count_4 increases for all 4-sets containing t_idx
                    # blue_count_5 decreases for all 5-sets containing t_idx
                    
                    delta_v = -1 # The target 5-set is fixed (10 -> 9)
                    
                    # Check other 5-sets containing t_idx
                    for s5_idx in triple_to_5sets[t_idx]:
                        if s5_idx == target_idx:
                            continue
                        c = blue_count_5[s5_idx]
                        if c == 10:
                            delta_v -= 1 # Fixed another violation
                        # c == 9 -> 8, no change
                        # c < 9 -> no change
                        
                    # Check 4-sets containing t_idx
                    for s4_idx in triple_to_4sets[t_idx]:
                        c = red_count_4[s4_idx]
                        if c == 4:
                            # Can't happen? If c==4, it's already a violation.
                            # Flipping Blue->Red increases c to 5? No, max red count in 4-set is 4.
                            # Wait. If a 4-set has 4 red triples, it is a violation.
                            # If we flip a triple in it from Blue to Red...
                            # But if it already has 4 red triples, the triple we are flipping MUST be Red?
                            # No, we are flipping a specific t_idx.
                            # If t_idx is Blue, and s4 contains t_idx, then s4 has at most 3 Red triples (since t_idx is not Red).
                            # So c (red_count_4) is at most 3.
                            # If c == 3, it becomes 4. New violation!
                            if c == 3:
                                delta_v += 1 # Created a new red violation
                            
                    if delta_v < best_delta:
                        best_delta = delta_v
                        best_flip = t_idx
                        
                if best_flip != -1:
                    t_idx = best_flip
                    colors[t_idx] = 0 # Red
                    
                    for s4_idx in triple_to_4sets[t_idx]:
                        red_count_4[s4_idx] += 1
                    for s5_idx in triple_to_5sets[t_idx]:
                        blue_count_5[s5_idx] -= 1
                        
                    v4, v5 = get_violations()
                    cur_v_count = len(v4) + len(v5)
                    if cur_v_count < best_v_count:
                        best_v_count = cur_v_count
                        best_colors = colors.copy()
                        
        return best_colors if best_v_count == 0 else None

    # Try n from 34 downwards
    # We will allocate time based on n. Larger n is harder.
    # Let's give 15s for 34, 10s for 33, etc.
    # Total budget ~ 115s.
    
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    
    # Time allocation:
    # 34: 20s
    # 33: 15s
    # 32: 10s
    # 31: 10s
    # 30: 10s
    # 29: 10s
    # 28: 10s
    # 27: 10s
    # 26: 5s
    # 25: 5s
    # 24: 5s
    # 23: 5s
    # 22: 5s
    # 21: 5s
    # 20: 5s
    # 19: 2s
    # 18: 2s
    # 17: 2s
    # 16: 2s
    # 15: 2s
    # 14: 2s
    # 13: 1s
    # 12: 1s
    # 11: 1s
    # 10: 1s
    # 9: 1s
    # 8: 1s
    # 7: 1s
    # 6: 1s
    # 5: 1s
    # 4: 1s
    # Sum = 20+15+10*5+5*6+2*6+1*7 = 20+15+50+30+12+7 = 134. Too much.
    
    # Let's simplify:
    # 34: 20s
    # 33: 15s
    # 32: 10s
    # 31: 10s
    # 30: 10s
    # 29: 10s
    # 28: 10s
    # 27: 10s
    # 26: 5s
    # 25: 5s
    # 24: 5s
    # 23: 5s
    # 22: 5s
    # 21: 5s
    # 20: 5s
    # 19: 2s
    # 18: 2s
    # 17: 2s
    # 16: 2s
    # 15: 2s
    # 14: 2s
    # 13: 1s
    # 12: 1s
    # 11: 1s
    # 10: 1s
    # 9: 1s
    # 8: 1s
    # 7: 1s
    # 6: 1s
    # 5: 1s
    # 4: 1s
    
    # Let's just do a loop and check time.
    
    for n in range(34, 4, -1):
        if time.time() > deadline:
            break
            
        # Dynamic time limit
        # If n >= 30, give 10s.
        # If n < 30, give 5s.
        # If n < 20, give 2s.
        if n >= 30:
            t_limit = 10.0
        elif n >= 20:
            t_limit = 5.0
        else:
            t_limit = 2.0
            
        time_limit = time.time() + t_limit
        if time_limit > deadline:
            time_limit = deadline
            
        res = solve(n, time_limit)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}
