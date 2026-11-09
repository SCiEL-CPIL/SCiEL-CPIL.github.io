"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_left():
        return deadline - time.time()

    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        return {
            'triples': triples,
            'num_triples': num_triples,
            'triple_idx': triple_idx,
            'k4s': k4s,
            'k5s': k5s,
            'k4_triples': k4_triples,
            'k5_triples': k5_triples,
            'triple_to_k4': triple_to_k4,
            'triple_to_k5': triple_to_k5
        }

    def solve_n(n, time_budget):
        if n < 4:
            return None
            
        structs = get_structures(n)
        triples = structs['triples']
        num_triples = structs['num_triples']
        k4_triples = structs['k4_triples']
        k5_triples = structs['k5_triples']
        triple_to_k4 = structs['triple_to_k4']
        triple_to_k5 = structs['triple_to_k5']
        num_k4s = len(structs['k4s'])
        num_k5s = len(structs['k5s'])
        
        # Initialize colors: 0=Red, 1=Blue
        # Use a structured init that might be closer to a solution
        colors = np.zeros(num_triples, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            # Simple heuristic: color based on sum mod 3
            # 0 -> Red, 1,2 -> Blue
            if (a + b + c) % 3 == 0:
                colors[i] = 0
            else:
                colors[i] = 1
        
        # Initialize counts
        red_count_k4 = np.zeros(num_k4s, dtype=np.int8)
        blue_count_k5 = np.zeros(num_k5s, dtype=np.int8)
        
        for i, ts in enumerate(k4_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 0:
                    cnt += 1
            red_count_k4[i] = cnt
            
        for i, ts in enumerate(k5_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 1:
                    cnt += 1
            blue_count_k5[i] = cnt
            
        red_violations = np.sum(red_count_k4 == 4)
        blue_violations = np.sum(blue_count_k5 == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors.tolist()
            
        # Simulated Annealing with Constraint Propagation hints
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        # Precompute lists of violating structures for faster selection
        # We will update these lists dynamically
        violating_k4s = [i for i in range(num_k4s) if red_count_k4[i] == 4]
        violating_k5s = [i for i in range(num_k5s) if blue_count_k5[i] == 10]
        
        end_time = time.time() + time_budget
        steps = 0
        
        while time.time() < end_time:
            if red_violations == 0 and blue_violations == 0:
                return colors.tolist()
                
            # Select a triple to flip
            # Bias towards triples in violating structures
            # If no violations, pick random (should not happen if we check above)
            if red_violations > 0 and (blue_violations == 0 or random.random() < red_violations / (red_violations + blue_violations)):
                if violating_k4s:
                    k4_idx = random.choice(violating_k4s)
                    t_idx = random.choice(k4_triples[k4_idx])
                else:
                    t_idx = random.randint(0, num_triples - 1)
            else:
                if violating_k5s:
                    k5_idx = random.choice(violating_k5s)
                    t_idx = random.choice(k5_triples[k5_idx])
                else:
                    t_idx = random.randint(0, num_triples - 1)
                    
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate Delta E
            delta_red = 0
            delta_blue = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4[t_idx]:
                if old_color == 0: # Was Red, becoming Blue
                    old_cnt = red_count_k4[k4_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 4 and new_cnt == 3:
                        delta_red -= 1
                        # Remove from violating list if it was the only one? 
                        # We handle list updates lazily or with a set
                    elif old_cnt == 3 and new_cnt == 4:
                        delta_red += 1
                    red_count_k4[k4_idx] = new_cnt
                else: # Was Blue, becoming Red
                    old_cnt = red_count_k4[k4_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 3 and new_cnt == 4:
                        delta_red += 1
                    elif old_cnt == 4 and new_cnt == 3:
                        delta_red -= 1
                    red_count_k4[k4_idx] = new_cnt
                    
            # Update K5 counts
            for k5_idx in triple_to_k5[t_idx]:
                if old_color == 1: # Was Blue, becoming Red
                    old_cnt = blue_count_k5[k5_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 10 and new_cnt == 9:
                        delta_blue -= 1
                    elif old_cnt == 9 and new_cnt == 10:
                        delta_blue += 1
                    blue_count_k5[k5_idx] = new_cnt
                else: # Was Red, becoming Blue
                    old_cnt = blue_count_k5[k5_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 9 and new_cnt == 10:
                        delta_blue += 1
                    elif old_cnt == 10 and new_cnt == 9:
                        delta_blue -= 1
                    blue_count_k5[k5_idx] = new_cnt
                    
            delta_E = delta_red + delta_blue
            
            accepted = False
            if delta_E < 0:
                accepted = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        accepted = True
                # else: reject
                
            if accepted:
                colors[t_idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
                
                # Update violating lists (lazy update: just rebuild occasionally or update incrementally)
                # For simplicity and speed, we'll just filter the lists periodically or rely on the fact
                # that we pick from the list and check if it's still violating.
                # However, the lists can become stale. Let's update them if they get too large or stale.
                # A simpler approach: just pick a random triple if the lists are empty/stale.
                # But we want to bias.
                # Let's update the lists every 100 steps or if they become empty.
                if steps % 100 == 0 or len(violating_k4s) == 0 or len(violating_k5s) == 0:
                    violating_k4s = [i for i in range(num_k4s) if red_count_k4[i] == 4]
                    violating_k5s = [i for i in range(num_k5s) if blue_count_k5[i] == 10]
                    
            else:
                # Rollback counts
                for k4_idx in triple_to_k4[t_idx]:
                    if old_color == 0:
                        red_count_k4[k4_idx] += 1
                    else:
                        red_count_k4[k4_idx] -= 1
                for k5_idx in triple_to_k5[t_idx]:
                    if old_color == 1:
                        blue_count_k5[k5_idx] += 1
                    else:
                        blue_count_k5[k5_idx] -= 1
                        
            T *= cooling
            steps += 1
            
        return None

    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try to solve for increasing n
    # Start from a lower n to warm up, then go higher
    # The time budget is shared, so we allocate based on remaining time
    # We'll try n=10 to 34
    
    for target_n in range(10, 35):
        if time_left() < 5.0:
            break
        
        # Allocate time budget
        # More time for larger n? Or equal?
        # Let's give more time to larger n as they are harder
        # But we need to finish.
        # If we have 100s left and 25 n's to try, ~4s each.
        # But n=34 is much harder.
        # Let's try a few random restarts for the current target_n if we have time.
        
        # Simple strategy: try to solve target_n with a fixed time slice
        # If we find a solution, move to next n.
        # If we don't, try again with a different seed/init?
        
        # Let's just try once per n with the remaining time divided by remaining n's?
        # No, that's risky.
        
        # Let's try to solve with a budget of min(10, time_left()/2) seconds
        # If successful, continue. If not, continue to next n (maybe with a different init).
        
        budget = min(15.0, time_left() / 2.0)
        
        result = solve_n(target_n, budget)
        if result is not None:
            best_n = target_n
            best_colors = result
            # We can continue to try for larger n
        else:
            # If we fail, we might try again with a different init
            # But for now, let's just move on or retry once
            if time_left() > 10.0:
                # Retry with a different random seed
                random.seed(random.randint(0, 1000000))
                result = solve_n(target_n, min(10.0, time_left() / 2.0))
                if result is not None:
                    best_n = target_n
                    best_colors = result

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
