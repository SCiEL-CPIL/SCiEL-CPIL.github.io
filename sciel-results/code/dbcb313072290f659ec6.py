"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin? No, 150s limit. 110s is safe.
    
    def time_left():
        return deadline - time.time()

    def solve_n(n):
        if n < 4:
            return None
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Map each K4 to its 4 triple indices
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        # Map each K5 to its 10 triple indices
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        # Map triple index to list of K4 indices and K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        # Initialize colors: 0=Red, 1=Blue
        # Try a structured init first, then random if fails
        # Let's use random for robustness, or mod 2/3
        # colors[t] = 0 if sum % 3 == 0 else 1
        colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
        
        # Initialize counts
        red_count_k4 = [0] * len(k4s)
        blue_count_k5 = [0] * len(k5s)
        
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
            
        red_violations = sum(1 for c in red_count_k4 if c == 4)
        blue_violations = sum(1 for c in blue_count_k5 if c == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        max_steps = 200000 # High cap, but time limited
        
        # Precompute list of triples for fast random selection? 
        # We need to pick a triple involved in a violation.
        # We can maintain lists of violating K4s and K5s.
        
        # For efficiency, we'll just pick a random triple from the whole set 
        # or bias towards violations.
        # Picking a random triple from a violation is better.
        
        # Let's maintain a list of violating K4 indices and K5 indices
        # But updating these lists is expensive.
        # Instead, we can just pick a random triple. If it's not in a violation, 
        # the flip might not help or might hurt.
        # Better: Pick a random K4 that is violating, pick a random triple in it.
        # Or Pick a random K5 that is violating, pick a random triple in it.
        
        # To avoid O(N) scan for violations, we can just pick a random triple 
        # and hope. But for high n, most triples are in non-violations.
        # Let's try: 50% chance pick from red violations, 50% from blue.
        # If no red violations, pick from blue.
        
        # We need a way to pick a violating K4 quickly.
        # We can keep a list of all K4s. If we pick one and it's not violating, we wasted time.
        # Let's just pick a random triple index.
        # This is simple and effective enough with SA.
        
        for step in range(max_steps):
            if time_left() < 5.0:
                break
                
            # Pick a random triple
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
            
            if delta_E < 0:
                # Accept
                colors[t_idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
            else:
                # Accept with probability
                if T > min_T:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        colors[t_idx] = new_color
                        red_violations += delta_red
                        blue_violations += delta_blue
                    else:
                        # Rollback
                        # We already updated the counts, so we need to revert
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
                else:
                    # Rollback
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
            
            if red_violations == 0 and blue_violations == 0:
                return colors
                
            T *= cooling
            
        return None

    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try increasing n
    for n in range(10, 35):
        if time_left() < 15.0:
            break
            
        # Give each n a fair chance, but larger n needs more time.
        # We rely on the internal time check in solve_n.
        # However, if n=10 fails quickly, we move to 11.
        # If n=20 takes 50s, we might not reach 25.
        # This is a tradeoff.
        
        result = solve_n(n)
        if result is not None:
            best_n = n
            best_colors = result
            # If we found a valid coloring, we can try the next one.
            # The time left will naturally limit further attempts.
        else:
            # If it fails, we might still try the next one if time permits,
            # but it's unlikely to succeed if a smaller one failed?
            # Actually, larger n is harder. If n fails, n+1 is harder.
            # But sometimes a specific n is "easy" due to structure.
            # Let's break if we fail a few in a row?
            # Or just let time run out.
            pass

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
