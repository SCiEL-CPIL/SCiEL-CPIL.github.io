import numpy as np
import time
import random
from itertools import combinations

def construct():
    # Target n
    n = 34
    
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
        
    # Precompute 4-sets and their constituent triples
    four_sets = list(combinations(range(n), 4))
    num_four_sets = len(four_sets)
    # four_set_triples[i] is a list of 4 triple indices
    four_set_triples = []
    # triple_to_four_sets[k] is a list of 4-set indices containing triple k
    triple_to_four_sets = [[] for _ in range(num_triples)]
    
    for i, s in enumerate(four_sets):
        ts = []
        for j in range(4):
            t = (s[0], s[1], s[2]) if j == 0 else \
                (s[0], s[1], s[3]) if j == 1 else \
                (s[0], s[2], s[3]) if j == 2 else \
                (s[1], s[2], s[3])
            idx = triple_index[t]
            ts.append(idx)
            triple_to_four_sets[idx].append(i)
        four_set_triples.append(ts)
        
    # Precompute 5-sets and their constituent triples
    five_sets = list(combinations(range(n), 5))
    num_five_sets = len(five_sets)
    # five_set_triples[j] is a list of 10 triple indices
    five_set_triples = []
    # triple_to_five_sets[k] is a list of 5-set indices containing triple k
    triple_to_five_sets = [[] for _ in range(num_triples)]
    
    for j, s in enumerate(five_sets):
        ts = []
        # Generate all 10 triples from 5 vertices
        for comb in combinations(s, 3):
            idx = triple_index[comb]
            ts.append(idx)
            triple_to_five_sets[idx].append(j)
        five_set_triples.append(ts)
        
    # State
    # colors: 0 is Red, 1 is Blue
    colors = np.random.randint(0, 2, num_triples)
    
    # Counts
    # red_counts_4[i]: number of red triples in 4-set i
    red_counts_4 = np.zeros(num_four_sets, dtype=np.int8)
    # blue_counts_5[j]: number of blue triples in 5-set j
    blue_counts_5 = np.zeros(num_five_sets, dtype=np.int8)
    
    # Initialize counts
    for i in range(num_four_sets):
        for idx in four_set_triples[i]:
            if colors[idx] == 0:
                red_counts_4[i] += 1
                
    for j in range(num_five_sets):
        for idx in five_set_triples[j]:
            if colors[idx] == 1:
                blue_counts_5[j] += 1
                
    # Initial Energy
    red_violations = np.sum(red_counts_4 == 4)
    blue_violations = np.sum(blue_counts_5 == 10)
    current_energy = red_violations + blue_violations
    
    best_energy = current_energy
    best_colors = colors.copy()
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if current_energy == 0:
            break
            
        # Pick a random triple to flip
        k = np.random.randint(num_triples)
        old_color = colors[k]
        new_color = 1 - old_color
        
        delta_energy = 0
        
        # Update 4-sets
        for i in triple_to_four_sets[k]:
            old_count = red_counts_4[i]
            if old_color == 0:
                # Was red, now blue. Red count decreases.
                new_count = old_count - 1
                # Check if we lost a violation (4 -> 3) or gained one (3 -> 4 impossible here)
                if old_count == 4:
                    delta_energy -= 1
                # new_count will be 3, no new violation
            else:
                # Was blue, now red. Red count increases.
                new_count = old_count + 1
                if new_count == 4:
                    delta_energy += 1
            red_counts_4[i] = new_count
            
        # Update 5-sets
        for j in triple_to_five_sets[k]:
            old_count = blue_counts_5[j]
            if old_color == 1:
                # Was blue, now red. Blue count decreases.
                new_count = old_count - 1
                if old_count == 10:
                    delta_energy -= 1
            else:
                # Was red, now blue. Blue count increases.
                new_count = old_count + 1
                if new_count == 10:
                    delta_energy += 1
            blue_counts_5[j] = new_count
            
        current_energy += delta_energy
        
        # Acceptance
        if delta_energy <= 0:
            colors[k] = new_color
        else:
            if T > min_T:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta_energy / T):
                    colors[k] = new_color
                else:
                    # Revert
                    colors[k] = old_color
                    # Revert counts
                    for i in triple_to_four_sets[k]:
                        if old_color == 0:
                            red_counts_4[i] += 1
                        else:
                            red_counts_4[i] -= 1
                    for j in triple_to_five_sets[k]:
                        if old_color == 1:
                            blue_counts_5[j] += 1
                        else:
                            blue_counts_5[j] -= 1
                    current_energy -= delta_energy
            else:
                # Revert
                colors[k] = old_color
                for i in triple_to_four_sets[k]:
                    if old_color == 0:
                        red_counts_4[i] += 1
                    else:
                        red_counts_4[i] -= 1
                for j in triple_to_five_sets[k]:
                    if old_color == 1:
                        blue_counts_5[j] += 1
                    else:
                        blue_counts_5[j] -= 1
                current_energy -= delta_energy
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_colors = colors.copy()
            
        T *= cooling
        
    # If we found a valid coloring, return it
    if best_energy == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # If not, we might try to reduce n or just return the best effort?
    # The task says "Maximize n". If we can't find valid 34, we should try 33, 32...
    # But given time, let's just return the best found for n=34 if valid, else try smaller n quickly?
    # Actually, if best_energy > 0, the coloring is invalid.
    # We should try to find a valid coloring for a smaller n.
    # However, usually if we can't find 34, we might find 33.
    # Let's just return the seed behavior if we fail? No, that scores 1.0.
    # Better to return a valid smaller n.
    # But re-running the whole search for n=33 is expensive.
    # Let's assume we find 34. If not, we return n=9 seed? 
    # To be safe, if best_energy > 0, we can try a quick greedy for n=20?
    # Or just return the best_colors for n=34? No, it must be valid.
    # The contract says "A valid colouring on n points certifies...". Invalid scores 0.
    # So we MUST return a valid coloring.
    # If we fail to find valid 34, we must find valid smaller n.
    
    # Fallback: Try n=20 with same method quickly?
    # Or just return n=9 seed?
    # Let's add a quick check: if best_energy > 0, try n=30, 25... with fewer iterations.
    
    # For simplicity and robustness within time, if 34 fails, try 30.
    if best_energy > 0:
        # Try n=30
        n = 30
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        num_four_sets = len(four_sets)
        four_set_triples = []
        triple_to_four_sets = [[] for _ in range(num_triples)]
        for i, s in enumerate(four_sets):
            ts = []
            for j in range(4):
                t = (s[0], s[1], s[2]) if j == 0 else \
                    (s[0], s[1], s[3]) if j == 1 else \
                    (s[0], s[2], s[3]) if j == 2 else \
                    (s[1], s[2], s[3])
                idx = triple_index[t]
                ts.append(idx)
                triple_to_four_sets[idx].append(i)
            four_set_triples.append(ts)
            
        five_sets = list(combinations(range(n), 5))
        num_five_sets = len(five_sets)
        five_set_triples = []
        triple_to_five_sets = [[] for _ in range(num_triples)]
        for j, s in enumerate(five_sets):
            ts = []
            for comb in combinations(s, 3):
                idx = triple_index[comb]
                ts.append(idx)
                triple_to_five_sets[idx].append(j)
            five_set_triples.append(ts)
            
        colors = np.random.randint(0, 2, num_triples)
        red_counts_4 = np.zeros(num_four_sets, dtype=np.int8)
        blue_counts_5 = np.zeros(num_five_sets, dtype=np.int8)
        
        for i in range(num_four_sets):
            for idx in four_set_triples[i]:
                if colors[idx] == 0:
                    red_counts_4[i] += 1
        for j in range(num_five_sets):
            for idx in five_set_triples[j]:
                if colors[idx] == 1:
                    blue_counts_5[j] += 1
                    
        current_energy = np.sum(red_counts_4 == 4) + np.sum(blue_counts_5 == 10)
        best_energy = current_energy
        best_colors = colors.copy()
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        start_time = time.time()
        time_limit = 110.0 - (time.time() - start_time) # Remaining time
        
        while time.time() - start_time < time_limit:
            if current_energy == 0:
                break
            k = np.random.randint(num_triples)
            old_color = colors[k]
            new_color = 1 - old_color
            delta_energy = 0
            for i in triple_to_four_sets[k]:
                old_count = red_counts_4[i]
                if old_color == 0:
                    new_count = old_count - 1
                    if old_count == 4: delta_energy -= 1
                else:
                    new_count = old_count + 1
                    if new_count == 4: delta_energy += 1
                red_counts_4[i] = new_count
            for j in triple_to_five_sets[k]:
                old_count = blue_counts_5[j]
                if old_color == 1:
                    new_count = old_count - 1
                    if old_count == 10: delta_energy -= 1
                else:
                    new_count = old_count + 1
                    if new_count == 10: delta_energy += 1
                blue_counts_5[j] = new_count
                
            current_energy += delta_energy
            if delta_energy <= 0:
                colors[k] = new_color
            else:
                if T > min_T and random.random() < np.exp(-delta_energy / T):
                    colors[k] = new_color
                else:
                    colors[k] = old_color
                    for i in triple_to_four_sets[k]:
                        if old_color == 0: red_counts_4[i] += 1
                        else: red_counts_4[i] -= 1
                    for j in triple_to_five_sets[k]:
                        if old_color == 1: blue_counts_5[j] += 1
                        else: blue_counts_5[j] -= 1
                    current_energy -= delta_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors.copy()
            T *= cooling
            
        if best_energy == 0:
            return {'n': n, 'colors': best_colors.tolist()}
            
    # Final fallback: return the seed n=9 valid coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

