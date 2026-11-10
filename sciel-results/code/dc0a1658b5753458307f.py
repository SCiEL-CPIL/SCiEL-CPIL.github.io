import itertools
import random
import time
import math

def construct():
    # Target n
    target_n = 34
    # Time limit
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Generate triples
    triples = list(itertools.combinations(range(target_n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their triples
    # 4-sets: (a,b,c,d)
    # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
    # We need to map each triple to the 4-sets it belongs to?
    # Or iterate over 4-sets to check violations?
    # For incremental update, we need:
    # For each triple t, list of 4-sets containing t.
    # For each triple t, list of 5-sets containing t.
    
    four_sets = list(itertools.combinations(range(target_n), 4))
    five_sets = list(itertools.combinations(range(target_n), 5))
    
    # Map triple -> list of indices of 4-sets containing it
    triple_to_fours = [[] for _ in range(num_triples)]
    for idx, quad in enumerate(four_sets):
        # 4 triples in a quad
        t1 = (quad[0], quad[1], quad[2])
        t2 = (quad[0], quad[1], quad[3])
        t3 = (quad[0], quad[2], quad[3])
        t4 = (quad[1], quad[2], quad[3])
        triple_to_fours[triple_to_idx[t1]].append(idx)
        triple_to_fours[triple_to_idx[t2]].append(idx)
        triple_to_fours[triple_to_idx[t3]].append(idx)
        triple_to_fours[triple_to_idx[t4]].append(idx)
        
    # Map triple -> list of indices of 5-sets containing it
    triple_to_fives = [[] for _ in range(num_triples)]
    for idx, quint in enumerate(five_sets):
        # 10 triples in a quint
        for t in itertools.combinations(quint, 3):
            triple_to_fives[triple_to_idx[t]].append(idx)
            
    # Initialize random coloring
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Calculate initial cost
    # Cost = (# Red K4) + (# Blue K5)
    red_k4_count = 0
    blue_k5_count = 0
    
    for quad in four_sets:
        # Check if all 4 triples are red (0)
        t1 = triple_to_idx[(quad[0], quad[1], quad[2])]
        t2 = triple_to_idx[(quad[0], quad[1], quad[3])]
        t3 = triple_to_idx[(quad[0], quad[2], quad[3])]
        t4 = triple_to_idx[(quad[1], quad[2], quad[3])]
        if colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0:
            red_k4_count += 1
            
    for quint in five_sets:
        # Check if all 10 triples are blue (1)
        is_blue = True
        for t in itertools.combinations(quint, 3):
            if colors[triple_to_idx[t]] != 1:
                is_blue = False
                break
        if is_blue:
            blue_k5_count += 1
            
    current_cost = red_k4_count + blue_k5_count
    
    best_cost = current_cost
    best_colors = colors[:]
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if current_cost == 0:
            break
            
        # Pick random triple
        t_idx = random.randint(0, num_triples - 1)
        
        # Calculate delta cost
        delta_cost = 0
        
        # Check 4-sets
        for quad_idx in triple_to_fours[t_idx]:
            quad = four_sets[quad_idx]
            # Get the 4 triple indices
            t1 = triple_to_idx[(quad[0], quad[1], quad[2])]
            t2 = triple_to_idx[(quad[0], quad[1], quad[3])]
            t3 = triple_to_idx[(quad[0], quad[2], quad[3])]
            t4 = triple_to_idx[(quad[1], quad[2], quad[3])]
            
            # Count reds before
            reds_before = 0
            if colors[t1] == 0: reds_before += 1
            if colors[t2] == 0: reds_before += 1
            if colors[t3] == 0: reds_before += 1
            if colors[t4] == 0: reds_before += 1
            
            # Determine new color for t_idx
            # t_idx is one of t1..t4
            # If we flip t_idx, reds_after = reds_before - 1 if colors[t_idx]==0 else reds_before + 1
            if colors[t_idx] == 0:
                reds_after = reds_before - 1
            else:
                reds_after = reds_before + 1
                
            # Violation before: 1 if reds_before == 4 else 0
            # Violation after: 1 if reds_after == 4 else 0
            viol_before = 1 if reds_before == 4 else 0
            viol_after = 1 if reds_after == 4 else 0
            delta_cost += viol_after - viol_before
            
        # Check 5-sets
        for quint_idx in triple_to_fives[t_idx]:
            quint = five_sets[quint_idx]
            # Count blues before
            blues_before = 0
            for t in itertools.combinations(quint, 3):
                if colors[triple_to_idx[t]] == 1:
                    blues_before += 1
            
            # Determine new color for t_idx
            if colors[t_idx] == 1:
                blues_after = blues_before - 1
            else:
                blues_after = blues_before + 1
                
            # Violation before: 1 if blues_before == 10 else 0
            # Violation after: 1 if blues_after == 10 else 0
            viol_before = 1 if blues_before == 10 else 0
            viol_after = 1 if blues_after == 10 else 0
            delta_cost += viol_after - viol_before
            
        # Acceptance
        if delta_cost <= 0:
            # Accept
            colors[t_idx] = 1 - colors[t_idx]
            current_cost += delta_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
        else:
            # Probabilistic accept
            if T > min_T:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    colors[t_idx] = 1 - colors[t_idx]
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
            T *= cooling_rate
            
        # Restart if stuck?
        # Simple strategy: if T is low and no improvement for a while, restart with random
        # But for now, just let it run.
        
    # If we found a valid coloring, return it
    if best_cost == 0:
        return {'n': target_n, 'colors': best_colors}
    
    # If not, try smaller n? Or just return the best found?
    # The task says "Maximize n". If we can't find valid for 34, we should try 33.
    # However, SA for 34 is very likely to succeed.
    # If it fails, it's better to return a valid smaller n than an invalid 34.
    # Let's try 33 if 34 fails.
    
    # For brevity and speed, if 34 fails, we return the best we have?
    # No, invalid result scores 0.
    # We must return a VALID coloring.
    # If best_cost > 0, we need to find a valid one for smaller n.
    
    # Let's implement a fallback: try n=33, 32...
    # But re-initializing everything is expensive.
    # Given the high probability of success for n=34, we assume it works.
    # If not, we can do a simple greedy for smaller n.
    
    # Actually, a simple modular coloring works for small n.
    # For n=9, (a+b+c)%3 works.
    # Let's check if we can just return the best found if valid.
    # If not valid, we should degrade to a known valid small n.
    
    # To be safe, if best_cost > 0, we run a quick search for n=20 or use the seed's logic for n=9.
    # But the score is n. So we want max valid n.
    
    # Let's assume SA finds it. If not, we return n=9 with the simple formula.
    if best_cost > 0:
         # Fallback to a known valid small case
         n_fallback = 9
         colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_fallback), 3)]
         return {'n': n_fallback, 'colors': colors_fallback}
         
    return {'n': target_n, 'colors': best_colors}
