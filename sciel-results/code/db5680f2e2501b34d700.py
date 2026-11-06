import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    n = 34
    
    # Precompute indices for triples, 4-sets, and 5-sets
    triple_to_idx = {}
    triples = []
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triple_to_idx[(a,b,c)] = idx
                triples.append((a,b,c))
                idx += 1
    num_triples = idx
    
    four_sets = []
    four_set_to_triples = []
    for quad in combinations(range(n), 4):
        q_triples = []
        for i in range(4):
            for j in range(i+1, 4):
                for k in range(j+1, 4):
                    t = (quad[i], quad[j], quad[k])
                    q_triples.append(triple_to_idx[t])
        four_sets.append(quad)
        four_set_to_triples.append(q_triples)
    num_four_sets = len(four_sets)
    
    five_sets = []
    five_set_to_triples = []
    for quint in combinations(range(n), 5):
        q_triples = []
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t = (quint[i], quint[j], quint[k])
                    q_triples.append(triple_to_idx[t])
        five_sets.append(quint)
        five_set_to_triples.append(q_triples)
    num_five_sets = len(five_sets)
    
    triple_to_fours = [[] for _ in range(num_triples)]
    triple_to_fives = [[] for _ in range(num_triples)]
    
    for i, q_triples in enumerate(four_set_to_triples):
        for t_idx in q_triples:
            triple_to_fours[t_idx].append(i)
            
    for i, q_triples in enumerate(five_set_to_triples):
        for t_idx in q_triples:
            triple_to_fives[t_idx].append(i)

    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    red_counts = [0] * num_four_sets
    blue_counts = [0] * num_five_sets
    
    for i, q_triples in enumerate(four_set_to_triples):
        for t_idx in q_triples:
            if colors[t_idx] == 0:
                red_counts[i] += 1
                
    for i, q_triples in enumerate(five_set_to_triples):
        for t_idx in q_triples:
            if colors[t_idx] == 1:
                blue_counts[i] += 1
                
    red_violations = sum(1 for c in red_counts if c == 4)
    blue_violations = sum(1 for c in blue_counts if c == 10)
    total_violations = red_violations + blue_violations
    
    best_colors = None
    if total_violations == 0:
        best_colors = colors[:]
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    
    while time.time() < deadline:
        if T < min_T:
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            red_counts = [0] * num_four_sets
            blue_counts = [0] * num_five_sets
            for i, q_triples in enumerate(four_set_to_triples):
                for t_idx in q_triples:
                    if colors[t_idx] == 0:
                        red_counts[i] += 1
            for i, q_triples in enumerate(five_set_to_triples):
                for t_idx in q_triples:
                    if colors[t_idx] == 1:
                        blue_counts[i] += 1
            red_violations = sum(1 for c in red_counts if c == 4)
            blue_violations = sum(1 for c in blue_counts if c == 10)
            total_violations = red_violations + blue_violations
            T = 1.0
            if total_violations == 0:
                best_colors = colors[:]
                break
        
        t_idx = random.randint(0, num_triples - 1)
        old_color = colors[t_idx]
        new_color = 1 - old_color
        
        delta_red = 0
        delta_blue = 0
        
        for f_idx in triple_to_fours[t_idx]:
            old_rc = red_counts[f_idx]
            if new_color == 0:
                if old_rc == 3: delta_red += 1
                if old_rc == 4: delta_red -= 1
            else:
                if old_rc == 4: delta_red -= 1
                if old_rc == 5: delta_red += 1
                
        for f_idx in triple_to_fives[t_idx]:
            old_bc = blue_counts[f_idx]
            if new_color == 1:
                if old_bc == 9: delta_blue += 1
                if old_bc == 10: delta_blue -= 1
            else:
                if old_bc == 10: delta_blue -= 1
                if old_bc == 11: delta_blue += 1

        delta_total = delta_red + delta_blue
        
        if delta_total < 0 or random.random() < math.exp(-delta_total / T):
            colors[t_idx] = new_color
            for f_idx in triple_to_fours[t_idx]:
                red_counts[f_idx] += (1 if new_color == 0 else -1)
            for f_idx in triple_to_fives[t_idx]:
                blue_counts[f_idx] += (1 if new_color == 1 else -1)
            total_violations += delta_total
            if total_violations == 0:
                best_colors = colors[:]
                break
        T *= cooling
        
    if best_colors:
        return {'n': n, 'colors': best_colors}
    else:
        # Fallback: return invalid coloring?
        # To maximize chance of non-zero score, we might want to try smaller n if we have time,
        # but we are at deadline.
        # Let's just return the current state.
        return {'n': n, 'colors': colors}
