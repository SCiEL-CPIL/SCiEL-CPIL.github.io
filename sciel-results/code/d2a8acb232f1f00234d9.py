import random
import time
from itertools import combinations

def construct():
    # Fallback solution
    def get_fallback():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

    best_result = get_fallback()
    deadline = time.time() + 145  # 145s limit to be safe
    
    # Try to find larger n
    # We try from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10:  # Need at least 10s for next iterations or safety
            break
            
        # Time limit for this specific n
        # Allocate more time to larger n? Or fixed?
        # Let's give 10s to each.
        t_limit = min(10.0, deadline - time.time() - 1)
        
        colors = local_search(n, t_limit)
        if colors is not None:
            best_result = {'n': n, 'colors': colors}
            # Since we are iterating downwards, the first success is the max n
            break
            
    return best_result

def local_search(n, time_limit):
    start_time = time.time()
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    quads = list(combinations(range(n), 4))
    num_quads = len(quads)
    quads_idx = {q: i for i, q in enumerate(quads)}
    
    pents = list(combinations(range(n), 5))
    num_pents = len(pents)
    pents_idx = {p: i for i, p in enumerate(pents)}
    
    # Precompute adjacency
    triple_to_quads = [[] for _ in range(num_triples)]
    triple_to_pents = [[] for _ in range(num_triples)]
    
    for q in quads:
        q_idx = quads_idx[q]
        for t in combinations(q, 3):
            t_idx = triple_to_idx[t]
            triple_to_quads[t_idx].append(q_idx)
            
    for p in pents:
        p_idx = pents_idx[p]
        for t in combinations(p, 3):
            t_idx = triple_to_idx[t]
            triple_to_pents[t_idx].append(p_idx)
            
    # Initialize colors randomly
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Initialize counts
    red_count = [0] * num_quads
    blue_count = [0] * num_pents
    
    for i, t in enumerate(triples):
        c = colors[i]
        for q_idx in triple_to_quads[i]:
            if c == 0:
                red_count[q_idx] += 1
        for p_idx in triple_to_pents[i]:
            if c == 1:
                blue_count[p_idx] += 1
                
    # Initialize violation sets
    red_violations = set(q for q in range(num_quads) if red_count[q] == 4)
    blue_violations = set(p for p in range(num_pents) if blue_count[p] == 10)
    
    if not red_violations and not blue_violations:
        return colors
        
    # Local search
    while time.time() - start_time < time_limit:
        if not red_violations and not blue_violations:
            return colors
            
        # Pick a violation
        # Prefer red if both exist? Or random?
        # Random is fine.
        if red_violations and (not blue_violations or random.random() < 0.5):
            q_idx = random.choice(list(red_violations))
            q = quads[q_idx]
            t = random.choice(list(combinations(q, 3)))
            t_idx = triple_to_idx[t]
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            # Update counts
            # Update quads
            for q_idx_2 in triple_to_quads[t_idx]:
                old_val = red_count[q_idx_2]
                if old_color == 0:
                    new_val = old_val - 1
                else:
                    new_val = old_val + 1
                red_count[q_idx_2] = new_val
                if old_val == 4:
                    red_violations.discard(q_idx_2)
                if new_val == 4:
                    red_violations.add(q_idx_2)
                    
            # Update pents
            for p_idx_2 in triple_to_pents[t_idx]:
                old_val = blue_count[p_idx_2]
                if old_color == 1:
                    new_val = old_val - 1
                else:
                    new_val = old_val + 1
                blue_count[p_idx_2] = new_val
                if old_val == 10:
                    blue_violations.discard(p_idx_2)
                if new_val == 10:
                    blue_violations.add(p_idx_2)
                    
        else:
            # Must be blue violation
            if not blue_violations:
                # This case shouldn't happen if we checked at start, 
                # but if red_violations was empty and we picked else branch?
                # If red_violations empty, condition `red_violations and ...` is False.
                # So we go to else.
                # If blue_violations is also empty, we returned at start.
                # So blue_violations must be non-empty.
                pass
                
            p_idx = random.choice(list(blue_violations))
            p = pents[p_idx]
            t = random.choice(list(combinations(p, 3)))
            t_idx = triple_to_idx[t]
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            # Update quads
            for q_idx_2 in triple_to_quads[t_idx]:
                old_val = red_count[q_idx_2]
                if old_color == 0:
                    new_val = old_val - 1
                else:
                    new_val = old_val + 1
                red_count[q_idx_2] = new_val
                if old_val == 4:
                    red_violations.discard(q_idx_2)
                if new_val == 4:
                    red_violations.add(q_idx_2)
                    
            # Update pents
            for p_idx_2 in triple_to_pents[t_idx]:
                old_val = blue_count[p_idx_2]
                if old_color == 1:
                    new_val = old_val - 1
                else:
                    new_val = old_val + 1
                blue_count[p_idx_2] = new_val
                if old_val == 10:
                    blue_violations.discard(p_idx_2)
                if new_val == 10:
                    blue_violations.add(p_idx_2)

    return None
