import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
        
    # Try to find a valid coloring for a given n
    def find_coloring(n, time_limit=5.0):
        # Precompute structures
        vertices = list(range(n))
        triples = list(combinations(vertices, 3))
        num_triples = len(triples)
        
        if num_triples == 0:
            return np.array([], dtype=np.int8)
            
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute 4-sets and their constituent triples
        four_sets = list(combinations(vertices, 4))
        num_4sets = len(four_sets)
        # For each 4-set, store the 4 triple indices
        # 4-set (a,b,c,d) -> (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        set4_triples = []
        for s in four_sets:
            a,b,c,d = s
            set4_triples.append([
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(b,c,d)]
            ])
        set4_triples = np.array(set4_triples, dtype=np.int32)
        
        # Precompute 5-sets and their constituent triples
        five_sets = list(combinations(vertices, 5))
        num_5sets = len(five_sets)
        set5_triples = []
        for s in five_sets:
            a,b,c,d,e = s
            set5_triples.append([
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,b,e)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(a,c,e)],
                triple_to_idx[(a,d,e)],
                triple_to_idx[(b,c,d)],
                triple_to_idx[(b,c,e)],
                triple_to_idx[(b,d,e)],
                triple_to_idx[(c,d,e)]
            ])
        set5_triples = np.array(set5_triples, dtype=np.int32)
        
        # Inverse mapping: edge -> list of 4-set indices, edge -> list of 5-set indices
        # This is crucial for fast updates
        edge_to_4sets = [[] for _ in range(num_triples)]
        for i, tris in enumerate(set4_triples):
            for t_idx in tris:
                edge_to_4sets[t_idx].append(i)
                
        edge_to_5sets = [[] for _ in range(num_triples)]
        for i, tris in enumerate(set5_triples):
            for t_idx in tris:
                edge_to_5sets[t_idx].append(i)
        
        # Initialize random coloring
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Initialize counts
        # red_count_4[i] = number of red edges in 4-set i
        # blue_count_5[i] = number of blue edges in 5-set i
        red_count_4 = np.zeros(num_4sets, dtype=np.int16)
        blue_count_5 = np.zeros(num_5sets, dtype=np.int16)
        
        # Calculate initial counts
        # For each 4-set, count reds
        # Vectorized approach might be tricky for sparse updates, but we can do it once
        for i in range(num_4sets):
            tris = set4_triples[i]
            red_count_4[i] = np.sum(colors[tris] == 0)
            
        for i in range(num_5sets):
            tris = set5_triples[i]
            blue_count_5[i] = np.sum(colors[tris] == 1)
            
        # Count violations
        # Violation in 4-set: red_count_4[i] == 4
        # Violation in 5-set: blue_count_5[i] == 10
        total_violations = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
        
        if total_violations == 0:
            return colors
            
        best_violations = total_violations
        best_colors = colors.copy()
        
        # Simulated Annealing / Local Search
        # We will flip edges and update counts incrementally
        
        temp = 1.0
        cooling = 0.9999
        step = 0
        max_steps = 500000
        
        # Precompute lists for faster iteration
        # Convert edge_to_4sets and edge_to_5sets to numpy arrays for faster access?
        # They are lists of lists. Access is okay.
        
        while time_left() > time_limit and step < max_steps:
            # Pick a random edge to flip
            # Weighted pick: pick edges that are in many violations?
            # For simplicity, uniform random.
            e_idx = np.random.randint(0, num_triples)
            
            old_color = colors[e_idx]
            new_color = 1 - old_color
            colors[e_idx] = new_color
            
            delta_v = 0
            
            # Update 4-sets
            for s_idx in edge_to_4sets[e_idx]:
                old_rc = red_count_4[s_idx]
                # If we flipped from Red (0) to Blue (1), red count decreases by 1
                # If we flipped from Blue (1) to Red (0), red count increases by 1
                if old_color == 0:
                    new_rc = old_rc - 1
                else:
                    new_rc = old_rc + 1
                    
                # Check violation change
                # Violation if count == 4
                was_violation = (old_rc == 4)
                is_violation = (new_rc == 4)
                
                if was_violation and not is_violation:
                    delta_v -= 1
                elif not was_violation and is_violation:
                    delta_v += 1
                    
                red_count_4[s_idx] = new_rc
                
            # Update 5-sets
            for s_idx in edge_to_5sets[e_idx]:
                old_bc = blue_count_5[s_idx]
                if old_color == 1: # Was Blue, now Red -> Blue count decreases
                    new_bc = old_bc - 1
                else: # Was Red, now Blue -> Blue count increases
                    new_bc = old_bc + 1
                    
                was_violation = (old_bc == 10)
                is_violation = (new_bc == 10)
                
                if was_violation and not is_violation:
                    delta_v -= 1
                elif not was_violation and is_violation:
                    delta_v += 1
                    
                blue_count_5[s_idx] = new_bc
                
            total_violations += delta_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                if best_violations == 0:
                    return best_colors
            elif delta_v > 0:
                # Accept with probability
                if temp > 0.01 and np.random.random() < np.exp(-delta_v / temp):
                    pass # Accept
                else:
                    # Reject: revert
                    colors[e_idx] = old_color
                    total_violations -= delta_v
                    
                    # Revert counts
                    for s_idx in edge_to_4sets[e_idx]:
                        if old_color == 0:
                            red_count_4[s_idx] += 1
                        else:
                            red_count_4[s_idx] -= 1
                            
                    for s_idx in edge_to_5sets[e_idx]:
                        if old_color == 1:
                            blue_count_5[s_idx] += 1
                        else:
                            blue_count_5[s_idx] -= 1
            
            temp *= cooling
            step += 1
            
        if best_violations == 0:
            return best_colors
        return None
        
    # Try n from 34 down to 21
    # We have ~110s.
    # Allocate time based on n.
    # n=34 is hardest. Give it more time? Or equal time?
    # Let's try 34, 33, 32...
    
    # To maximize score, we want the highest n.
    # If we fail on 34, we try 33.
    
    # However, if we don't find 34, we should return the best we found.
    # The loop below returns immediately on success.
    # If we exhaust time, we return the last valid n found?
    # No, we must return a valid coloring.
    # If we don't find a valid coloring for any n >= 21, we fall back to a known small n.
    
    # Let's reserve 10s for fallback.
    # Time per n: (110 - 10) / 14 ~ 7s per n.
    
    best_n = 0
    best_colors = None
    
    for n in range(34, 20, -1):
        if time_left() < 15.0:
            break
            
        # Give more time to larger n?
        # Actually, if we find a solution for n, we stop.
        # So we just try.
        
        # Use a tighter time limit for the search itself
        search_limit = min(5.0, max(1.0, time_left() - 15.0))
        
        colors = find_coloring(n, search_limit)
        
        if colors is not None:
            best_n = n
            best_colors = colors
            break # Found a valid one, return it immediately to save time for potential higher n? 
            # Wait, we are going downwards. So if we find n=34, we are done.
            # If we fail n=34, we try n=33.
            
    if best_n == 0:
        # Fallback: return a small valid coloring
        # n=9 is known to be easy.
        n = 9
        triples = list(combinations(range(n), 3))
        # Simple algebraic coloring that works for small n
        # Just random and check? Or a known pattern.
        # For n=9, we can just use a random search which is very fast.
        colors = find_coloring(n, 1.0)
        if colors is None:
            # Hardcode a valid one?
            # Or just return n=1?
            # n=1 is trivially valid.
            n = 1
            colors = np.array([], dtype=np.int8)
        best_n = n
        best_colors = colors
        
    return {'n': best_n, 'colors': best_colors.tolist()}
