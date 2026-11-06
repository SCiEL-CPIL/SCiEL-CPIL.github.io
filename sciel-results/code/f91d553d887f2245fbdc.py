import random
import time

def construct():
    # We aim for n=34, but will fallback if not found quickly
    # The known bound is >34, so 34 is a target.
    # We will use Simulated Annealing / Local Search.
    
    def solve(n, max_time=110):
        if n < 1:
            return None
        
        # Generate all triples
        triples = []
        triple_idx = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx = len(triples)
                    triples.append((a, b, c))
                    triple_idx[(a, b, c)] = idx
        
        m = len(triples)
        
        # Pre-compute for each triple, the other triples in the K4s and K5s it forms
        # A K4 is defined by 4 vertices. It has 4 triples.
        # A K5 is defined by 5 vertices. It has 10 triples.
        
        # For each triple i, list of (other_triple_indices, clique_type)
        # Actually, it's easier to iterate over all K4s and K5s and map them to their constituent triple indices.
        
        k4s = [] # list of lists of triple indices
        k5s = [] # list of lists of triple indices
        
        # Generate K4s
        from itertools import combinations
        for verts in combinations(range(n), 4):
            k4_triples = []
            for t in combinations(verts, 3):
                k4_triples.append(triple_idx[t])
            k4s.append(k4_triples)
            
        # Generate K5s
        for verts in combinations(range(n), 5):
            k5_triples = []
            for t in combinations(verts, 3):
                k5_triples.append(triple_idx[t])
            k5s.append(k5_triples)
            
        # Map each triple to the K4s and K5s it belongs to
        # triple_to_k4s[i] = list of k4 indices
        # triple_to_k5s[i] = list of k5 indices
        triple_to_k4s = [[] for _ in range(m)]
        triple_to_k5s = [[] for _ in range(m)]
        
        for k4_idx, t_list in enumerate(k4s):
            for t_idx in t_list:
                triple_to_k4s[t_idx].append(k4_idx)
                
        for k5_idx, t_list in enumerate(k5s):
            for t_idx in t_list:
                triple_to_k5s[t_idx].append(k5_idx)

        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(m)]
        
        # Calculate initial cost
        # cost = num_red_k4 + num_blue_k5
        def calc_cost(colors):
            cost = 0
            for t_list in k4s:
                # check if all red (0)
                if all(colors[t] == 0 for t in t_list):
                    cost += 1
            for t_list in k5s:
                # check if all blue (1)
                if all(colors[t] == 1 for t in t_list):
                    cost += 1
            return cost
            
        current_cost = calc_cost(colors)
        
        if current_cost == 0:
            return colors
            
        start_time = time.time()
        temp = 1.0
        cooling = 0.999
        
        while time.time() - start_time < max_time:
            # Pick random triple
            t_idx = random.randint(0, m - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_cost = 0
            
            # Check K4s containing t_idx
            for k4_idx in triple_to_k4s[t_idx]:
                t_list = k4s[k4_idx]
                # Count how many are 0 and 1
                # If old_color was 0, we are removing a 0.
                # If old_color was 1, we are removing a 1.
                
                # Efficient check:
                # Before flip:
                # If all were 0, cost was 1. Now one is 1, so cost 0. Delta -1.
                # If all were 1, cost 0. Now one is 0, cost 0. Delta 0.
                # If mixed, cost 0.
                
                # Let's just recount the 4 elements. It's small.
                # Or logic:
                # Let c0 = count of 0s in t_list before flip
                # Let c1 = count of 1s in t_list before flip
                
                # We can maintain counts? Or just check the other 3.
                others = [t for t in t_list if t != t_idx]
                
                # Before:
                if old_color == 0:
                    # If others are all 0, it was a red K4. Now it's not. Delta -1.
                    if all(colors[o] == 0 for o in others):
                        delta_cost -= 1
                    # If others are all 1, it wasn't a red K4 (mixed). Now mixed. Delta 0.
                    # If others mixed, Delta 0.
                else: # old_color == 1
                    # If others are all 1, it wasn't a red K4. Now mixed. Delta 0.
                    # Wait, we are looking for Red K4 (all 0).
                    # If old_color was 1, it couldn't have been part of a Red K4.
                    # So no red K4 was lost.
                    # Can it form a new Red K4? No, because new_color is 0.
                    # If others are all 0, then after flip, all are 0. So it BECOMES a Red K4. Delta +1.
                    if all(colors[o] == 0 for o in others):
                        delta_cost += 1
            
            # Check K5s containing t_idx
            for k5_idx in triple_to_k5s[t_idx]:
                t_list = k5s[k5_idx]
                others = [t for t in t_list if t != t_idx]
                
                # We are looking for Blue K5 (all 1).
                if old_color == 1:
                    # If others are all 1, it was a Blue K5. Now it's not. Delta -1.
                    if all(colors[o] == 1 for o in others):
                        delta_cost -= 1
                else: # old_color == 0
                    # If others are all 1, it wasn't a Blue K5. Now new_color is 1, so it BECOMES a Blue K5. Delta +1.
                    if all(colors[o] == 1 for o in others):
                        delta_cost += 1
            
            # Acceptance
            if delta_cost <= 0:
                colors[t_idx] = new_color
                current_cost += delta_cost
                if current_cost == 0:
                    return colors
            else:
                # Probabilistic acceptance
                # Simple: accept with prob exp(-delta/temp)
                import math
                prob = math.exp(-delta_cost / temp)
                if random.random() < prob:
                    colors[t_idx] = new_color
                    current_cost += delta_cost
                    if current_cost == 0:
                        return colors
            
            temp *= cooling
            
        return None

    # Try to find a solution for n=34
    # To be safe, we might try a few random restarts or just one long run.
    # Given the 120s limit, let's use 110s for the search.
    
    # If n=34 is too hard, we can try smaller n.
    # But the score is n/9.
    # Let's try n=34.
    result = solve(34, max_time=115)
    
    if result is not None:
        return {'n': 34, 'colors': result}
    
    # Fallback: try n=30
    result = solve(30, max_time=10)
    if result is not None:
        return {'n': 30, 'colors': result}
        
    # Fallback: n=20
    result = solve(20, max_time=5)
    if result is not None:
        return {'n': 20, 'colors': result}
        
    # Fallback: n=10 (guaranteed to be fast)
    result = solve(10, max_time=1)
    if result is not None:
        return {'n': 10, 'colors': result}
        
    # Ultimate fallback: the seed
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
