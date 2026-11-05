import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    def precompute_sets(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Map triple index to list of 4-set indices and 5-set indices
        # 4-sets are tuples of 4 vertices
        # 5-sets are tuples of 5 vertices
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        # For each triple, find which 4-sets and 5-sets contain it
        triple_to_4sets = [[] for _ in range(len(triples))]
        triple_to_5sets = [[] for _ in range(len(triples))]
        
        # Efficiently map 4-sets to their constituent triples
        # A 4-set {a,b,c,d} has triples (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        for i, s4 in enumerate(four_sets):
            a, b, c, d = s4
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            triple_to_4sets[triple_index[t1]].append(i)
            triple_to_4sets[triple_index[t2]].append(i)
            triple_to_4sets[triple_index[t3]].append(i)
            triple_to_4sets[triple_index[t4]].append(i)
            
        # A 5-set {a,b,c,d,e} has 10 triples
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def solve_for_n(n, time_limit):
        if n < 4:
            return [0] * 0 # Trivial
        
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute_sets(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # colors: 0 is red, 1 is blue
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # red_count_4[i] = number of red triples in 4-set i
        # blue_count_5[j] = number of blue triples in 5-set j
        red_count_4 = [0] * num_4sets
        blue_count_5 = [0] * num_5sets
        
        # Initialize counts
        for t_idx, t in enumerate(triples):
            is_red = (colors[t_idx] == 0)
            for s4_idx in triple_to_4sets[t_idx]:
                if is_red:
                    red_count_4[s4_idx] += 1
            for s5_idx in triple_to_5sets[t_idx]:
                if not is_red: # blue
                    blue_count_5[s5_idx] += 1
                    
        # Calculate initial cost
        # Cost = (# of 4-sets with red_count_4 == 4) + (# of 5-sets with blue_count_5 == 10)
        def calc_cost():
            cost = 0
            for c in red_count_4:
                if c == 4:
                    cost += 1
            for c in blue_count_5:
                if c == 10:
                    cost += 1
            return cost
            
        current_cost = calc_cost()
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors.copy()
        
        # Simulated Annealing / Local Search
        # We flip one triple at a time
        temp = 1.0
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            if temp < 0.01:
                break
                
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta cost
            # If we flip t_idx from red to blue:
            # - For each 4-set containing t_idx: red_count_4 decreases by 1
            # - For each 5-set containing t_idx: blue_count_5 increases by 1
            # If we flip t_idx from blue to red:
            # - For each 4-set containing t_idx: red_count_4 increases by 1
            # - For each 5-set containing t_idx: blue_count_5 decreases by 1
            
            delta_cost = 0
            affected_4 = triple_to_4sets[t_idx]
            affected_5 = triple_to_5sets[t_idx]
            
            if old_color == 0: # Red -> Blue
                # Red count decreases
                for s4_idx in affected_4:
                    old_c = red_count_4[s4_idx]
                    new_c = old_c - 1
                    if old_c == 4: delta_cost -= 1
                    if new_c == 4: delta_cost += 1
                    # Update counts temporarily for delta calc? No, we can just check boundaries
                # Blue count increases
                for s5_idx in affected_5:
                    old_c = blue_count_5[s5_idx]
                    new_c = old_c + 1
                    if old_c == 10: delta_cost -= 1
                    if new_c == 10: delta_cost += 1
                    
            else: # Blue -> Red
                # Red count increases
                for s4_idx in affected_4:
                    old_c = red_count_4[s4_idx]
                    new_c = old_c + 1
                    if old_c == 4: delta_cost -= 1
                    if new_c == 4: delta_cost += 1
                # Blue count decreases
                for s5_idx in affected_5:
                    old_c = blue_count_5[s5_idx]
                    new_c = old_c - 1
                    if old_c == 10: delta_cost -= 1
                    if new_c == 10: delta_cost += 1
            
            # Accept if delta < 0 or with probability exp(-delta/temp)
            accept = False
            if delta_cost < 0:
                accept = True
            elif delta_cost == 0:
                accept = True
            else:
                import math
                if temp > 0:
                    if random.random() < math.exp(-delta_cost / temp):
                        accept = True
            
            if accept:
                # Apply flip
                colors[t_idx] = new_color
                for s4_idx in affected_4:
                    if old_color == 0: # R->B
                        red_count_4[s4_idx] -= 1
                    else: # B->R
                        red_count_4[s4_idx] += 1
                for s5_idx in affected_5:
                    if old_color == 0: # R->B
                        blue_count_5[s5_idx] += 1
                    else: # B->R
                        blue_count_5[s5_idx] -= 1
                
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject, do nothing
                pass
                
            temp *= cooling_rate
            
        # If we didn't find 0, return the best we found (which might be invalid)
        # But we need to return a VALID coloring.
        # So we should only return if cost is 0.
        # If not, return None or a smaller n solution.
        if best_cost == 0:
            return best_colors
        return None

    # Try n from 34 downwards
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        # Give more time to larger n? Or equal time?
        # Larger n is harder. Let's give 10s per n.
        time_limit = time.time() + 10.0
        if time_limit > deadline:
            time_limit = deadline
            
        res = solve_for_n(n, time_limit)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback: return a small valid coloring
    # n=4: 4 triples. If all red, red K4. If 3 red 1 blue, no red K4.
    # Blue K5 requires 5 vertices. So for n<5, only red K4 matters.
    # For n=4, avoid all 4 triples red.
    # Coloring: 0,0,0,1 (one blue).
    # Triples: (0,1,2)->0, (0,1,3)->0, (0,2,3)->0, (1,2,3)->1.
    # Red K4? No, because (1,2,3) is blue.
    # So n=4 is always valid if not all red.
    
    # Let's just return a valid small one if all else fails
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}
