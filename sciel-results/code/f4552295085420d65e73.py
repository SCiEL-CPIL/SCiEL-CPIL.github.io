import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
        
    def compute_conflicts(a, sums):
        """
        Returns a set of values x such that x cannot be added to a.
        """
        forbidden = set()
        for y in a:
            for s in sums:
                x = s - y
                if 1 <= x <= 284:
                    forbidden.add(x)
        for s in sums:
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= 284:
                    forbidden.add(x)
        return forbidden

    def get_new_sums(a, x):
        return {x + y for y in a} | {2 * x}

    def greedy_expand_fast(initial_list, order):
        a = list(initial_list)
        sums = compute_sums(a)
        a_set = set(a)
        forbidden = compute_conflicts(a, sums)
        
        for x in order:
            if x in a_set:
                continue
            if x in forbidden:
                continue
            
            a.append(x)
            a_set.add(x)
            new_sums = get_new_sums(a, x)
            sums |= new_sums
            forbidden = compute_conflicts(a, sums)
            
        return a

    def local_search(a, deadline, max_rounds=30):
        """
        Local search: try removing 1-3 elements and refilling greedily.
        Accept if size improves.
        """
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            best_a = list(a)
            best_size = len(a)
            
            # Try removing 1, 2, or 3 non-fixed elements
            non_fixed = [x for x in a if x not in fixed_set]
            
            for rem_size in range(1, 4):
                if len(non_fixed) < rem_size:
                    break
                
                # Try several random removals
                for trial in range(15):
                    if time.time() > deadline:
                        break
                    
                    if len(non_fixed) < rem_size:
                        break
                    
                    removed = random.sample(non_fixed, rem_size)
                    temp_a = [x for x in a if x not in removed]
                    
                    # Refill greedily in random order
                    candidates = [x for x in range(1, 285) if x not in set(temp_a)]
                    random.shuffle(candidates)
                    
                    temp_result = greedy_expand_fast(temp_a, candidates)
                    
                    if len(temp_result) > best_size:
                        best_size = len(temp_result)
                        best_a = temp_result
                        
                        if best_size >= 20:
                            break
                
                if best_size >= 20:
                    break
            
            if best_size > len(a):
                a = best_a
                improved = True
            else:
                # Try single swap: remove one, add best possible
                if len(a) < 20:
                    # Try to just add any valid element
                    sums_cur = compute_sums(a)
                    a_set_cur = set(a)
                    forbidden = compute_conflicts(a, sums_cur)
                    available = [x for x in range(1, 285) if x not in a_set_cur and x not in forbidden]
                    if available:
                        random.shuffle(available)
                        for x in available[:10]:
                            new_a = a + [x]
                            if len(new_a) > len(a):
                                a = new_a
                                improved = True
                                break
                if not improved:
                    break
        
        return a

    # Main search logic
    best = []
    
    # Strategy 1: Ascending order
    a1 = greedy_expand_fast(fixed_elements, list(range(12, 285)))
    if len(a1) > len(best):
        best = a1
    
    # Strategy 2: Descending order
    a2 = greedy_expand_fast(fixed_elements, list(range(284, 11, -1)))
    if len(a2) > len(best):
        best = a2

    # Strategy 3: Multiple randomized greedy with local search
    random.seed(42)
    
    # Do several passes with different random seeds/orders
    for restart in range(50):
        if time.time() > deadline:
            break
        
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a3 = greedy_expand_fast(fixed_elements, candidates)
        
        # Apply local search
        a3_improved = local_search(a3, deadline, max_rounds=15)
        
        if len(a3_improved) > len(best):
            best = a3_improved
            if len(best) >= 20:
                break

    # Final polish with more local search rounds
    if len(best) < 20:
        best = local_search(best, deadline, max_rounds=50)
    
    # Final attempt: try to add any remaining valid elements
    if len(best) < 284:
        best_set = set(best)
        best_sums = compute_sums(best)
        forbidden = compute_conflicts(best, best_sums)
        available = [x for x in range(1, 285) if x not in best_set and x not in forbidden]
        for x in available:
            best.append(x)
    
    return best
