from fractions import Fraction
import random
import time
import math

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution
    best_set = set()
    best_deficit = Fraction(1, 1)
    
    current_set = set()
    current_total = Fraction(0, 1)
    for i in range(n):
        if current_total + recips[i] < 1:
            current_set.add(i)
            current_total += recips[i]
    deficit = 1 - current_total
    if deficit < best_deficit:
        best_deficit = deficit
        best_set = set(current_set)
    
    random.seed(42)
    start_time = time.time()
    max_time = 55.0
    
    # Simulated Annealing
    # We want to minimize deficit = 1 - sum(1/d_i) subject to sum(1/d_i) < 1
    # Deficit is positive. We want to minimize it.
    
    current_set = set(best_set)
    current_total = Fraction(0, 1)
    for i in current_set:
        current_total += recips[i]
    current_deficit = 1 - current_total
    
    best_set_sa = set(current_set)
    best_deficit_sa = current_deficit
    
    # Parameters
    T = 0.01
    cooling = 0.9999
    min_T = 1e-6
    
    while time.time() - start_time < max_time and T > min_T:
        # Generate a random neighbor
        move_type = random.choice(['add', 'remove', 'swap'])
        new_set = set(current_set)
        
        if move_type == 'add':
            # Pick a random element not in set
            candidates = [i for i in range(n) if i not in current_set]
            if not candidates: continue
            i = random.choice(candidates)
            new_set.add(i)
        elif move_type == 'remove':
            if not current_set: continue
            i = random.choice(list(current_set))
            new_set.remove(i)
        elif move_type == 'swap':
            if not current_set: continue
            in_list = list(current_set)
            out_list = [i for i in range(n) if i not in current_set]
            if not out_list: continue
            i = random.choice(in_list)
            j = random.choice(out_list)
            new_set.remove(i)
            new_set.add(j)
        
        # Calculate new total and deficit
        new_total = Fraction(0, 1)
        for i in new_set:
            new_total += recips[i]
        
        if new_total >= 1:
            # Infeasible, reject
            T *= cooling
            continue
            
        new_deficit = 1 - new_total
        
        # Acceptance criteria
        if new_deficit < current_deficit:
            # Better solution
            current_set = new_set
            current_total = new_total
            current_deficit = new_deficit
            
            if current_deficit < best_deficit_sa:
                best_deficit_sa = current_deficit
                best_set_sa = set(current_set)
        else:
            # Worse solution, accept with probability exp(-delta / T)
            delta = (new_deficit - current_deficit)
            # delta is positive (worsening)
            # We use float for probability calculation for speed
            # Deficit is a Fraction, convert to float for SA probability
            delta_float = float(delta)
            if delta_float < 1e-15:
                # Numerical tie, accept
                current_set = new_set
                current_total = new_total
                current_deficit = new_deficit
            else:
                prob = math.exp(-delta_float / T)
                if random.random() < prob:
                    current_set = new_set
                    current_total = new_total
                    current_deficit = new_deficit
        
        T *= cooling
    
    # Update global best
    if best_deficit_sa < best_deficit:
        best_set = best_set_sa
        best_deficit = best_deficit_sa
        
    # Final local search on the best found set to polish
    def local_search(start_set, max_iters=2000):
        current_set = set(start_set)
        current_total = Fraction(0, 1)
        for i in current_set:
            current_total += recips[i]
        if current_total >= 1:
            return None, None
        current_deficit = 1 - current_total
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding
            best_add = -1
            best_add_def = current_deficit
            for i in range(n):
                if i not in current_set:
                    new_total = current_total + recips[i]
                    if new_total < 1:
                        new_def = 1 - new_total
                        if new_def < best_add_def:
                            best_add_def = new_def
                            best_add = i
            if best_add >= 0:
                current_set.add(best_add)
                current_total += recips[best_add]
                current_deficit = best_add_def
                improved = True
                continue
            
            # Try best swap
            best_swap_i = -1
            best_swap_j = -1
            best_swap_def = current_deficit
            for i in list(current_set):
                for j in range(n):
                    if j not in current_set:
                        new_total = current_total - recips[i] + recips[j]
                        if new_total < 1:
                            new_def = 1 - new_total
                            if new_def < best_swap_def:
                                best_swap_def = new_def
                                best_swap_i = i
                                best_swap_j = j
            if best_swap_i >= 0:
                current_set.remove(best_swap_i)
                current_set.add(best_swap_j)
                current_total = current_total - recips[best_swap_i] + recips[best_swap_j]
                current_deficit = best_swap_def
                improved = True
        
        if current_deficit < best_deficit:
            return set(current_set), current_deficit
        return None, None
        
    if time.time() - start_time < max_time:
        new_set, new_def = local_search(best_set, max_iters=5000)
        if new_set is not None:
            best_set = new_set
            best_deficit = new_def

    return sorted(best_set)
