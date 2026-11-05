import time
import random
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute all 3-APs
    # A 3-AP is (a, b, c) where a + c = 2b, a < b < c
    ap_list = []  # List of (a, b, c) tuples
    for b in range(1, n + 1):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n and c > b:
                ap_list.append((a, b, c))
    
    # Build conflict graph: for each number, which other numbers conflict with it
    # Two numbers conflict if they appear together in some 3-AP
    # Actually, we need: if we add x, which pairs (y,z) are forbidden?
    # Better: for each x, precompute the set of (y,z) pairs such that x,y,z form a 3-AP
    
    # For efficient checking, for each x, store:
    # - pairs where x is the middle: (y, z) with y + z = 2x
    # - pairs where x is the left endpoint: (y, z) with x + z = 2y, i.e., z = 2y - x
    # - pairs where x is the right endpoint: (y, z) with y + x = 2z, i.e., y = 2z - x
    
    conflict_pairs = [[] for _ in range(n + 1)]
    for a, b, c in ap_list:
        # a, b, c form a 3-AP
        # If we have a and b, we can't have c
        # If we have a and c, we can't have b
        # If we have b and c, we can't have a
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Reciprocals
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i
    
    def can_add(x, in_set):
        """Check if x can be added to the set represented by in_set (boolean array)."""
        for y, z in conflict_pairs[x]:
            if in_set[y] and in_set[z]:
                return False
        return True
    
    def get_sum(in_set):
        total = 0.0
        for i in range(1, n + 1):
            if in_set[i]:
                total += recip[i]
        return total
    
    def get_exact_sum(in_set):
        total = Fraction(0, 1)
        for i in range(1, n + 1):
            if in_set[i]:
                total += Fraction(1, i)
        return total
    
    def count_elements(in_set):
        return sum(in_set[1:])
    
    # Greedy: add elements in order of decreasing weight (smallest first)
    def greedy_construct():
        in_set = [False] * (n + 1)
        for x in range(1, n + 1):
            if can_add(x, in_set):
                in_set[x] = True
        return in_set
    
    # Local search: try single swaps and double swaps
    def local_search(in_set, max_iters=1000):
        current_sum = get_sum(in_set)
        elements = [i for i in range(1, n + 1) if in_set[i]]
        non_elements = [i for i in range(1, n + 1) if not in_set[i]]
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try single swaps: remove one, add one
            for rem in elements[:]:
                if not in_set[rem]:
                    continue
                in_set[rem] = False
                # Try adding each non-element
                best_add = None
                best_gain = 0.0
                for add in non_elements:
                    if in_set[add]:
                        continue
                    if can_add(add, in_set):
                        gain = recip[add] - recip[rem]
                        if gain > best_gain + 1e-15:
                            best_gain = gain
                            best_add = add
                if best_add is not None:
                    in_set[best_add] = True
                    elements.remove(rem)
                    non_elements.remove(best_add)
                    elements.append(best_add)
                    non_elements.append(rem)
                    current_sum += best_gain
                    improved = True
                else:
                    in_set[rem] = True
            
            # Try double swaps: remove two, add two
            if not improved and time.time() < deadline - 10.0:
                n_elem = len(elements)
                found = False
                for i in range(n_elem):
                    if time.time() > deadline - 10.0:
                        break
                    rem1 = elements[i]
                    for j in range(i + 1, n_elem):
                        if time.time() > deadline - 10.0:
                            break
                        rem2 = elements[j]
                        in_set[rem1] = False
                        in_set[rem2] = False
                        
                        # Try adding pairs
                        best_pair = None
                        best_gain = 0.0
                        for add1 in non_elements:
                            if in_set[add1]:
                                continue
                            if not can_add(add1, in_set):
                                continue
                            in_set[add1] = True
                            for add2 in non_elements:
                                if in_set[add2] or add2 == add1:
                                    continue
                                if can_add(add2, in_set):
                                    gain = recip[add1] + recip[add2] - recip[rem1] - recip[rem2]
                                    if gain > best_gain + 1e-15:
                                        best_gain = gain
                                        best_pair = (add1, add2)
                            in_set[add1] = False
                        
                        if best_pair is not None:
                            add1, add2 = best_pair
                            in_set[add1] = True
                            in_set[add2] = True
                            elements.remove(rem1)
                            elements.remove(rem2)
                            non_elements.remove(add1)
                            non_elements.remove(add2)
                            elements.extend([add1, add2])
                            non_elements.extend([rem1, rem2])
                            current_sum += best_gain
                            improved = True
                            found = True
                            break
                        else:
                            in_set[rem1] = True
                            in_set[rem2] = True
                        
                        if found:
                            break
                    if found:
                        break
                if not found:
                    # Restore if we broke out early
                    pass
        
        return in_set
    
    # Simulated annealing
    def simulated_annealing(in_set, max_iters=5000):
        current_sum = get_sum(in_set)
        best_in_set = in_set[:]
        best_sum = current_sum
        
        elements = [i for i in range(1, n + 1) if in_set[i]]
        non_elements = [i for i in range(1, n + 1) if not in_set[i]]
        
        T_start = 0.1
        T_end = 0.0001
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            T *= cooling
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5 and elements:
                # Single swap
                rem = random.choice(elements)
                if not non_elements:
                    continue
                add = random.choice(non_elements)
                
                in_set[rem] = False
                if can_add(add, in_set):
                    in_set[add] = True
                    delta = recip[add] - recip[rem]
                    new_sum = current_sum + delta
                    
                    if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                        current_sum = new_sum
                        elements.remove(rem)
                        non_elements.remove(add)
                        elements.append(add)
                        non_elements.append(rem)
                        
                        if current_sum > best_sum + 1e-15:
                            best_sum = current_sum
                            best_in_set = in_set[:]
                    else:
                        in_set[add] = False
                in_set[rem] = True
                
            elif move_type < 0.8 and len(elements) >= 2 and len(non_elements) >= 2:
                # Double swap
                rem1, rem2 = random.sample(elements, 2)
                add1, add2 = random.sample(non_elements, 2)
                
                in_set[rem1] = False
                in_set[rem2] = False
                
                valid = True
                if not can_add(add1, in_set):
                    valid = False
                else:
                    in_set[add1] = True
                    if not can_add(add2, in_set):
                        valid = False
                        in_set[add1] = False
                    else:
                        in_set[add2] = True
                
                if valid:
                    delta = recip[add1] + recip[add2] - recip[rem1] - recip[rem2]
                    new_sum = current_sum + delta
                    
                    if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                        current_sum = new_sum
                        elements.remove(rem1)
                        elements.remove(rem2)
                        non_elements.remove(add1)
                        non_elements.remove(add2)
                        elements.extend([add1, add2])
                        non_elements.extend([rem1, rem2])
                        
                        if current_sum > best_sum + 1e-15:
                            best_sum = current_sum
                            best_in_set = in_set[:]
                    else:
                        in_set[add1] = False
                        in_set[add2] = False
                else:
                    pass
                
                in_set[rem1] = True
                in_set[rem2] = True
                
            elif move_type < 0.9 and elements:
                # Remove only
                rem = random.choice(elements)
                in_set[rem] = False
                delta = -recip[rem]
                new_sum = current_sum + delta
                
                if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                    current_sum = new_sum
                    elements.remove(rem)
                    non_elements.append(rem)
                    
                    if current_sum > best_sum + 1e-15:
                        best_sum = current_sum
                        best_in_set = in_set[:]
                else:
                    in_set[rem] = True
            
            else:
                # Add only
                if not non_elements:
                    continue
                add = random.choice(non_elements)
                if can_add(add, in_set):
                    in_set[add] = True
                    delta = recip[add]
                    new_sum = current_sum + delta
                    
                    if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                        current_sum = new_sum
                        non_elements.remove(add)
                        elements.append(add)
                        
                        if current_sum > best_sum + 1e-15:
                            best_sum = current_sum
                            best_in_set = in_set[:]
                    else:
                        in_set[add] = False
        
        return best_in_set
    
    # Main algorithm
    random.seed(42)
    
    # Phase 1: Greedy construction
    best_in_set = greedy_construct()
    best_sum = get_sum(best_in_set)
    
    # Phase 2: Local search
    if time.time() < deadline - 15.0:
        improved = local_search(best_in_set[:])
        s = get_sum(improved)
        if s > best_sum + 1e-15:
            best_in_set = improved
            best_sum = s
    
    # Phase 3: Simulated annealing with restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the current best
        perturbed = best_in_set[:]
        num_removals = random.randint(2, 8)
        elements = [i for i in range(1, n + 1) if perturbed[i]]
        
        for _ in range(min(num_removals, len(elements))):
            if not elements:
                break
            idx = random.randint(0, len(elements) - 1)
            rem = elements[idx]
            perturbed[rem] = False
            elements.pop(idx)
        
        # Run simulated annealing
        sa_result = simulated_annealing(perturbed, max_iters=3000)
        sa_sum = get_sum(sa_result)
        
        if sa_sum > best_sum + 1e-15:
            best_in_set = sa_result
            best_sum = sa_sum
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5.0:
        polished = local_search(best_in_set[:])
        p_sum = get_sum(polished)
        if p_sum > best_sum + 1e-15:
            best_in_set = polished
            best_sum = p_sum
    
    # Extract result
    result = [i for i in range(1, n + 1) if best_in_set[i]]
    
    # Validation
    s_set = set(result)
    for a, b, c in ap_list:
        if a in s_set and b in s_set and c in s_set:
            raise AssertionError(f"3-AP found: {a}, {b}, {c}")
    
    return result
