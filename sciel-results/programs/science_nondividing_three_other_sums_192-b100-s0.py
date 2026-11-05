import random
import time
from itertools import combinations

def is_valid_fast(s):
    """
    Fast validity check: for each element a, check if any 3 distinct others sum to multiple of a.
    Uses sorted list and early termination.
    """
    lst = sorted(s)
    n = len(lst)
    if n < 4:
        return True
    
    for idx_a, a in enumerate(lst):
        others = lst[:idx_a] + lst[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        
        # If a is larger than the max possible sum of 3 others, no violation
        max_sum3 = others[-1] + others[-2] + others[-3]
        if a > max_sum3:
            continue
        
        # Min sum of 3 others
        min_sum3 = others[0] + others[1] + others[2]
        
        others_set = set(others)
        
        start_mult = (min_sum3 + a - 1) // a
        end_mult = max_sum3 // a
        
        if start_mult > end_mult:
            continue
            
        for k in range(start_mult, end_mult + 1):
            target = k * a
            found = False
            for i in range(m - 2):
                xi = others[i]
                if xi + others[i+1] + others[i+2] > target:
                    break
                for j in range(i + 1, m - 1):
                    xj = others[j]
                    if xi + xj + others[j+1] > target:
                        break
                    rem = target - xi - xj
                    if rem > xj and rem in others_set:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def is_valid_incremental(existing_set, new_elem):
    """
    Check if adding new_elem to existing_set maintains validity.
    Only need to check:
    1. new_elem doesn't divide sum of any 3 existing elements
    2. No existing element divides sum of new_elem + 2 existing elements
    """
    if not existing_set:
        return True
    
    existing_sorted = sorted(existing_set)
    n = len(existing_sorted)
    
    # Check 1: new_elem vs triples from existing_set
    a = new_elem
    if n >= 3:
        max_sum3 = existing_sorted[-1] + existing_sorted[-2] + existing_sorted[-3]
        if a <= max_sum3:
            min_sum3 = existing_sorted[0] + existing_sorted[1] + existing_sorted[2]
            start_mult = (min_sum3 + a - 1) // a
            end_mult = max_sum3 // a
            if start_mult <= end_mult:
                existing_set_s = set(existing_sorted)
                for k in range(start_mult, end_mult + 1):
                    target = k * a
                    found = False
                    for i in range(n - 2):
                        xi = existing_sorted[i]
                        if xi + existing_sorted[i+1] + existing_sorted[i+2] > target:
                            break
                        for j in range(i + 1, n - 1):
                            xj = existing_sorted[j]
                            if xi + xj + existing_sorted[j+1] > target:
                                break
                            rem = target - xi - xj
                            if rem > xj and rem in existing_set_s:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    
    # Check 2: Each existing element vs triples including new_elem
    for idx_a, a in enumerate(existing_sorted):
        # Triples that include new_elem: new_elem + 2 from existing_set (excluding a)
        # Need to check if a divides new_elem + x + y for some x, y in existing_set, x != y, x != a, y != a
        others_excluding_a = [x for x in existing_sorted if x != a]
        m = len(others_excluding_a)
        if m < 2:
            continue
        
        # We need: a | (new_elem + x + y) for some distinct x, y in others_excluding_a
        # Equivalently: (new_elem + x + y) % a == 0
        # i.e., x + y ≡ -new_elem (mod a)
        
        # If a is large, few multiples. If a is small, more to check.
        # Max possible sum: new_elem + others_excluding_a[-1] + others_excluding_a[-2]
        max_sum = new_elem + others_excluding_a[-1] + others_excluding_a[-2]
        if a > max_sum:
            continue
        
        min_sum = new_elem + others_excluding_a[0] + others_excluding_a[1]
        
        start_mult = (min_sum + a - 1) // a
        end_mult = max_sum // a
        
        if start_mult > end_mult:
            continue
        
        # For each multiple target = k*a, check if x + y = target - new_elem for distinct x,y in others_excluding_a
        others_set_s = set(others_excluding_a)
        for k in range(start_mult, end_mult + 1):
            target = k * a
            need = target - new_elem
            if need < 0:
                continue
            # Check if any pair in others_excluding_a sums to need
            found = False
            for i in range(m - 1):
                xi = others_excluding_a[i]
                if xi + others_excluding_a[i+1] > need:
                    break
                rem = need - xi
                if rem > xi and rem in others_set_s:
                    found = True
                    break
            if found:
                return False
    
    return True

def greedy_construct(seed_rng=None):
    """
    Greedy construction with priority on larger numbers (they're easier to satisfy).
    Also tries multiple orderings.
    """
    best_result = []
    
    # Strategy 1: Descending order
    selected = []
    for x in range(192, 0, -1):
        if is_valid_incremental(set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Strategy 2: Descending but skip some small numbers
    selected = []
    for x in range(192, 0, -1):
        if x <= 10:
            # Be more selective with small numbers
            if len(selected) < 40:
                if is_valid_incremental(set(selected), x):
                    selected.append(x)
        else:
            if is_valid_incremental(set(selected), x):
                selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Strategy 3: Start with large primes (harder to be divisors)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    selected = []
    # Add large primes first
    for x in range(192, 0, -1):
        if is_prime(x):
            if is_valid_incremental(set(selected), x):
                selected.append(x)
    # Then add composites
    for x in range(192, 0, -1):
        if x not in selected:
            if is_valid_incremental(set(selected), x):
                selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    rng = random.Random(42)
    
    # Start with multiple greedy strategies
    best = greedy_construct()
    best_set = frozenset(best)
    best_size = len(best)
    
    all_numbers = set(range(1, 193))
    
    def local_search(s, rng, max_iters=500):
        """
        Improved hill-climb with better swap strategies.
        """
        s = set(s)
        cur_size = len(s)
        stall_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding elements (prioritize larger numbers)
            candidates = sorted(all_numbers - s, reverse=True)
            rng.shuffle(candidates)
            for x in candidates[:15]:
                if is_valid_incremental(s, x):
                    s.add(x)
                    cur_size += 1
                    improved = True
                    stall_count = 0
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add another
            elems = list(s)
            rng.shuffle(elems)
            # Prioritize removing smaller numbers (they cause more constraints)
            elems.sort(key=lambda x: x)
            
            for rem in elems[:8]:
                candidates = sorted(all_numbers - s, reverse=True)
                rng.shuffle(candidates)
                for add in candidates[:12]:
                    if add == rem:
                        continue
                    new_s = (s - {rem}) | {add}
                    if is_valid_fast(new_s):
                        s = new_s
                        improved = True
                        stall_count = 0
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove 2, add 2
            if rng.random() < 0.4:
                elems = list(s)
                rng.shuffle(elems)
                # Try removing two smaller elements
                elems.sort(key=lambda x: x)
                rem_pairs = list(combinations(elems[:10], 2))
                rng.shuffle(rem_pairs)
                
                for rem1, rem2 in rem_pairs[:6]:
                    candidates = sorted(all_numbers - s, reverse=True)
                    rng.shuffle(candidates)
                    add_pairs = list(combinations(candidates[:10], 2))
                    for add1, add2 in add_pairs[:8]:
                        new_s = (s - {rem1, rem2}) | {add1, add2}
                        if is_valid_fast(new_s):
                            s = new_s
                            improved = True
                            stall_count = 0
                            break
                    if improved:
                        break
            
            if not improved:
                stall_count += 1
                if stall_count > 30:
                    break
        return s
    
    # Run multiple local search rounds with diverse starting points
    for round_num in range(200):
        if time.time() > deadline:
            break
        
        # Generate diverse starting points
        if round_num == 0:
            start_s = set(best)
        elif round_num == 1:
            start_s = set(greedy_construct())
        else:
            # Random start: take a random subset biased toward larger numbers
            k = rng.randint(35, 45)
            # Weight toward larger numbers
            weights = [x for x in range(1, 193)]
            start_s = set(rng.choices(range(1, 193), weights=weights, k=k))
            
            # Filter to valid set by greedily adding (larger first)
            valid_s = set()
            for x in sorted(start_s, reverse=True):
                if is_valid_incremental(valid_s, x):
                    valid_s.add(x)
            start_s = valid_s
        
        # Local search
        result = local_search(start_s, rng, max_iters=200)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = frozenset(result)
            best = list(result)
            
    return list(best_set)
