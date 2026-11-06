import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 192
    max_size = 0
    best_set = []
    
    def is_valid_add(x, selected):
        # 1. Check if x divides sum of any 3 distinct others in selected
        if len(selected) >= 3:
            # Optimization: 
            # If x is large, few multiples.
            # Just check all triples.
            for t in combinations(selected, 3):
                if sum(t) % x == 0:
                    return False
        
        # 2. Check if any a in selected divides sum of {x, b, c}
        # where b, c are distinct from a and each other, and in selected.
        for a in selected:
            # We need b, c in selected \ {a} such that (x + b + c) % a == 0
            # Let target = x + b + c
            # target % a == 0
            
            # Iterate pairs b, c from selected, excluding a
            # To avoid O(N^2) per a, we can just do it.
            # N is small.
            
            # Optimization: 
            # Precompute list of values in selected != a
            # vals = [v for v in selected if v != a]
            # This is O(N).
            # Then pairs O(N^2).
            # Total O(N^3).
            
            # Let's try to optimize the inner loop.
            # For a fixed a and x, we want to know if there exist b, c in S\{a} 
            # such that b + c = k*a - x.
            # Let R = k*a - x.
            # We need b + c = R.
            # This is a 2-sum problem.
            # We can use a set for S\{a}.
            
            # However, k is variable.
            # Range of b+c: min ~ 3, max ~ 384.
            # So k*a - x must be in [3, 384].
            # k*a in [x+3, x+384].
            # k is roughly (x+192)/a.
            
            # Let's just use the pair iteration, it's simpler and N is small.
            
            # To speed up, we can break early.
            
            # Create a set of available values for 2-sum?
            # avail = set(selected) - {a}
            # for b in avail:
            #   for c in avail:
            #     if b == c: continue
            #     if (x + b + c) % a == 0: return False
            
            # This is O(N^2).
            
            pass # Implemented below

    # Faster implementation
    def check_conflict(x, selected):
        # 1. x as divisor
        if len(selected) >= 3:
            for b, c, d in combinations(selected, 3):
                if (b + c + d) % x == 0:
                    return True
        
        # 2. a in selected as divisor
        # For each a, check if any pair b, c in selected\{a} satisfies (x+b+c) % a == 0
        for a in selected:
            # Optimization: 
            # We can precompute a set of selected values?
            # Let's just iterate.
            # To avoid b=c, we iterate indices.
            n_sel = len(selected)
            # List of (index, value) excluding a?
            # Just value is fine.
            
            # We can use a set for O(1) lookup?
            # We need b + c = k*a - x.
            # Let's iterate b.
            # For each b, we need c = k*a - x - b.
            # c must be in selected, c != a, c != b.
            
            # Range of k:
            # Min sum b+c = 1+2=3 (if 1,2 in set).
            # Max sum b+c = 191+192=383.
            # So x + b + c is in [x+3, x+383].
            # k*a is in [x+3, x+383].
            # k_min = ceil((x+3)/a)
            # k_max = floor((x+383)/a)
            
            k_min = (x + 3 + a - 1) // a
            k_max = (x + 383) // a
            
            if k_min > k_max:
                continue
                
            # For each k in [k_min, k_max]:
            # Target sum for b+c is S_target = k*a - x
            # We need b, c in selected\{a}, b < c, b+c = S_target.
            
            # We can use a set of selected values for fast lookup.
            # But we need to ensure b != c and b,c != a.
            
            # Let's create a set of values in selected, excluding a.
            # This is O(N).
            # Do this outside the k loop?
            # Yes.
            
            pass # Logic inside loop

    # Let's implement the optimized check
    def is_valid_add(x, selected):
        # 1. x divides sum of 3
        if len(selected) >= 3:
            for t in combinations(selected, 3):
                if sum(t) % x == 0:
                    return False
        
        # 2. a divides sum of x, b, c
        # Precompute set of selected values
        # Note: selected contains distinct values.
        
        for a in selected:
            # Create set of values in selected, excluding a
            # To speed up, we can just check pairs directly if N is small.
            # N ~ 40. N^2 = 1600.
            # 40 * 1600 = 64000 checks.
            
            # Let's use the k-range optimization to reduce checks.
            # But set lookup is O(1).
            
            # avail_set = set(selected) - {a}
            # This is O(N).
            
            # We need to check if there exist b, c in avail_set, b != c, 
            # such that (x + b + c) % a == 0.
            
            # Let's iterate b in avail_set.
            # For each b, we need c such that c = k*a - x - b.
            # c must be in avail_set and c != b.
            
            # k range:
            # b, c >= 1.
            # x + b + c >= x + 1 + 2 = x + 3 (if 1,2 in set, but min values in set might be higher)
            # Let min_val = min(selected) if selected else 1
            # Let max_val = max(selected) if selected else 192
            # Min b+c = 2 * min_val (if min_val is in set and distinct? No, b!=c)
            # Actually, just use 3 and 383 as bounds.
            
            k_min = max(1, (x + 3 + a - 1) // a)
            k_max = (x + 383) // a
            
            if k_min > k_max:
                continue
            
            # For efficiency, create a set of available values once per a
            # But we are inside the loop for a.
            # Let's create avail_set = set(selected) - {a}
            
            avail_set = set(selected)
            avail_set.discard(a)
            
            found = False
            for b in avail_set:
                for k in range(k_min, k_max + 1):
                    c = k * a - x - b
                    if c == b:
                        continue
                    if c in avail_set:
                        # Check distinctness and validity
                        # b, c are in avail_set, so they are in selected and != a.
                        # b != c is checked.
                        # Are b, c distinct from x? Yes, x not in selected.
                        # So {x, b, c} are distinct.
                        found = True
                        break
                if found:
                    break
            if found:
                return False
                
        return True

    def greedy(order):
        selected = []
        for x in order:
            if is_valid_add(x, selected):
                selected.append(x)
        return selected

    # 1. Descending
    s1 = greedy(range(192, 0, -1))
    if len(s1) > max_size:
        max_size = len(s1)
        best_set = s1

    # 2. Ascending
    s2 = greedy(range(1, 193))
    if len(s2) > max_size:
        max_size = len(s2)
        best_set = s2

    # 3. Random restarts
    random.seed(42)
    while time.time() < deadline:
        order = list(range(1, 193))
        random.shuffle(order)
        s = greedy(order)
        if len(s) > max_size:
            max_size = len(s)
            best_set = s

    return best_set
