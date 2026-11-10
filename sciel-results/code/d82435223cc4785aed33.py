import itertools

def construct():
    n = 256
    
    # Step 1: Greedy Construction
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        # Check if x forms a 3-AP with any pair in selected
        # Since we add in increasing order, x is the largest, so it can only be the 3rd term.
        # Check if there exist a, b in selected such that 2b - a == x
        valid = True
        # Optimization: iterate through selected
        for b in selected:
            # a = 2b - x
            a = 2 * b - x
            if a in selected_set:
                valid = False
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    
    current_sum = sum(1.0 / x for x in selected)
    best_set = set(selected)
    best_sum = current_sum
    
    # Step 2: Local Search (Hill Climbing)
    # We try to improve by swapping elements.
    # A swap (remove x, add y) improves the sum if 1/y > 1/x, i.e., y < x.
    # So we only need to consider y < x for potential improvements.
    
    improved = True
    while improved:
        improved = False
        # To avoid infinite loops or oscillation, we might need to be careful,
        # but strictly increasing sum ensures termination.
        
        # Sort elements to iterate in a stable order
        current_list = sorted(best_set)
        
        for x in current_list:
            if x not in best_set:
                continue # In case set changed? No, we are iterating a copy.
            
            S_minus_x = best_set - {x}
            base_val = best_sum - (1.0 / x)
            
            # We want to add y such that y < x (to increase reciprocal sum)
            # and y not in S_minus_x.
            # Check y from 1 to x-1
            best_y = None
            best_gain = 0.0
            
            for y in range(1, x):
                if y in S_minus_x:
                    continue
                
                # Check if y is valid in S_minus_x
                # y is the smallest, so it can be the 1st term of a 3-AP: y, y+d, y+2d
                # Or y can be the middle term? No, y is small.
                # Or y can be the 3rd term? No, y is small.
                # Actually, y can participate in any AP.
                # Check if adding y creates a 3-AP with any pair in S_minus_x.
                # Pairs (a, b) in S_minus_x such that:
                # 1. y, a, b is AP -> 2a = y + b -> b = 2a - y
                # 2. a, y, b is AP -> 2y = a + b -> b = 2y - a
                # 3. a, b, y is AP -> 2b = a + y -> a = 2b - y (but a,b > y usually? not necessarily)
                
                # Efficient check:
                valid = True
                # For each a in S_minus_x:
                #   If 2a - y in S_minus_x and 2a - y != y and 2a - y != a: AP (y, a, 2a-y)
                #   If 2y - a in S_minus_x and 2y - a != y and 2y - a != a: AP (a, y, 2y-a)
                #   If 2b - y == a? This is covered by symmetry if we iterate all a.
                #   Wait, the condition "no 3-AP" means no distinct a,b,c in set such that a+c=2b.
                #   If we add y, we must ensure no (y, a, b), (a, y, b), (a, b, y) exist.
                
                # Let's just check all pairs involving y.
                # For each a in S_minus_x:
                #   Check if (y, a, 2a-y) is a valid AP in S_minus_x U {y}
                #   Check if (a, y, 2y-a) is a valid AP in S_minus_x U {y}
                #   Check if (2a-y, y, a) is a valid AP? Same as first.
                
                # More simply:
                # For each a in S_minus_x:
                #   b1 = 2*a - y
                #   if b1 in S_minus_x and b1 != a and b1 != y: invalid
                #   b2 = 2*y - a
                #   if b2 in S_minus_x and b2 != a and b2 != y: invalid
                #   b3 = a + y # if a,y are 1st, 2nd? No.
                #   Let's stick to the definition: a, b, c are in set.
                #   If y is in the set, it must not form an AP with any two other elements.
                #   So for any a, b in S_minus_x:
                #     if a+b == 2*y: invalid (a, y, b)
                #     if a+y == 2*b: invalid (a, b, y) -> b = (a+y)/2
                #     if b+y == 2*a: invalid (b, a, y) -> a = (b+y)/2
                
                # Iterating all pairs is O(N^2). N=256, so ~65k ops per y.
                # 256 y's * 256 x's * 65k = 4 billion. Too slow.
                
                # Optimization:
                # When checking validity of y in S_minus_x:
                # 1. Check if there exists a in S_minus_x such that 2a - y in S_minus_x.
                #    This covers APs where y is the smallest term (y, a, 2a-y).
                #    Note: if 2a-y == y, then a=y, but y not in S_minus_x.
                #    If 2a-y == a, then a=y, impossible.
                # 2. Check if there exists a in S_minus_x such that 2y - a in S_minus_x.
                #    This covers APs where y is the largest term? No.
                #    If AP is (a, y, b), then 2y = a + b => b = 2y - a.
                #    So if b is in S_minus_x, invalid.
                # 3. Check if there exists a, b in S_minus_x such that a + b = 2y.
                #    This is the same as 2.
                
                # So we just need to check:
                # For a in S_minus_x:
                #   if (2*a - y) in S_minus_x: invalid
                #   if (2*y - a) in S_minus_x: invalid
                
                # This is O(N) per y.
                # Total ops: 256 (x) * 256 (y) * 256 (a) = 16M. This is fine.
                
                for a in S_minus_x:
                    b1 = 2 * a - y
                    if b1 in S_minus_x and b1 != a and b1 != y:
                        valid = False
                        break
                    b2 = 2 * y - a
                    if b2 in S_minus_x and b2 != a and b2 != y:
                        valid = False
                        break
                    if not valid:
                        break
                
                if valid:
                    gain = (1.0 / y) - (1.0 / x)
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
            
            if best_y is not None:
                best_set = S_minus_x | {best_y}
                best_sum += best_gain
                improved = True
                # Restart the outer loop to allow further improvements
                break
    
    return list(best_set)
