import time
import random
import itertools

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # Safety margin
    
    # Fast validity check
    def is_valid(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        S_list = sorted(S)
        if 2 * S_list[0] > S_list[-1]:
            return True
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True
    
    def is_valid_add(S, x):
        """Check if S | {x} is valid. S is known to be valid."""
        if x in S:
            return True
        S_set = set(S)
        # Check triples involving x: (x, a, b) for a < b in S
        # Need x+a, x+b, a+b all in S | {x}
        # Since a,b in S, a+b is in S | {x} iff a+b in S or a+b == x
        # x+a is in S | {x} iff x+a in S or x+a == x (impossible since a>0)
        # So x+a must be in S, x+b must be in S, and (a+b in S or a+b == x)
        
        # Also check triples where x is a sum: (a, b, c) in S^3 where one of a+b, a+c, b+c == x
        # If a+b == x, then we need a+c in S|{x} and b+c in S|{x}
        # a+c in S|{x} iff a+c in S or a+c == x
        # a+c == x means c = x - a = b (since a+b=x), but c must be distinct from a,b
        # So a+c != x, meaning a+c must be in S. Similarly b+c must be in S.
        
        # Efficient check:
        # 1. For all pairs (a,b) in S with a < b:
        #    if x+a in S_set and x+b in S_set and (a+b in S_set or a+b == x):
        #        return False
        # 2. For all pairs (a,b) in S with a < b and a+b == x:
        #    for all c in S, c != a, c != b:
        #        if a+c in S_set and b+c in S_set:
        #            return False
        
        S_list = sorted(S)
        
        # Check 1: x is the smallest element in the triple
        for i in range(len(S_list)):
            a = S_list[i]
            if x + a not in S_set:
                continue
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if x + b in S_set and (a + b in S_set or a + b == x):
                    return False
        
        # Check 2: x is one of the pair sums
        # Find all pairs (a,b) in S with a+b == x
        pairs_summing_to_x = []
        for i in range(len(S_list)):
            a = S_list[i]
            if x - a in S_set and x - a != a:
                pairs_summing_to_x.append((a, x - a))
        
        for a, b in pairs_summing_to_x:
            for c in S_list:
                if c == a or c == b:
                    continue
                if a + c in S_set and b + c in S_set:
                    return False
        
        return True
    
    # Start with the known valid set [63, 128] of size 66
    best_set = set(range(63, n + 1))
    best_size = len(best_set)
    
    # Local search: try to add elements and remove blockers
    current_set = set(best_set)
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        available = [x for x in range(1, n+1) if x not in current_set]
        if not available:
            break
        
        random.shuffle(available)
        found_addition = False
        
        for candidate in available:
            if is_valid_add(current_set, candidate):
                current_set.add(candidate)
                found_addition = True
                break
        
        if found_addition:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        
        # Try to improve: remove one element to potentially allow more additions
        if time.time() > deadline - 5:
            break
        
        # Try removing a random element and see if we can add two
        if len(current_set) > 10:
            elements = list(current_set)
            random.shuffle(elements)
            for rem in elements:
                if time.time() > deadline:
                    break
                
                test_set = set(current_set)
                test_set.discard(rem)
                
                # Try to add two elements
                available = [x for x in range(1, n+1) if x not in test_set]
                random.shuffle(available)
                added = 0
                for candidate in available:
                    if is_valid_add(test_set, candidate):
                        test_set.add(candidate)
                        added += 1
                        if added >= 2:
                            break
                
                if added >= 2:
                    # Net gain of at least 1
                    if len(test_set) > len(current_set):
                        current_set = test_set
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
    
    # Final check
    if not is_valid(best_set):
        # Fallback
        return list(range(63, n+1))
    
    return list(best_set)
