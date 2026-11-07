import time
import random

def _can_add(sums, x):
    """Check if x can be added to a set with existing subset sums."""
    # x itself must not be in sums (would create duplicate sum value)
    if x in sums:
        return False
    
    # Generate new sums
    new_sums = set()
    for s in sums:
        ns = s + x
        if ns in sums:
            return False
        new_sums.add(ns)
    new_sums.add(x)
    
    # Check divisibility between x and existing sums
    for s in sums:
        if s != x:
            if s % x == 0 or x % s == 0:
                return False
    
    # Check divisibility between new_sums and existing sums
    for ns in new_sums:
        for s in sums:
            if ns != s:
                if ns % s == 0 or s % ns == 0:
                    return False
    
    # Check divisibility within new_sums
    new_sums_list = sorted(new_sums)
    for i in range(len(new_sums_list)):
        a = new_sums_list[i]
        for j in range(i + 1, len(new_sums_list)):
            b = new_sums_list[j]
            if b % a == 0:
                return False
    
    return True

def construct():
    deadline = time.time() + 85
    best_set = [31, 32, 34]
    best_size = 3
    
    random.seed(42)
    
    # Try multiple strategies
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        # Strategy: start with a random element and greedily add
        start_val = random.randint(50, 256)
        current = [start_val]
        sums = {start_val}
        
        # Candidates: all numbers not already used
        candidates = [x for x in range(1, 257) if x != start_val]
        
        # Score candidates by how few conflicts they create
        # Prefer numbers that are "large" relative to current sums
        # and don't create divisibility issues
        
        # Sort candidates: prefer larger numbers (less likely to divide others)
        candidates.sort(reverse=True)
        
        # Add some randomness to the order
        # Take top 100, shuffle them, then append the rest
        if len(candidates) > 100:
            top = candidates[:100]
            rest = candidates[100:]
            random.shuffle(top)
            candidates = top + rest
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
            
            if _can_add(sums, x):
                current.append(x)
                # Update sums
                new_sums = set()
                for s in sums:
                    new_sums.add(s + x)
                new_sums.add(x)
                sums.update(new_sums)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    return list(best_set)
