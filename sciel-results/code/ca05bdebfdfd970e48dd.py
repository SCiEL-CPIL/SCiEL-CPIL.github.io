import time
import random
import sys
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    max_val = 4096
    max_sum = 3 * max_val
    
    best_set = []
    best_size = 0
    
    def get_triple_sums(s):
        """Get all triple sums for a set s (unordered, with repetition)."""
        sums = set()
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                b = s[j]
                for k in range(j, n):
                    c = s[k]
                    sums.add(a + b + c)
        return sums
    
    def can_add(x, current_set, used_sums):
        """Check if x can be added to current_set without creating duplicate triple sums."""
        n = len(current_set)
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        new_sums.add(s)
        
        # (x, x, y) for y in current_set
        for y in current_set:
            s = 2 * x + y
            new_sums.add(s)
        
        # (x, y, y) for y in current_set
        for y in current_set:
            s = x + 2 * y
            new_sums.add(s)
        
        # (x, y, z) for y <= z in current_set
        for i in range(n):
            y = current_set[i]
            for j in range(i, n):
                z = current_set[j]
                s = x + y + z
                new_sums.add(s)
        
        # Check for conflicts
        for s in new_sums:
            if s in used_sums:
                return False, None
        
        return True, new_sums
    
    def greedy_construct(order, used_sums=None):
        """Greedy construction following a given order."""
        if used_sums is None:
            used_sums = set()
        current_set = []
        
        for x in order:
            ok, new_sums = can_add(x, current_set, used_sums)
            if ok:
                current_set.append(x)
                used_sums.update(new_sums)
        
        return current_set, used_sums
    
    def local_improve(s, used_sums, time_limit):
        """Try to improve set s via local search (remove one, add two; or swap)."""
        current_set = list(s)
        current_used = set(used_sums)
        current_size = len(current_set)
        best_local = current_set[:]
        best_local_size = current_size
        
        # Build a set of all elements for quick lookup
        in_set = set(current_set)
        
        # Pre-compute triple sums for current set
        def compute_used_sums(st):
            sums = set()
            n = len(st)
            for i in range(n):
                a = st[i]
                for j in range(i, n):
                    b = st[j]
                    for k in range(j, n):
                        c = st[k]
                        sums.add(a + b + c)
            return sums
        
        # Try: remove one element, add two new ones
        candidates = [x for x in range(1, max_val + 1) if x not in in_set]
        
        # Score candidates by "safety" - prefer those whose triple sums don't conflict much
        # Simple heuristic: prefer larger numbers (they push sums to higher ranges)
        # or numbers that are "far" from existing elements
        
        for attempt in range(200):
            if time.time() > time_limit:
                break
            
            # Try removing a random element and adding two
            if current_size < 2:
                break
                
            idx = random.randint(0, current_size - 1)
            removed = current_set[idx]
            
            # Temporarily remove
            temp_set = current_set[:idx] + current_set[idx+1:]
            temp_used = compute_used_sums(temp_set)
            
            # Try to add two new elements
            added = []
            for _ in range(2):
                if time.time() > time_limit:
                    break
                
                # Pick a random candidate
                if not candidates:
                    break
                x = random.choice(candidates)
                
                ok, new_sums = can_add(x, temp_set, temp_used)
                if ok:
                    temp_set.append(x)
                    temp_used.update(new_sums)
                    added.append(x)
                    candidates.remove(x)
            
            if len(added) == 2:
                new_size = len(temp_set)
                if new_size > best_local_size:
                    best_local = temp_set[:]
                    best_local_size = new_size
                    current_set = temp_set[:]
                    current_used = temp_used
                    current_size = new_size
                    in_set = set(current_set)
                    # Update candidates
                    candidates = [x for x in range(1, max_val + 1) if x not in in_set]
            
            # Also try: swap one element for another
            if time.time() > time_limit:
                break
                
            idx = random.randint(0, current_size - 1)
            old_elem = current_set[idx]
            
            # Try replacing with a different element
            if candidates:
                new_elem = random.choice(candidates)
                temp_set = current_set[:]
                temp_set[idx] = new_elem
                temp_used = compute_used_sums(temp_set)
                
                if len(temp_used) == len(set(sum(temp_used))):  # all unique
                    # Check size is same (swap), so no improvement in size
                    # But might enable future additions - skip for simplicity
                    pass
        
        return best_local, compute_used_sums(best_local)
    
    # Multiple strategies
    strategies = []
    
    # Strategy 1: Ascending order
    strategies.append(list(range(1, max_val + 1)))
    
    # Strategy 2: Descending order  
    strategies.append(list(range(max_val, 0, -1)))
    
    # Strategy 3: Powers of 3 inspired (1, 3, 9, 27, ...) then fill in
    pow3 = [3**i for i in range(10) if 3**i <= max_val]
    rest = [x for x in range(1, max_val + 1) if x not in set(pow3)]
    strategies.append(pow3 + rest)
    
    # Strategy 4: Random orders
    for _ in range(5):
        c = list(range(1, max_val + 1))
        random.shuffle(c)
        strategies.append(c)
    
    # Strategy 5: Modular orders
    for mod in [3, 5, 7, 11]:
        c = sorted(range(1, max_val + 1), key=lambda x: (x % mod, x))
        strategies.append(c)
    
    # Run each strategy
    for order in strategies:
        if time.time() > deadline - 5.0:
            break
        
        s, used = greedy_construct(order)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
        
        # Local improvement
        if time.time() < deadline - 10.0:
            improved, improved_used = local_improve(s, used, deadline - 5.0)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    # Final: try to add more elements to best_set
    if time.time() < deadline - 3.0:
        used = compute_used_sums(best_set)
        in_set = set(best_set)
        current = best_set[:]
        
        # Try adding elements in ascending order
        for x in range(1, max_val + 1):
            if x in in_set:
                continue
            if time.time() > deadline - 2.0:
                break
            ok, new_sums = can_add(x, current, used)
            if ok:
                current.append(x)
                used.update(new_sums)
                in_set.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    return best_set
