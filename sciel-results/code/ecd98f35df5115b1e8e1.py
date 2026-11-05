"""Combined program: greedy + integer verification + randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 60s limit (90s kill, but 60s per eval)
    
    universe = list(range(1, 23))
    
    # Precompute scaled reciprocal values: LCM(1..22) / x
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
    
    scaled = {}
    for x in universe:
        scaled[x] = L // x
    
    def is_dissociated(S):
        """Check if all subset sums of reciprocals of S are distinct using integer arithmetic."""
        if not S:
            return True
        sums = {0}
        for x in S:
            val = scaled[x]
            # Check collision before updating
            for s in sums:
                if (s + val) in sums:
                    return False
            # Update sums
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        return True

    def try_add(S, x):
        """Check if S union {x} is dissociated."""
        return is_dissociated(S + [x])

    best_set = []
    best_size = 0
    
    def greedy_construct(order):
        """Greedy construction maintaining subset sums incrementally."""
        current_set = []
        sums = {0}
        for x in order:
            val = scaled[x]
            # Check collision
            collision = False
            for s in sums:
                if (s + val) in sums:
                    collision = True
                    break
            if not collision:
                current_set.append(x)
                new_sums = [s + val for s in sums]
                sums.update(new_sums)
        return current_set
    
    # Strategy 1: Greedy with different deterministic orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        [n for n in universe if n >= 12],
        [n for n in universe if n % 2 == 0],
        [n for n in universe if n % 2 == 1],
        sorted(universe, key=lambda x: -scaled[x]),  # Largest reciprocals first
        sorted(universe, key=lambda x: scaled[x]),   # Smallest reciprocals first
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        current_set = greedy_construct(order)
        
        # Try to expand with remaining elements
        remaining = [x for x in universe if x not in current_set]
        sums = {0}
        for x in current_set:
            val = scaled[x]
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                val = scaled[x]
                collision = False
                for s in sums:
                    if (s + val) in sums:
                        collision = True
                        break
                if not collision:
                    current_set.append(x)
                    new_sums = [s + val for s in sums]
                    sums.update(new_sums)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Strategy 2: Randomized restarts with incremental sums
    random.seed(42)
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        order = universe[:]
        random.shuffle(order)
        
        current_set = greedy_construct(order)
        
        # Try to expand with remaining elements
        remaining = [x for x in universe if x not in current_set]
        sums = {0}
        for x in current_set:
            val = scaled[x]
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                val = scaled[x]
                collision = False
                for s in sums:
                    if (s + val) in sums:
                        collision = True
                        break
                if not collision:
                    current_set.append(x)
                    new_sums = [s + val for s in sums]
                    sums.update(new_sums)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we found a very large set, we can stop early
            if best_size >= 18:
                break
    
    # Fallback to a known valid set if search fails
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
