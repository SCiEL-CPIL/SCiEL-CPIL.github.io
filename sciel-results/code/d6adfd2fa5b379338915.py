import time
import random
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    # The universe of candidates
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_collision_int(sums_set, new_val):
        """Check if adding new_val causes any collision using integer representation."""
        for s in sums_set:
            if (s + new_val) in sums_set:
                return True
        return False

    def try_order(order, use_int=False):
        """Try to build a set by greedily adding elements in the given order."""
        current_set = []
        if use_int:
            # Use integer representation: multiply all by LCM of all denominators
            # But we'll use a different approach: keep track of sums as integers
            # where sum = (numerator) / LCM, and we check collisions in numerator space
            # Actually, let's use a simpler approach: just use Fraction but be smart
            pass
        
        # Use Fraction for correctness, but optimize the collision check
        from fractions import Fraction
        current_set = []
        sums = {Fraction(0)}
        
        for x in order:
            val = Fraction(1, x)
            if not check_collision_int(sums, val):
                current_set.append(x)
                new_sums = [s + val for s in sums]
                sums.update(new_sums)
                
        return current_set

    def backtrack_search(candidates, current_set, sums, remaining, time_limit):
        """Backtracking search to find a large dissociated set."""
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Pruning: if we can't beat best_size, return
        if len(current_set) + len(remaining) <= best_size:
            return
        
        # Try adding the first element
        first = remaining[0]
        rest = remaining[1:]
        val = Fraction(1, first)
        
        # Try including first
        if not check_collision_int(sums, val):
            new_sums = sums | {s + val for s in sums}
            backtrack_search(candidates, current_set + [first], new_sums, rest, time_limit)
        
        # Try excluding first
        backtrack_search(candidates, current_set, sums, rest, time_limit)
    
    # Strategy 1: Greedy with different orders
    orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        [n for n in sorted(candidates, reverse=True) if n > 1],
        [n for n in sorted(candidates, reverse=True) if n > 10],
        [n for n in candidates if n % 2 == 0],
        [n for n in candidates if n % 2 != 0],
        # Primes
        [n for n in candidates if all(n % p != 0 for p in range(2, int(n**0.5)+1))],
        # Composites
        [n for n in candidates if not all(n % p != 0 for p in range(2, int(n**0.5)+1))],
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        s = try_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Strategy 2: Randomized greedy
    for _ in range(30):
        if time.time() - start_time > timeout:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        s = try_order(shuffled)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Strategy 3: Local search - try to extend the best set
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in candidates if x not in best_set_set]
        
        # Try adding one element at a time
        from fractions import Fraction
        current_sums = {Fraction(0)}
        for x in best_set:
            val = Fraction(1, x)
            current_sums = current_sums | {s + val for s in current_sums}
        
        for x in remaining:
            if time.time() - start_time > timeout:
                break
            val = Fraction(1, x)
            if not check_collision_int(current_sums, val):
                # Can add x
                new_set = best_set + [x]
                new_sums = current_sums | {s + val for s in current_sums}
                # Try to add more
                further_remaining = [y for y in remaining if y != x and y not in best_set_set]
                for y in further_remaining:
                    if time.time() - start_time > timeout:
                        break
                    val_y = Fraction(1, y)
                    if not check_collision_int(new_sums, val_y):
                        new_set2 = new_set + [y]
                        if len(new_set2) > best_size:
                            best_size = len(new_set2)
                            best_set = new_set2
                            current_sums = new_sums | {s + val_y for s in new_sums}
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                    current_sums = new_sums
    
    # Strategy 4: Backtracking from a good starting point
    if time.time() - start_time < timeout - 10:
        # Start from the best set found and try to extend via backtracking
        best_set_set = set(best_set)
        remaining = [x for x in candidates if x not in best_set_set]
        
        from fractions import Fraction
        current_sums = {Fraction(0)}
        for x in best_set:
            val = Fraction(1, x)
            current_sums = current_sums | {s + val for s in current_sums}
        
        # Backtrack on remaining elements
        def extend_backtrack(remaining, current_set, sums, time_limit):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if not remaining:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            # Pruning
            if len(current_set) + len(remaining) <= best_size:
                return
            
            first = remaining[0]
            rest = remaining[1:]
            val = Fraction(1, first)
            
            # Try including first
            if not check_collision_int(sums, val):
                new_sums = sums | {s + val for s in sums}
                extend_backtrack(rest, current_set + [first], new_sums, time_limit)
            
            # Try excluding first
            extend_backtrack(rest, current_set, sums, time_limit)
        
        extend_backtrack(remaining, best_set[:], current_sums, timeout)
    
    if not best_set:
        best_set = [22]  # Fallback

    return best_set
# EVOLVE-BLOCK-END
