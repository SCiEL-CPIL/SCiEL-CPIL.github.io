import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initial set
    N = 256
    all_elements = set(range(2, N + 1))
    
    def build_multiples(S):
        """Precompute multiples[k] = set of x in S such that k divides x."""
        multiples = [set() for _ in range(N + 1)]
        for x in S:
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    multiples[d].add(x)
                    other = x // d
                    if other != d:
                        multiples[other].add(x)
        return multiples
    
    def count_conflicts_for_a(a, S, multiples):
        """Count conflicts for element a in set S.
        Conflict: exists b, c in S, b != c, b != a, c != a such that a | b*c.
        """
        count = 0
        for b in S:
            if b == a:
                continue
            g = gcd(a, b)
            a_prime = a // g
            if a_prime == 1:
                # a | b*c for any c, need c in S, c != a, c != b
                if len(S) > 2:
                    count += 1
            else:
                c_set = multiples[a_prime]
                # Need c in c_set, c != a, c != b
                available = len(c_set)
                if a in c_set:
                    available -= 1
                if b in c_set:
                    available -= 1
                if available > 0:
                    count += 1
        return count
    
    def check_all_conflicts(S, multiples):
        """Return dict {a: conflict_count} for all a in S."""
        counts = {}
        for a in S:
            counts[a] = count_conflicts_for_a(a, S, multiples)
        return counts
    
    def can_add(candidate, S, multiples):
        """Check if candidate can be added to S without creating any conflicts."""
        if candidate in S:
            return True
        
        # Check 1: For each a in S, does adding candidate create conflict for a?
        # Conflict: exists b in S, b != a, b != candidate such that a | candidate * b
        for a in S:
            g = gcd(a, candidate)
            a_prime = a // g
            if a_prime == 1:
                # a | candidate * b always, need b in S, b != a
                if len(S) > 1:
                    return False
            else:
                b_set = multiples[a_prime]
                # Need b in b_set, b != a, b != candidate
                available = len(b_set)
                if a in b_set:
                    available -= 1
                if candidate in b_set:
                    available -= 1
                if available > 0:
                    return False
        
        # Check 2: For each b in S, does candidate | b * c for some c in S, c != b, c != candidate?
        for b in S:
            g = gcd(candidate, b)
            c_prime = candidate // g
            if c_prime == 1:
                # candidate | b * c always, need c in S, c != b
                if len(S) > 1:
                    return False
            else:
                c_set = multiples[c_prime]
                # Need c in c_set, c != b, c != candidate
                available = len(c_set)
                if b in c_set:
                    available -= 1
                if candidate in c_set:
                    available -= 1
                if available > 0:
                    return False
        
        return True
    
    def remove_from_multiples(x, multiples):
        """Remove x from all multiples sets."""
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                multiples[d].discard(x)
                other = x // d
                if other != d:
                    multiples[other].discard(x)
    
    def add_to_multiples(x, multiples):
        """Add x to all multiples sets."""
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                multiples[d].add(x)
                other = x // d
                if other != d:
                    multiples[other].add(x)
    
    best_S = set()
    best_size = 0
    
    # Run multiple randomized greedy removals with local search
    for run in range(150):
        if time.time() > deadline:
            break
            
        S = set(all_elements)
        multiples = build_multiples(S)
        removed = []
        
        # Phase 1: Greedy removal - remove element with highest conflict degree
        while True:
            if time.time() > deadline:
                break
                
            counts = check_all_conflicts(S, multiples)
            bads = [a for a, c in counts.items() if c > 0]
            
            if not bads:
                break
            
            # Select removal candidate with highest conflict count
            max_count = max(counts[a] for a in bads)
            candidates = [a for a in bads if counts[a] == max_count]
            a_to_remove = random.choice(candidates)
            
            S.remove(a_to_remove)
            removed.append(a_to_remove)
            remove_from_multiples(a_to_remove, multiples)
        
        # Phase 2: Add-back phase - try to add back removed elements
        random.shuffle(removed)
        added_any = True
        while added_any:
            if time.time() > deadline:
                break
            added_any = False
            for x in list(removed):
                if x in S:
                    continue
                if can_add(x, S, multiples):
                    S.add(x)
                    removed.remove(x)
                    add_to_multiples(x, multiples)
                    added_any = True
        
        # Phase 3: Local search with swaps - try swapping x from S with y from removed
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Get current conflicts
            counts_S = check_all_conflicts(S, multiples)
            
            for x in list(S):
                if time.time() > deadline:
                    break
                conflict_x = counts_S.get(x, 0)
                if conflict_x == 0:
                    continue  # x has no conflicts, don't remove it
                
                for y in list(removed):
                    if y in S:
                        continue
                    
                    # Try swap: remove x, add y
                    # Temporarily remove x
                    S.remove(x)
                    remove_from_multiples(x, multiples)
                    removed.append(x)
                    
                    # Check if y can be added to the set without x
                    if can_add(y, S, multiples):
                        S.add(y)
                        add_to_multiples(y, multiples)
                        removed.remove(y)
                        # x stays removed, y is now in S
                        improved = True
                    else:
                        # Undo: put x back
                        S.add(x)
                        add_to_multiples(x, multiples)
                        removed.remove(x)
                    
                    if time.time() > deadline:
                        break
            
            # After swap attempts, do another add-back pass
            random.shuffle(removed)
            added_any = True
            while added_any:
                if time.time() > deadline:
                    break
                added_any = False
                for x in list(removed):
                    if x in S:
                        continue
                    if can_add(x, S, multiples):
                        S.add(x)
                        removed.remove(x)
                        add_to_multiples(x, multiples)
                        added_any = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Early stop if we found a very large set
        if best_size > 60:
            break

    # Final verification and cleanup
    # Ensure all elements are Python ints
    result = [int(x) for x in best_S]
    return result
