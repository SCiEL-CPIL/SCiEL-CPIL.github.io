import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    
    Strategy:
    1. Start with all integers in [129, 256] (size 128).
    2. Greedily remove the most-conflicted element until valid.
    3. Then perform local search with 2-for-1 swaps using incremental validation.
    """
    deadline = time.time() + 55  # 55 second safety margin
    
    # Start with all numbers in [129, 256]
    S = list(range(129, 257))
    S_set = set(S)
    
    def count_violations_fast(S):
        """Count violations efficiently using a set for O(1) lookup."""
        S_set_local = set(S)
        n = len(S)
        conflict_count = {x: 0 for x in S}
        
        for a in S:
            for b in S:
                if a == b:
                    continue
                g = math.gcd(a, b)
                a_prime = a // g
                if a_prime == 1:
                    continue
                # Find c in S, c != a, c != b, c % a_prime == 0
                # We need c % a_prime == 0, c in S, c != a, c != b
                # Since all elements are >= 129, and a_prime < a <= 256,
                # a_prime can be at most 256. c must be a multiple of a_prime in S.
                # Multiples of a_prime in [129, 256]: at most 256/a_prime - 129/a_prime + 1
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % a_prime == 0:
                        conflict_count[a] += 1
                        conflict_count[b] += 1
                        conflict_count[c] += 1
        
        return conflict_count
    
    def is_valid_fast(S):
        """Check if S is valid (no violations). Early exit."""
        S_set_local = set(S)
        n = len(S)
        
        for a in S:
            for b in S:
                if a == b:
                    continue
                g = math.gcd(a, b)
                a_prime = a // g
                if a_prime == 1:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % a_prime == 0:
                        return False
        return True
    
    def check_swap_validity(S, S_set, x, y):
        """
        Check if removing y and adding x to S gives a valid set.
        Only need to check violations involving x (new) or that were resolved by removing y.
        Actually, we need to check:
        1. No new violation involving x
        2. No existing violation that was 'masked' by y's presence (but removing y can't create violations)
        
        Wait - removing y can only remove violations, not create them. Adding x can only create violations.
        So we just need to check: in S - {y} + {x}, is there any violation involving x?
        Also, we need to make sure S - {y} + {x} doesn't have violations among its other elements,
        but those were already valid (since S was valid, removing y keeps it valid, and adding x 
        only creates violations involving x).
        
        Actually that's not quite right. S might not be valid at the start of Phase 2.
        Let me just do a full check but optimized.
        """
        S_new = [z for z in S if z != y]
        S_new.append(x)
        S_new_set = set(S_new)
        
        # Check violations involving x
        for b in S_new:
            if b == x:
                continue
            # Check if x divides b*c for some c in S_new, c != x, c != b
            for c in S_new:
                if c == x or c == b:
                    continue
                if (b * c) % x == 0:
                    return False
            # Check if b divides x*c for some c in S_new, c != b, c != x
            g = math.gcd(b, x)
            b_prime = b // g
            if b_prime != 1:
                for c in S_new:
                    if c == b or c == x:
                        continue
                    if c % b_prime == 0:
                        return False
            # Check if c divides b*x for some... already covered by symmetry above
            # Actually the condition is: for every triple (a,b,c) distinct, a ∤ b*c
            # So we need to check all ordered triples. Let me just check all pairs with x.
        
        # Check violations NOT involving x (these should be same as before removing y, 
        # but since we removed y, some violations might have been resolved)
        # We need to verify the remaining set is still valid.
        # This is expensive. Let's just do full validation for safety.
        
        return is_valid_fast(S_new)
    
    def check_swap_validity_fast(S, S_set, x, y):
        """
        Fast check: assume S is valid. After removing y, S\{y} is still valid.
        We just need to check if adding x to S\{y} creates any violation.
        A violation involving x: for distinct a,b in S\{y}, x ∤ a*b, a ∤ x*b, b ∤ x*a.
        """
        S_without_y = [z for z in S if z != y]
        
        # Check: does x divide a*b for any distinct a,b in S_without_y?
        for i, a in enumerate(S_without_y):
            for b in S_without_y[i+1:]:
                if (a * b) % x == 0:
                    return False
        
        # Check: does any a in S_without_y divide x*b for some b in S_without_y, b != a?
        for a in S_without_y:
            for b in S_without_y:
                if a == b:
                    continue
                g = math.gcd(a, x)
                a_prime = a // g
                if a_prime == 1:
                    continue
                if b % a_prime == 0:
                    return False
        
        # Check: does any b in S_without_y divide a*x for some a in S_without_y, a != b?
        # This is the same as the previous check with roles swapped, already covered.
        
        return True
    
    # Phase 1: Greedy removal
    while time.time() < deadline:
        if len(S) <= 2:
            break
        conflict_count = count_violations_fast(S)
        
        max_conflicts = max(conflict_count.values()) if conflict_count else 0
        
        if max_conflicts == 0:
            break
        
        # Remove the element with the most conflicts
        worst_element = None
        for x in S:
            if conflict_count[x] == max_conflicts:
                worst_element = x
                break
        
        if worst_element is not None:
            S.remove(worst_element)
            S_set.remove(worst_element)
        else:
            break
    
    # Phase 2: Local search with swaps using fast validation
    removed = [x for x in range(129, 257) if x not in S_set]
    random.shuffle(removed)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for x in removed[:]:
            if x in S_set:
                continue
            if time.time() > deadline:
                break
            
            # Try to add x back by removing one element from S
            for y in S[:]:
                if time.time() > deadline:
                    break
                
                # Fast check: assume S is valid, check if S\{y}+{x} is valid
                if check_swap_validity_fast(S, S_set, x, y):
                    # Accept the swap
                    S.remove(y)
                    S_set.remove(y)
                    S.append(x)
                    S_set.add(x)
                    removed.remove(x)
                    removed.append(y)
                    random.shuffle(removed)
                    improved = True
                    break
    
    # Final verification
    if not is_valid_fast(S):
        # If somehow invalid, fall back to greedy removal again
        while not is_valid_fast(S) and time.time() < deadline:
            if len(S) <= 2:
                break
            conflict_count = count_violations_fast(S)
            max_conflicts = max(conflict_count.values()) if conflict_count else 0
            if max_conflicts == 0:
                break
            worst_element = None
            for x in S:
                if conflict_count[x] == max_conflicts:
                    worst_element = x
                    break
            if worst_element is not None:
                S.remove(worst_element)
                S_set.remove(worst_element)
            else:
                break
    
    return S
# EVOLVE-BLOCK-END
