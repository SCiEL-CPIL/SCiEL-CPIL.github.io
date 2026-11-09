import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 192
    MAX_K = 3
    TARGET_MAX = 48
    
    # Precompute all triplets of indices or values?
    # We will compute sums on the fly for the current set S.
    
    def is_valid_add(x, S):
        """
        Check if x can be added to S.
        Condition 1: x does not divide sum of any 3 distinct elements in S.
        Condition 2: For every y in S, y does not divide x + a + b for any distinct a, b in S \ {y}.
        """
        # Condition 1: x divides sum of 3 from S
        if len(S) >= 3:
            # Optimization: precompute triple sums? No, S changes.
            # Just iterate.
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Condition 2: y in S divides x + a + b
        # For each y in S, check all pairs a, b in S\{y}
        # This is O(|S|^3)
        # Optimization: 
        # We can precompute pair sums of S?
        # Let's just do it directly, it's fast enough for |S|<=48.
        
        # To speed up, we can check:
        # For each y in S:
        #   Let S_y = S \ {y}
        #   We need to check if there exist a, b in S_y such that (x + a + b) % y == 0
        #   This is equivalent to: (a + b) % y == (-x) % y
        #   Let target = (-x) % y
        #   We need to know if any pair sum in S_y is congruent to target mod y.
        
        for y in S:
            target = (-x) % y
            if target == 0:
                # Need a + b == 0 mod y. Since a,b > 0, a+b >= 3.
                # If y is large, a+b < y is possible?
                # Just check pairs.
                pass
            
            # Iterate pairs in S excluding y
            # To avoid O(N^3) overhead in Python, we can be smart.
            # But N=48, N^3 = 110k. 110k * 48 (outer loop) = 5M ops per check?
            # No, the outer loop is over y (48). Inner is pairs (1k). Total 50k modulo ops.
            # This is fine.
            
            # However, we can optimize by precomputing pair sums of S once per check?
            # No, pairs depend on y.
            
            # Let's just use combinations for clarity and speed in C-level iteration if possible?
            # combinations(S, 2) includes y. We need to skip pairs with y.
            
            found_conflict = False
            for a, b in combinations(S, 2):
                if a == y or b == y:
                    continue
                if (x + a + b) % y == 0:
                    found_conflict = True
                    break
            if found_conflict:
                return False
        
        return True

    def greedy_construction(order):
        S = []
        for x in order:
            if is_valid_add(x, S):
                S.append(x)
        return S

    def local_search(S):
        """
        Try to improve S by adding elements or swapping.
        Returns improved S.
        """
        S = list(S)
        remaining = [x for x in range(1, N+1) if x not in S]
        
        # Phase 1: Try to add any remaining element
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            for x in remaining:
                if is_valid_add(x, S):
                    S.append(x)
                    remaining.remove(x)
                    changed = True
                    break # Restart loop as S changed
            if not changed:
                # If we didn't break, we iterated all. 
                # Wait, the loop above breaks on first success.
                # If no success, we exit while.
                pass
        
        # Phase 2: Try to swap elements to free up space for more
        # This is complex. Let's try a simpler "remove and refill" strategy.
        # Identify elements that block many candidates.
        
        return S

    best_S = []
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    # Large numbers first is usually good.
    # Random shuffles.
    
    for i in range(200): # Max restarts, limited by time
        if time.time() > deadline:
            break
        
        # Heuristic: Sort by size descending, with some randomness
        candidates = list(range(1, N+1))
        
        # Bias towards larger numbers
        # We can assign weights or just sort and then shuffle partially?
        # Let's try sorting descending first, then shuffling the first K?
        # Or just random shuffle but weighted?
        
        # Simple random shuffle
        random.shuffle(candidates)
        
        # Try a few different orderings in one restart?
        # No, just one greedy pass per restart.
        
        S = greedy_construction(candidates)
        
        # Local search: Try to add more
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            
            # If we reached the max possible, we can stop early?
            # Max is 48.
            if best_size >= 48:
                break

    # Final check: Is best_S valid?
    # The construction ensures validity.
    
    return best_S

# Test run
# print(construct())
