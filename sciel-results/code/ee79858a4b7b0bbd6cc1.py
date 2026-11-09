import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # 1. Generate initial Stanley Sequence (Greedy)
    S = set()
    selected = []
    for x in range(1, 257):
        # Check if x forms a 3-AP with any two elements in selected
        # Since we add in increasing order, x is the largest element in any potential new AP.
        # So we check if there exist a, b in selected such that 2*b - a == x.
        # Equivalent to: does x/2 + a/2 = b? i.e. is there a pair (a, b) with b = (a+x)/2?
        # Simpler: for b in selected, check if a = 2*b - x is in selected.
        valid = True
        for b in selected:
            a = 2 * b - x
            if a in S:
                valid = False
                break
        if valid:
            S.add(x)
            selected.append(x)
    
    # 2. Local Search: Swap large elements for small ones
    # We want to maximize sum(1/x).
    # So we try to replace x in S with y not in S, where y < x.
    
    # Precompute reciprocals
    inv = [0.0] * 257
    for i in range(1, 257):
        inv[i] = 1.0 / i
    
    current_score = sum(inv[x] for x in S)
    
    # Helper to check if a number y can be added to a set represented by a boolean array 'in_set'
    # in_set[i] is True if i is in the set.
    # We check if y forms a 3-AP with any two elements in the set.
    def can_add(y, in_set):
        # y is the new element.
        # Check if y is the middle: a, y, c => c = 2y - a.
        # Check if y is the end: a, b, y => b = (a+y)/2.
        # Since we are adding to a set that is already AP-free, we only need to check APs involving y.
        
        # Case 1: y is the largest element (a, b, y).
        # a, b in set, b = (a+y)/2.
        # So a must have same parity as y.
        # a = y - 2d, b = y - d.
        # Check if (y-2d) and (y-d) are in set.
        d = 1
        while y - 2*d >= 1:
            if in_set[y - 2*d] and in_set[y - d]:
                return False
            d += 1
            
        # Case 2: y is the middle element (a, y, c).
        # a = y - d, c = y + d.
        # Check if (y-d) and (y+d) are in set.
        d = 1
        while y + d <= 256:
            if in_set[y - d] and in_set[y + d]:
                return False
            d += 1
            
        # Case 3: y is the smallest element (y, b, c).
        # b = y + d, c = y + 2d.
        # Check if (y+d) and (y+2d) are in set.
        d = 1
        while y + 2*d <= 256:
            if in_set[y + d] and in_set[y + 2*d]:
                return False
            d += 1
            
        return True

    # Convert S to boolean array for fast lookup
    in_set = [False] * 257
    for x in S:
        in_set[x] = True
        
    # List of elements in S, sorted descending (to try removing large ones first)
    # List of elements NOT in S, sorted ascending (to try adding small ones first)
    # We will iterate through potential swaps.
    
    # To make the search effective, we can use a priority queue or just simple loops.
    # Given N=256, simple loops are fine.
    
    # We perform multiple passes until no improvement or time limit.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all swaps (x in S, y not in S, y < x)
        # To be efficient, we can iterate y from 1 to 256, and for each y, try all x in S with x > y.
        for y in range(1, 257):
            if in_set[y]:
                continue
            
            # Try to add y
            # We need to remove some x from S.
            # Which x?
            # Ideally, we remove an x that "blocks" y, or just any x > y.
            # Actually, if we add y, we must remove SOME x to keep the set size constant?
            # No, the problem does NOT require fixed size.
            # "Maximize the exact sum of reciprocals".
            # If we can just ADD y without removing anything, we should do it!
            
            # Wait, the greedy algorithm already added all possible y in increasing order.
            # So any y not in S was rejected because it formed an AP with existing elements.
            # To add y, we MUST remove at least one element from S that is part of the conflicting APs.
            
            # So a move is: Remove x, Add y.
            # We want to maximize gain: inv[y] - inv[x].
            # So we prefer small y and large x.
            
            # Let's try to find if there exists an x in S such that (S - {x}) + {y} is valid.
            # We can iterate x in S (sorted descending) and check validity.
            
            # Optimization: If we remove x, the APs that were blocking y might be broken.
            
            # For each y not in S:
            #   For each x in S (sorted desc):
            #      if inv[y] - inv[x] <= 0: break (since x is getting smaller, gain decreases)
            #      Temporarily remove x, check if y can be added.
            #      If yes, accept swap, update S, in_set, score. improved=True. break.
            
            # To avoid O(N^3) complexity (256 * 256 * 256), we need to be careful.
            # But 256^3 is ~16 million. In Python, this might be slow.
            # We can limit the number of x we try.
            
            # Let's try a simpler heuristic:
            # For each y not in S:
            #   Identify the "blocking" elements.
            #   Actually, just try removing the largest x in S.
            #   If that works, great.
            #   If not, try next largest.
            
            # Let's implement a limited search.
            
            # Sort S descending
            S_list = sorted(S, reverse=True)
            
            # We only need to try x > y
            for x in S_list:
                if x <= y:
                    break
                # Potential gain
                gain = inv[y] - inv[x]
                if gain <= 0:
                    break
                
                # Check if (S - {x}) + {y} is valid
                # Remove x
                in_set[x] = False
                if can_add(y, in_set):
                    # Accept
                    in_set[y] = True
                    S.discard(x)
                    S.add(y)
                    current_score += gain
                    improved = True
                    # Update S_list? We can just rebuild it or break and restart outer loop
                    # Rebuilding is safer.
                    break
                else:
                    # Restore x
                    in_set[x] = True
            
            if improved:
                # Restart the y loop with new state?
                # Or just continue?
                # Continuing is fine, but S_list is stale.
                # We can just break out of y loop and restart while loop.
                break
                
    return list(S)
