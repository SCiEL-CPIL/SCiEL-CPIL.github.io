import time
import random

def construct():
    n = 256
    # Greedy initialization
    A = set()
    for x in range(1, n + 1):
        valid = True
        for d in range(1, (x - 1) // 3 + 1):
            if all((x - j * d) in A for j in (1, 2, 3)):
                valid = False
                break
        if valid:
            A.add(x)
    
    # Helper to count 3-APs involving x in set S
    def count_3aps_with_x(x, S):
        count = 0
        # x is middle: x-d, x, x+d
        for d in range(1, n + 1):
            if (x - d) in S and (x + d) in S:
                count += 1
        # x is first: x, x+d, x+2d
        for d in range(1, n // 2 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                count += 1
        # x is last: x-2d, x-d, x
        for d in range(1, n // 2 + 1):
            if (x - 2 * d) in S and (x - d) in S:
                count += 1
        return count

    # Helper to check if adding y to S creates a 4-AP
    def creates_4ap(y, S):
        # y is 1st: y, y+d, y+2d, y+3d in S U {y} => y+d, y+2d, y+3d in S
        for d in range(1, n // 3 + 1):
            if (y + d) in S and (y + 2 * d) in S and (y + 3 * d) in S:
                return True
        # y is 2nd: y-d, y, y+d, y+2d => y-d, y+d, y+2d in S
        for d in range(1, n // 3 + 1):
            if (y - d) in S and (y + d) in S and (y + 2 * d) in S:
                return True
        # y is 3rd: y-2d, y-d, y, y+d => y-2d, y-d, y+d in S
        for d in range(1, n // 3 + 1):
            if (y - 2 * d) in S and (y - d) in S and (y + d) in S:
                return True
        # y is 4th: y-3d, y-2d, y-d, y => y-3d, y-2d, y-d in S
        for d in range(1, n // 3 + 1):
            if (y - 3 * d) in S and (y - 2 * d) in S and (y - d) in S:
                return True
        return False

    def total_3aps(S):
        count = 0
        # d from 1 to 128
        for d in range(1, n // 2 + 1):
            for a in range(1, n - 2 * d + 1):
                if a in S and (a + d) in S and (a + 2 * d) in S:
                    count += 1
        return count

    # Initial score
    current_score = total_3aps(A)
    
    # Precompute contributions for current A
    contrib = {x: count_3aps_with_x(x, A) for x in A}
    
    deadline = time.time() + 55  # 55s limit
    
    while time.time() < deadline:
        best_gain = 0
        best_x = None
        best_y = None
        
        # Try swaps
        # To speed up, we can sample or just iterate all if fast enough
        # |A| ~ 100, |Not A| ~ 150. 15000 pairs.
        # Each pair check: creates_4ap(y, A\{x}) is O(N).
        # 15000 * 256 = 3.8M ops. Might be too slow in Python per iteration.
        # Let's optimize creates_4ap or limit search.
        
        # Alternative: Only try swaps with y that are "close" to x or random samples.
        # Or just try Add and Remove first.
        
        # Try Adds
        for y in range(1, n + 1):
            if y not in A:
                if not creates_4ap(y, A):
                    gain = count_3aps_with_x(y, A)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = None # None means add
                        best_y = y
        
        # Try Removes
        for x in A:
            gain = -contrib[x]
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_y = None # None means remove
        
        # Try Swaps (Sampled)
        # If we have time, do a few random swaps
        if time.time() < deadline - 1:
            for _ in range(100): # 100 random swaps
                x = random.choice(list(A))
                y = random.randint(1, n)
                if y in A: continue
                if y == x: continue
                
                # Check if y can be added to A\{x}
                # This is equivalent to: creates_4ap(y, A) is False OR (all 4-APs formed by y in A contain x)
                # Simpler: temporarily remove x, check y, restore x
                A.remove(x)
                valid = not creates_4ap(y, A)
                A.add(x)
                
                if valid:
                    # Calculate gain
                    # New score = current_score - contrib[x] + contrib_y_new
                    # contrib_y_new is count of 3-APs with y in A\{x}
                    # count_3aps_with_x(y, A) counts 3-APs with y in A.
                    # We must subtract 3-APs that contain both x and y.
                    cy = count_3aps_with_x(y, A)
                    # Check 3-APs containing x and y
                    shared = 0
                    # x, y, z => z = 2y-x
                    if 2*y - x in A and 2*y - x not in [x, y]: shared += 1
                    # y, x, z => z = 2x-y
                    if 2*x - y in A and 2*x - y not in [x, y]: shared += 1
                    # z, x, y => z = 2x-y (same as above)
                    # x, z, y => z = (x+y)/2
                    if (x + y) % 2 == 0:
                        z = (x + y) // 2
                        if z in A and z not in [x, y]: shared += 1
                    # y, z, x => same as above
                    # z, y, x => z = 2y-x (same as first)
                    
                    cy_new = cy - shared
                    gain = -contrib[x] + cy_new
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = y
        
        if best_gain > 0:
            if best_x is None: # Add
                A.add(best_y)
                current_score += best_gain
                contrib[best_y] = best_gain
                # Update contrib for others? 
                # Adding best_y creates new 3-APs with other elements.
                # We don't update contrib for all others, so we might miss future improvements.
                # For simplicity, we recompute contrib if we want accuracy, but it's expensive.
                # Or we just accept that contrib is stale and rely on total_3aps for final check?
                # No, we use contrib for gain calculation.
                # Let's just recompute contrib for all x in A if we make a move? 
                # That's O(|A| * N) = 100 * 256 = 25k ops. Acceptable.
                contrib = {x: count_3aps_with_x(x, A) for x in A}
                
            elif best_y is None: # Remove
                A.remove(best_x)
                current_score += best_gain
                del contrib[best_x]
                # Update contrib for others?
                contrib = {x: count_3aps_with_x(x, A) for x in A}
                
            else: # Swap
                A.remove(best_x)
                A.add(best_y)
                current_score += best_gain
                contrib = {x: count_3aps_with_x(x, A) for x in A}
                
        else:
            # Stuck. Try a random restart or break.
            # Let's break to avoid wasting time.
            break
            
    return sorted(A)
