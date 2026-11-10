import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Helper functions
    def has_4ap_with_add(A, x):
        # Check if adding x to A creates a 4-AP
        # x is not in A
        # Cases:
        # 1. x, x+d, x+2d, x+3d
        d = 1
        while x + 3*d <= 256:
            if (x+d in A) and (x+2*d in A) and (x+3*d in A):
                return True
            d += 1
        # 2. x-d, x, x+d, x+2d
        d = 1
        while x - d >= 1 and x + 2*d <= 256:
            if (x-d in A) and (x+d in A) and (x+2*d in A):
                return True
            d += 1
        # 3. x-2d, x-d, x, x+d
        d = 1
        while x - 2*d >= 1 and x + d <= 256:
            if (x-2*d in A) and (x-d in A) and (x+d in A):
                return True
            d += 1
        # 4. x-3d, x-2d, x-d, x
        d = 1
        while x - 3*d >= 1:
            if (x-3*d in A) and (x-2*d in A) and (x-d in A):
                return True
            d += 1
        return False

    def count_3aps_gain(A, x):
        # Count 3-APs involving x that would be formed by adding x to A
        count = 0
        # 1. x is middle: a, x, b -> a+b=2x
        for a in A:
            if a < x:
                b = 2*x - a
                if b > x and b in A:
                    count += 1
        # 2. x is start: x, x+d, x+2d
        d = 1
        while x + 2*d <= 256:
            if (x+d in A) and (x+2*d in A):
                count += 1
            d += 1
        # 3. x is end: x-2d, x-d, x
        d = 1
        while x - 2*d >= 1:
            if (x-2*d in A) and (x-d in A):
                count += 1
            d += 1
        return count

    def total_3aps(A):
        # Calculate total 3-APs in A
        count = 0
        # a, b, c with b-a = c-b => 2b = a+c
        # Iterate b
        for b in A:
            # a < b < c
            # a = 2b - c
            # c > b
            # a >= 1 => 2b - c >= 1 => c <= 2b - 1
            # c <= 256
            min_c = b + 1
            max_c = min(256, 2*b - 1)
            for c in range(min_c, max_c + 1):
                a = 2*b - c
                if a in A:
                    count += 1
        return count

    # Initial construction: Greedy
    A = set()
    for x in range(1, 257):
        if not has_4ap_with_add(A, x):
            A.add(x)
    
    best_score = total_3aps(A)
    best_set = set(A)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_x = None
        
        # Try adding elements
        for x in range(1, 257):
            if x in A:
                continue
            if not has_4ap_with_add(A, x):
                gain = count_3aps_gain(A, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
        
        if best_x is not None:
            A.add(best_x)
            improved = True
            # Update best if better
            current_score = total_3aps(A)
            if current_score > best_score:
                best_score = current_score
                best_set = set(A)
        else:
            # No single addition improves. Try swaps?
            # Or break to restart?
            # Let's try a limited number of swaps
            swap_improved = False
            # Sample some swaps to save time
            # Iterate y in A, x not in A
            # This is O(N^2) checks.
            # We can limit this.
            if time.time() < deadline - 5.0:
                for y in list(A):
                    if time.time() > deadline - 5.0: break
                    for x in range(1, 257):
                        if x in A: continue
                        if not has_4ap_with_add(A - {y}, x):
                            # Calculate gain
                            # Gain = gain(x in A\{y}) - loss(y in A)
                            # This is complex.
                            # Simpler: Calculate total_3aps of new set?
                            # Too slow.
                            # Let's approximate or skip swaps for now.
                            pass
            # If no swap implemented, just break
            break

    return sorted(best_set)
