"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    N = 128
    
    def time_left():
        return deadline - time.time()
    
    # Bitmask-based validity check for adding x to a set represented by mask
    def can_add(mask, x):
        """Check if x can be added to set (bitmask) without creating forbidden triple."""
        # New forbidden triples must involve x.
        # Case 1: x is the largest element in triple (a, b, x) with a < b < x
        #   Need: a+b in S∪{x}, a+x in S∪{x}, b+x in S∪{x}
        #   Since a+x > x and b+x > x, a+x and b+x must be in S (not x itself unless sum==x, impossible since a,b>=1)
        #   a+b could be in S or equal to x
        max_a = min(x - 2, N - x)  # a < b < x, so a <= x-3, and a+x <= N so a <= N-x
        if max_a >= 1:
            for a in range(1, max_a + 1):
                if not ((mask >> a) & 1):
                    continue
                xa = x + a
                if xa > N or not ((mask >> xa) & 1):
                    continue
                # Now need b: a < b < x, b in S, b+x in S, and a+b in S or a+b == x
                for b in range(a + 1, min(x, N - x) + 1):
                    if not ((mask >> b) & 1):
                        continue
                    xb = x + b
                    if xb > N or not ((mask >> xb) & 1):
                        continue
                    ab = a + b
                    if ab == x:
                        return False
                    if ab <= N and ((mask >> ab) & 1):
                        return False
        
        # Case 2: x is the middle element (a, x, c) with a < x < c
        #   Need: a+x in S∪{x}, x+c in S∪{x}, a+c in S∪{x}
        #   a+x > x so must be in S. x+c > x so must be in S. a+c could be in S or... a+c < x+c and a+c > a+x
        #   Actually a+c could be anything. But a+c <= (x-1)+(N) which could exceed N.
        #   a+x in S: a <= N-x
        #   x+c in S: c <= N-x
        #   a+c in S or a+c... just check if in mask
        max_a2 = N - x  # a + x <= N
        max_c2 = N - x  # x + c <= N
        if max_a2 >= 1 and max_c2 >= x + 1:
            for a in range(1, max_a2 + 1):
                if not ((mask >> a) & 1):
                    continue
                ax = a + x
                if ax > N or not ((mask >> ax) & 1):
                    continue
                for c in range(x + 1, max_c2 + 1):
                    if not ((mask >> c) & 1):
                        continue
                    xc = x + c
                    if xc > N or not ((mask >> xc) & 1):
                        continue
                    ac = a + c
                    if ac == x:
                        return False
                    if ac <= N and ((mask >> ac) & 1):
                        return False
        
        # Case 3: x is the smallest element (x, b, c) with x < b < c
        #   Need: x+b in S∪{x}, x+c in S∪{x}, b+c in S∪{x}
        #   x+b > x so in S. x+c > x so in S. b+c > x so in S (or... b+c could equal x? No, b,c > x >= 1, so b+c > x)
        #   So all three sums must be in S.
        max_b = N - x  # x + b <= N
        max_c = N - x  # x + c <= N
        if max_b >= x + 1 and max_c >= x + 2:
            for b in range(x + 1, max_b + 1):
                if not ((mask >> b) & 1):
                    continue
                xb = x + b
                if xb > N or not ((mask >> xb) & 1):
                    continue
                for c in range(b + 1, max_c + 1):
                    if not ((mask >> c) & 1):
                        continue
                    xc = x + c
                    if xc > N or not ((mask >> xc) & 1):
                        continue
                    bc = b + c
                    if bc == x:
                        return False
                    if bc <= N and ((mask >> bc) & 1):
                        return False
        
        # Case 4: x completes a sum. A triple (a,b,c) in S where one of a+b, a+c, b+c equals x.
        #   Say a+b = x, and a+c, b+c in S.
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a or b > N:
                continue
            if not ((mask >> a) & 1) or not ((mask >> b) & 1):
                continue
            # Need c != a, c != b, c in S, a+c in S, b+c in S
            max_c = N - max(a, b)
            for c in range(1, max_c + 1):
                if c == a or c == b:
                    continue
                if not ((mask >> c) & 1):
                    continue
                ac = a + c
                bc = b + c
                if ac > N or bc > N:
                    continue
                if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                    return False
        
        return True
    
    def is_valid(mask):
        """Full validity check."""
        elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
        k = len(elems)
        for i in range(k):
            a = elems[i]
            for j in range(i + 1, k):
                b = elems[j]
                ab = a + b
                if ab > N:
                    continue
                if not ((mask >> ab) & 1):
                    continue
                for k_idx in range(j + 1, k):
                    c = elems[k_idx]
                    ac = a + c
                    bc = b + c
                    if ac > N or bc > N:
                        continue
                    if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                        return False
        return True
    
    def popcount(mask):
        return bin(mask).count('1')
    
    # Start with upper block {65..128} which is trivially valid (all pair sums > 128)
    best_mask = 0
    for i in range(65, 129):
        best_mask |= (1 << i)
    best_size = 64
    
    # Greedy: try adding elements from 1 to 64
    for x in range(1, 65):
        if time_left() < 5:
            break
        if can_add(best_mask, x):
            best_mask |= (1 << x)
            best_size += 1
    
    # Local search: try swaps to improve
    rng = random.Random(42)
    
    # Try to add more elements after initial greedy
    for _ in range(20):
        if time_left() < 5:
            break
        improved = False
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Swap-based local search
    for iteration in range(50000):
        if time_left() < 5:
            break
        
        # Try removing one element and adding one or more
        elems_in = [i for i in range(1, 129) if (best_mask >> i) & 1]
        elems_out = [i for i in range(1, 129) if not ((best_mask >> i) & 1)]
        
        if not elems_out:
            break
        
        # Try single swap
        r = rng.choice(elems_in)
        a = rng.choice(elems_out)
        
        temp_mask = best_mask & ~(1 << r)
        if can_add(temp_mask, a):
            new_mask = temp_mask | (1 << a)
            # Try to add more elements after swap
            for x in range(1, 129):
                if (new_mask >> x) & 1:
                    continue
                if can_add(new_mask, x):
                    new_mask |= (1 << x)
            
            if popcount(new_mask) > best_size:
                best_mask = new_mask
                best_size = popcount(best_mask)
    
    # Final greedy pass
    for x in range(1, 129):
        if time_left() < 2:
            break
        if (best_mask >> x) & 1:
            continue
        if can_add(best_mask, x):
            best_mask |= (1 << x)
            best_size += 1
    
    # Convert to list
    result = [int(i) for i in range(1, 129) if (best_mask >> i) & 1]
    
    return result
# EVOLVE-BLOCK-END
