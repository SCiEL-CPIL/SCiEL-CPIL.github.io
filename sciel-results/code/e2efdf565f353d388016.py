import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def full_check(S):
        """Check if set S is valid."""
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                for k in range(j, m):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if adding x to set S keeps it valid."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: x is the smallest (a)
        # a = x, b, c in S with x < b <= c
        # x must not divide b + c
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False

        # Case 2: x is the middle (b)
        # a, c in S with a < x <= c
        # a must not divide x + c
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Case 3: x is the largest (c)
        # a, b in S with a < b <= x
        # a must not divide b + x
        for a in S_list:
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False

        return True

    # Initial greedy solution
    def greedy_descending():
        S = set()
        for x in range(n, 0, -1):
            if can_add(S, x):
                S.add(x)
        return S

    best_set = greedy_descending()
    best_size = len(best_set)
    
    # Local search
    all_numbers = set(range(1, n + 1))
    
    # Precompute conflicts for faster checking? 
    # can_add is O(N^2) worst case. N=256, so ~65k ops. 
    # 80 seconds / 65k ops is plenty for thousands of iterations.
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
            
        S = set(best_set)
        S_list = list(S)
        random.shuffle(S_list)
        
        # Remove 1-2 random elements
        num_remove = random.randint(1, 2)
        for i in range(min(num_remove, len(S_list))):
            S.remove(S_list[i])
            
        # Try to fill with candidates
        # Sort candidates by some heuristic? 
        # Let's try adding in descending order, which tends to work well.
        candidates = [x for x in range(n, 0, -1) if x not in S]
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Optional: Swap local search
        # Pick one from S, one not in S, swap if valid
        if random.random() < 0.5 and len(S) > 0:
            S_swap = set(best_set)
            u = random.choice(list(S_swap))
            S_swap.remove(u)
            
            # Try to add a random v
            available = [x for x in range(1, n + 1) if x not in S_swap]
            if available:
                v = random.choice(available)
                if can_add(S_swap, v):
                    S_swap.add(v)
                    if len(S_swap) > best_size:
                        best_size = len(S_swap)
                        best_set = set(S_swap)

    # Final validation
    if full_check(best_set) and len(best_set) > 0:
        return list(best_set)
    return list(range(171, 257)) # Fallback

