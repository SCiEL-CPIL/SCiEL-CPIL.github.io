import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def check_valid(colors):
        """
        Check if the coloring is valid.
        colors is a list of length 128, indices 0..127 correspond to positions 1..128.
        """
        n = len(colors)
        # Check all 4-APs
        # i, i+d, i+2d, i+3d
        for i in range(n):
            # d such that i+3d < n
            max_d = (n - 1 - i) // 3
            for d in range(1, max_d + 1):
                c1 = colors[i]
                c2 = colors[i + d]
                c3 = colors[i + 2 * d]
                c4 = colors[i + 3 * d]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    def greedy_construct(K, seed):
        """
        Attempt to construct a valid coloring with K colors using randomized greedy.
        """
        random.seed(seed)
        n = 128
        colors = [None] * n
        
        for i in range(n):
            valid_colors = set(range(K))
            
            # Check constraints imposed by previous positions
            # APs ending at i: i, i-d, i-2d, i-3d
            # d >= 1, i-3d >= 0 => d <= i//3
            max_d = i // 3
            for d in range(1, max_d + 1):
                idx1 = i
                idx2 = i - d
                idx3 = i - 2 * d
                idx4 = i - 3 * d
                
                c2 = colors[idx2]
                c3 = colors[idx3]
                c4 = colors[idx4]
                
                # We are choosing c1 (for idx1)
                # Condition: len({c1, c2, c3, c4}) >= 3
                # If c2, c3, c4 use < 3 colors, we might restrict c1.
                
                # Case 1: c2, c3, c4 are all same (A, A, A).
                # Then {c1, A, A, A} must have size >= 3.
                # This is impossible with 1 new color. Size is 1 or 2.
                # So if c2==c3==c4, NO valid color exists for c1.
                if c2 == c3 and c3 == c4:
                    return None
                    
                # Case 2: c2, c3, c4 use exactly 2 colors (e.g., A, A, B).
                # Set S = {A, B}.
                # We need len({c1} U S) >= 3.
                # So c1 must be a color NOT in S.
                if c2 != c3 or c3 != c4: # At least 2 distinct
                    if c2 == c3 or c3 == c4 or c2 == c4:
                         # Two are same, one different? Or all distinct?
                         # If all distinct, S has size 3. Any c1 works.
                         # If 2 same, 1 diff, S has size 2. c1 must be new.
                         pass
                     
                    distinct_prev = {c2, c3, c4}
                    if len(distinct_prev) == 2:
                        # c1 must be outside distinct_prev
                        valid_colors = valid_colors - distinct_prev
                    # If len(distinct_prev) == 3, any c1 is fine.
                    # If len(distinct_prev) == 1, handled above (impossible).
            
            if not valid_colors:
                return None
                
            # Choose a color
            # To encourage diversity or reuse?
            # If we reuse, we might create the "3 same" trap later?
            # Actually, reusing is good to minimize colors, but we are fixing K.
            # Random choice from valid.
            c = random.choice(list(valid_colors))
            colors[i] = c
            
        return colors

    # Try K=3 first (unlikely to work for n=128, but fast to check)
    # Then K=4
    # Then K=5
    
    best_solution = None
    
    # K=3
    # Greedy with few restarts
    for seed in range(100):
        if time.time() > deadline:
            break
        sol = greedy_construct(3, seed)
        if sol is not None:
            if check_valid(sol):
                return sol
    # If K=3 fails, try K=4
    for seed in range(1000):
        if time.time() > deadline:
            break
        sol = greedy_construct(4, seed)
        if sol is not None:
            if check_valid(sol):
                return sol
                
    # If K=4 fails, try K=5
    for seed in range(1000):
        if time.time() > deadline:
            break
        sol = greedy_construct(5, seed)
        if sol is not None:
            if check_valid(sol):
                return sol

    # Fallback: Deterministic greedy with K=128 (always works, but bad score)
    # Or a smarter fallback.
    # Let's try a simple deterministic greedy with K=8
    for K in [8, 10, 12]:
        colors = [0] * 128
        success = True
        for i in range(128):
            valid = set(range(K))
            max_d = i // 3
            for d in range(1, max_d + 1):
                c2 = colors[i - d]
                c3 = colors[i - 2 * d]
                c4 = colors[i - 3 * d]
                if c2 == c3 == c4:
                    success = False
                    break
                distinct_prev = {c2, c3, c4}
                if len(distinct_prev) == 2:
                    valid -= distinct_prev
            if not valid:
                success = False
                break
            colors[i] = min(valid) # Deterministic choice
        if success and check_valid(colors):
            return colors

    # Absolute fallback
    return list(range(128))
