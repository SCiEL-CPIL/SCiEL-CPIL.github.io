import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_diff_counts(S):
        """
        Calculate the count of 3-term APs for each difference d.
        Returns a dict: {d: count}
        """
        S_set = set(S)
        if not S_set:
            return {}
        max_val = max(S_set)
        min_val = min(S_set)
        max_d = (max_val - min_val) // 2
        counts = {}
        for d in range(1, max_d + 1):
            count = 0
            for a in S_set:
                if a + d in S_set and a + 2 * d in S_set:
                    count += 1
            if count > 0:
                counts[d] = count
        return counts

    def calc_score(counts):
        return len(counts)
        
    def try_improve(S, counts, score):
        """
        Attempts to improve the set S by performing random swaps.
        Returns (new_S, new_counts, new_score) if an improvement is found, else None.
        """
        # Try a few random swaps
        # Using a small number of attempts per call keeps the function fast
        # and allows for frequent time checks in the outer loop.
        for _ in range(10): 
            if time.time() > deadline:
                return None
                
            # Pick a random element to remove
            x = random.choice(list(S))
            
            # Pick a random element to add
            y = random.randint(1, 512)
            if y in S:
                continue
                
            # Simulate the swap
            S_new = S.copy()
            S_new.remove(x)
            S_new.add(y)
            
            # Recalculate score
            # This is O(32 * 256) ~ 8000 ops, which is fast enough for ~1000-5000 iterations
            new_counts = get_diff_counts(S_new)
            new_score = calc_score(new_counts)
            
            if new_score > score:
                return S_new, new_counts, new_score
        return None

    best_score = -1
    best_set = None
    
    # Define several initializations to explore different parts of the solution space
    inits = [
        set(range(1, 33)),  # Dense, small d
        set(random.sample(range(1, 513), 32)), # Sparse, large d
        set(random.sample(range(1, 513), 32)),
        set(random.sample(range(1, 513), 32)),
        set([1 + i * 16 for i in range(32)]), # Structured spread
        set([1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204, 205, 206, 207, 300, 301, 302, 303, 304, 305, 306, 307]) # Multi-block
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        S = set(init)
        counts = get_diff_counts(S)
        score = calc_score(counts)
        
        if score > best_score:
            best_score = score
            best_set = set(S)
            
        no_improve = 0
        # Local search loop
        while time.time() < deadline:
            res = try_improve(S, counts, score)
            if res:
                S, counts, score = res
                no_improve = 0
                if score > best_score:
                    best_score = score
                    best_set = set(S)
            else:
                no_improve += 1
                
                # If stuck, perform a random perturbation (kick) to escape local maximum
                if no_improve > 50:
                    # Randomly swap 1-2 elements
                    for _ in range(2):
                        x = random.choice(list(S))
                        y = random.randint(1, 512)
                        if y not in S:
                            S.remove(x)
                            S.add(y)
                            break
                    
                    # Recalculate after kick
                    counts = get_diff_counts(S)
                    score = calc_score(counts)
                    no_improve = 0
                    
                    if score > best_score:
                        best_score = score
                        best_set = set(S)
                
                # If we've kicked and still not improving for a long time, restart this init
                if no_improve > 200:
                    break

    return list(best_set)
