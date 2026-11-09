import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 180
    # Precompute reciprocal constraints: 
    # 1/a = 1/b + 1/c  =>  bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    # So for distinct a,b,c in set, if (b-a)(c-a) == a^2, it's a conflict.
    # We can precompute for each a, the pairs (b,c) that form a conflict.
    # But for SA, we need to check if adding/removing x creates conflicts.
    
    # A faster check: 
    # For a fixed set S, a conflict exists if there are distinct a,b,c in S such that (b-a)(c-a)=a^2.
    # Let's precompute all conflict triples (a,b,c) with a<b, a<c, b!=c, a,b,c in [1,180].
    # Actually, we can just check validity of the whole set quickly.
    
    def is_valid(s_set):
        # s_set is a set of ints
        # Check all triples? Too slow if done naively.
        # Better: For each a in s_set, check if there exist b,c in s_set 
        # such that (b-a)(c-a) = a^2.
        # Let d1 = b-a, d2 = c-a. Then d1*d2 = a^2.
        # d1, d2 must be divisors of a^2.
        # Also b = a+d1, c = a+d2 must be in s_set.
        # And b != c => d1 != d2.
        # Since b,c > a (assuming wlog a is the smallest? No, a is the 'sum' side, so 1/a is larger than 1/b, 1/c, so a < b and a < c).
        # So we only need to look at b,c > a.
        
        for a in s_set:
            a2 = a * a
            # Find divisors of a^2
            # b = a + d, c = a + a^2/d
            # We need b, c in s_set, b != c.
            # Iterate d such that a^2 % d == 0.
            # d must be such that b = a+d <= 180 and c = a+a^2/d <= 180.
            # Also d != a^2/d (to ensure b != c).
            
            # Optimization: precompute divisors for each a? Or just check on the fly.
            # Since n=180, a^2 is at most 32400. Divisors are few.
            for d in range(1, int(math.isqrt(a2)) + 1):
                if a2 % d != 0:
                    continue
                d1 = d
                d2 = a2 // d
                if d1 == d2:
                    continue
                b = a + d1
                c = a + d2
                if b > 180 or c > 180:
                    continue
                if b in s_set and c in s_set:
                    return False
        return True

    # Initial state: Random subset of size ~100?
    # Or start with a known good seed and perturb?
    # Let's start with a random set of size 100.
    random.seed(12345)
    current_set = set(random.sample(range(1, n+1), 100))
    if not is_valid(current_set):
        # If invalid, fix it by removing elements involved in conflicts
        # Or just start with a smaller set or the high-end set
        # Let's use the high-end set as a safe start
        current_set = set(range(91, 181))
        
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Simulated Annealing
    # State: set of elements
    # Move: add a random element from [1,180]\S, or remove a random element from S
    # Cost: -size (we want to maximize size, so cost = -size)
    # But we must maintain validity. 
    # Strategy: Only accept moves that keep the set valid.
    # If we only accept valid moves, it's just random walk on the space of valid sets.
    # To escape local optima, we need to allow invalid states? 
    # No, the problem requires the final set to be valid.
    # If we allow invalid intermediate states, we need a penalty.
    # Let's try: Cost = -size + penalty * (number of conflicts).
    # This is complex. 
    # Alternative: Random walk on valid sets. 
    # If stuck, try a "kick" (random restart or larger perturbation).
    
    # Let's try a different approach: 
    # Local Search with Random Restarts.
    # 1. Start with a random valid set.
    # 2. Try to add elements. If stuck, remove a random element and try again.
    
    def random_valid_set(target_size):
        # Greedy random construction
        s = set()
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for c in candidates:
            if time.time() > deadline:
                break
            if len(s) >= target_size:
                break
            if is_valid(s | {c}):
                s.add(c)
        return s

    # Run multiple random restarts with local improvement
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with a random valid set of moderate size
        current_set = random_valid_set(80)
        
        # Local improvement: try to add elements
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            candidates = list(set(range(1, n+1)) - current_set)
            random.shuffle(candidates)
            for c in candidates:
                if time.time() > deadline:
                    break
                if is_valid(current_set | {c}):
                    current_set.add(c)
                    improved = True
                    # Don't break, keep trying to add more
                    # But if we add one, the candidate list is slightly stale, but that's ok
                    # Actually, if we add c, we should continue checking others?
                    # Yes, but is_valid is expensive. 
                    # Let's break and restart the inner loop to be safe and simple
                    # Or just continue. 
                    pass
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(best_set)
            
        # Try to improve best_set directly
        # Random local search on best_set
        # Pick a random element to remove, then greedily add back as many as possible
        for _ in range(10):
            if time.time() > deadline:
                break
            if not best_set:
                break
            removed = random.choice(list(best_set))
            temp_set = set(best_set)
            temp_set.discard(removed)
            
            # Greedily add elements to temp_set
            candidates = list(set(range(1, n+1)) - temp_set)
            random.shuffle(candidates)
            for c in candidates:
                if time.time() > deadline:
                    break
                if is_valid(temp_set | {c}):
                    temp_set.add(c)
                    
            if len(temp_set) > best_size:
                best_set = temp_set
                best_size = len(best_set)
            else:
                # If not improved, maybe try removing a different element?
                # Or just accept the current best.
                pass

    # Final check and return
    # Ensure best_set is valid
    if not is_valid(best_set):
        # Fallback to a known good set
        best_set = set(range(91, 181))
        
    return list(best_set)

# EVOLVE-BLOCK-END
