"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    # Helper to check validity and update counts efficiently?
    # For N=1000, |S| is small (~60). Recalculating counts from scratch is O(|S|^2) ~ 3600 ops.
    # This is very fast. We can just recalculate counts for simplicity and safety.
    
    def is_valid(S):
        if not S:
            return True
        S_list = sorted(S)
        n = len(S_list)
        counts = {}
        for i in range(n):
            a = S_list[i]
            # a + a
            s = 2 * a
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 2:
                return False
            # a + b for b > a
            for j in range(i + 1, n):
                b = S_list[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True

    def greedy_random(seed):
        random.seed(seed)
        S = set()
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        for x in candidates:
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
        return S

    def greedy_sorted(reverse=False):
        S = set()
        candidates = list(range(1, 1001))
        if reverse:
            candidates.reverse()
        for x in candidates:
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
        return S

    # 1. Initialize with best of several heuristics
    best_S = set()
    
    # Deterministic Greedy (Smallest first)
    S1 = greedy_sorted(reverse=False)
    if len(S1) > len(best_S):
        best_S = S1
        
    # Deterministic Greedy (Largest first)
    S2 = greedy_sorted(reverse=True)
    if len(S2) > len(best_S):
        best_S = S2

    # Random Greedy (Multiple seeds)
    for seed in range(20):
        S3 = greedy_random(seed)
        if len(S3) > len(best_S):
            best_S = S3

    # 2. Local Search: Swap-Add
    # We try to find a move (remove x, add y) that allows adding z.
    # This increases size by 1.
    
    current_S = set(best_S)
    max_size = len(current_S)
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    while time.time() - start_time < time_limit:
        improved = False
        S_list = list(current_S)
        random.shuffle(S_list)
        
        for x in S_list:
            if time.time() - start_time > time_limit:
                break
            
            # Try swapping x with some y not in S
            # To speed up, we don't try ALL y, but a random subset or all y > x?
            # Let's try a random sample of candidates first, then if stuck, try all?
            # Given N=1000, trying all y (900+) for each x (60) is ~54000 checks.
            # Each check is is_valid (O(3600)). Total ~200M ops. Too slow in Python.
            
            # Optimization:
            # We can maintain counts for current_S.
            # But is_valid from scratch is simpler.
            # Let's limit the search for y.
            
            # Strategy: Try random y's first.
            candidates_y = [y for y in range(1, 1001) if y not in current_S]
            random.shuffle(candidates_y)
            
            # Try only a subset of y to keep it fast, e.g., 50 randoms
            for y in candidates_y[:50]:
                new_S = (current_S - {x}) | {y}
                if is_valid(new_S):
                    # Now try to add z
                    # Try to add the smallest possible z? Or random?
                    # Smallest z is often good for packing.
                    # Let's try to add any z.
                    # To be fast, we can iterate z from 1 to 1000 and check.
                    # But that's 1000 * is_valid.
                    
                    # Alternative: Just check if we can add ANY z.
                    # We can iterate z in random order or sorted order.
                    # Let's try sorted order (smallest first) to find a valid z quickly if one exists.
                    # Or, we can just pick a random z and check.
                    
                    # Let's try to find ANY z that makes it valid.
                    # We'll iterate through a random sample of z's.
                    candidates_z = [z for z in range(1, 1001) if z not in new_S]
                    random.shuffle(candidates_z)
                    
                    found_z = None
                    for z in candidates_z[:20]:
                        final_S = new_S | {z}
                        if is_valid(final_S):
                            found_z = z
                            break
                    
                    if found_z is not None:
                        current_S = final_S
                        max_size = len(current_S)
                        improved = True
                        break # Restart outer loop with new S
            
            if improved:
                break
        
        if not improved:
            # If we didn't improve in this pass, we might be stuck in a local optimum.
            # We can try a "kick": random swap that keeps size same but changes structure,
            # or just break and try a different initial seed?
            # Given time constraints, breaking is safer to return a valid large set.
            # But we can try one more level of randomization: 
            # Randomly remove 2 and add 2? No, too complex.
            
            # Let's just break if we don't improve after a full pass.
            # The "random y" and "random z" sampling provides stochasticity.
            # If we run this loop multiple times, we might escape.
            # But the loop condition is time-based.
            # If we break, we stop.
            # To allow escaping, we should NOT break immediately if no improvement, 
            # but the "improved" flag resets each while iteration.
            # Wait, if `improved` is False, the while loop continues.
            # It will shuffle S_list again and try again.
            # This is effectively Iterated Local Search with random restarts of the neighbor selection.
            pass

    return list(current_S)
# EVOLVE-BLOCK-END
