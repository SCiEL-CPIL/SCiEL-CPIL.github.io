import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    def descending_greedy():
        S = set()
        pair_sums = set()
        
        # Process from largest to smallest
        for a in range(n, 0, -1):
            # Check if a can be added
            # Condition: no sum in pair_sums is divisible by a
            # Note: pair_sums contains sums b+c for b,c in S with b<=c.
            # Since S contains only elements > a, any b,c in S satisfy a < b <= c.
            # So we just check if any s in pair_sums is divisible by a.
            
            valid = True
            for s in pair_sums:
                if s % a == 0:
                    valid = False
                    break
            
            if valid:
                S.add(a)
                # Update pair_sums
                # New pairs: (a, a) and (a, b) for b in S (excluding a? No, S currently has elements > a)
                # Wait, S currently contains elements > a.
                # New elements in S will be S_old U {a}.
                # New pair sums to add:
                # 1. a + a = 2a
                # 2. a + b for all b in S_old
                pair_sums.add(2 * a)
                for b in S:
                    pair_sums.add(a + b)
                    
        return S

    # Run the greedy algorithm
    best_set = descending_greedy()
    best_size = len(best_set)
    
    # Try to improve with local search or randomized restarts
    # Since the greedy is deterministic, let's try a few randomized variants
    # Variant: Randomly skip some numbers in the descending order?
    # Or: Randomized order of checking? No, descending is required for the fast check.
    
    # Let's try a "Soft" Greedy: 
    # If adding 'a' is invalid, maybe we should have skipped some larger numbers?
    # This suggests a backtracking or beam search.
    
    # For now, let's try to optimize the local search.
    # Try to remove an element and re-greedy?
    
    # A simple improvement: 
    # The greedy might get stuck. 
    # Let's try running the greedy on a subset of starting points?
    # Actually, the descending greedy is quite strong.
    
    # Let's check the time and if we have more, do a local search.
    if time.time() < deadline - 10:
        # Local Search: Randomly remove an element, then greedily add back
        # This is expensive. 
        # Instead, let's try to "shuffle" the decision at the end?
        
        # Alternative: 
        # Use the result of descending greedy as a seed for a random local search.
        # But the state space is large.
        
        # Let's just verify the set and return it.
        # If we have time, we can try a few random restarts of a *different* heuristic.
        
        # Heuristic 2: Randomized Ascending Greedy
        # This is slower but might find different solutions.
        
        def can_add_slow(S, a):
            S_list = sorted(S)
            larger = [x for x in S_list if x > a]
            for i, b in enumerate(larger):
                for c in larger[i:]:
                    if (b + c) % a == 0:
                        return False
            return True

        def randomized_ascending_greedy():
            S = set()
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            for a in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add_slow(S, a):
                    S.add(a)
            return S

        # Try a few randomized ascending runs
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            S_rand = randomized_ascending_greedy()
            if len(S_rand) > best_size:
                best_set = S_rand
                best_size = len(S_rand)
                
        # Also try randomized descending? 
        # The fast check relies on descending order. 
        # If we shuffle, we lose the fast check.
        
        # Let's try a "Hybrid": 
        # Start with descending greedy result.
        # Try to swap elements.
        
    # Final verification
    S_final = sorted(best_set)
    valid = True
    for i, a in enumerate(S_final):
        for j in range(i + 1, len(S_final)):
            b = S_final[j]
            for c in S_final[j:]:
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback
        return list(range(171, 257))

    return S_final
