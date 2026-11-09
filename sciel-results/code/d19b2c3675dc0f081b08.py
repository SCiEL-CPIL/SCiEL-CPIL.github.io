import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    best_S = []
    best_size = 0
    
    def run_one_trial():
        S = []
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        
        def try_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    return False
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
            return True

        def try_remove(x):
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1

        # 1. Randomized Greedy Initialization
        # Shuffle the candidates
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if try_add(c):
                pass
            # Early exit if we have enough time? No, just finish greedy.
        
        current_size = len(S)
        
        # 2. Local Search
        # We try to add random elements, and if blocked, try to swap with blockers.
        # We also try random swaps of elements in S with elements not in S?
        # The current logic:
        # Pick random c not in S.
        # If addable, add it.
        # Else, find blockers, try to swap one blocker with c.
        
        while time.time() < deadline - 5: # Leave 5s for cleanup/return
            c = random.randint(1, max_val)
            if in_S[c]:
                continue
            
            # Try direct add
            if try_add(c):
                current_size += 1
                continue
            
            # If direct add failed, find blockers
            # Blockers are elements y in S such that counts[c+y] >= 2 or counts[2c] >= 2
            # Actually, if counts[2c] >= 2, the blockers are the pairs summing to 2c.
            # If counts[c+y] >= 2, the blockers are the pairs summing to c+y.
            
            # Efficient blocker finding:
            # We just need to find ONE blocker to swap.
            # Iterate through S, check if removing y allows adding c.
            # This is O(M^2) if done naively.
            
            # Heuristic: Pick a random y from S.
            # Try to swap y with c.
            # If successful, we maintain size.
            # This is a random walk on the state space of fixed size.
            
            # To improve, we can try multiple random swaps per step.
            
            # Let's try a few random swaps
            for _ in range(3):
                y = random.choice(S)
                # Try swap y with c
                try_remove(y)
                if try_add(c):
                    # Success, size maintained
                    break
                else:
                    # Restore y
                    try_add(y)
                    # c is not in S, so we just failed to swap
                    # We need to check if c was added? No, try_add failed.
                    # So state is restored.
                    
            # If we have time, we can also try to find a "better" blocker?
            # For now, random swap is a structural change from "deterministic blocker selection"
            # in the original code (which iterated all blockers and shuffled).
            # Actually, the original code found ALL blockers and shuffled them.
            # This is more thorough than picking 3 random ones.
            # Let's stick to the "find blockers" approach but optimize it.
            
            # Find one blocker quickly:
            # Check 2*c
            if counts[2 * c] >= 2:
                # Find a pair a,b in S such that a+b=2c
                # Iterate S
                for a in S:
                    b = 2 * c - a
                    if b < a: continue
                    if b > max_val: continue
                    if in_S[b]:
                        # a and b are blockers
                        # Try swapping a with c
                        try_remove(a)
                        if try_add(c):
                            break
                        else:
                            try_add(a)
            else:
                # Check c+y
                # Pick a random y from S
                # If counts[c+y] >= 2, then y is a "potential" blocker?
                # No, y is part of the set, but the blocker is the OTHER pair.
                # If counts[c+y] == 2, the representations are {c, y} and {u, v}.
                # So u and v are the blockers.
                # We need to find u, v.
                
                # Let's just pick a random y in S.
                # If counts[c+y] >= 2, find the other pair.
                # This is expensive.
                
                # Alternative: Just try random swaps of any y in S with c.
                # If the swap is valid, it means c could fit in place of y.
                # This is effectively what we did in the "random swap" loop above.
                # Let's just use the random swap loop.
                pass

        return S

    # Run multiple trials if time permits?
    # Greedy is fast. Local search is the bottleneck.
    # Let's just run one trial with the improved local search.
    # The structural change is:
    # 1. Randomized Greedy Init.
    # 2. Local Search uses random swaps instead of exhaustive blocker search.
    # This might be faster per step, allowing more steps.
    # Or it might be less effective.
    # Let's try to combine:
    # Use the exhaustive blocker search but limit the number of candidates?
    
    # Actually, the original code's `get_blockers` was O(M^2).
    # If M=50, M^2=2500.
    # If we do this 1000 times, 2.5M ops. Very fast.
    # So the bottleneck is not the blocker search.
    # The bottleneck is the number of iterations.
    
    # So, let's keep the exhaustive blocker search but ensure we use the time well.
    # The main change: Randomized Greedy Init.
    
    S = run_one_trial()
    return S
