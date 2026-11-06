import time
import random

def construct():
    N = 256
    deadline = time.time() + 50  # 50 seconds safety margin
    
    def is_valid_addition(x, in_set, S):
        """Check if x can be added to S without creating a 3-AP."""
        if in_set[x]:
            return False
        # Check all y in S
        for y in S:
            # Case 1: x is the end term (a, m, x) => m = (a+x)/2
            if y < x and (x + y) % 2 == 0:
                m = (x + y) // 2
                if in_set[m]:
                    return False
            # Case 2: x is the middle term (a, x, z) => z = 2x - a
            if y < x:
                z = 2 * x - y
                if z <= N and in_set[z]:
                    return False
            # Case 3: x is the start term (x, m, z) => m = x + d, z = x + 2d
            # This is equivalent to: for any z in S with z > x, if 2z - x in S?
            # Actually, if x is start, and y is in S, y could be middle or end.
            # If y is middle: 2y - x in S?
            # If y is end: (x+y)/2 in S?
            # We already covered "y is middle" in Case 2 (where y was 'a' and x was 'm'?? No).
            # Let's re-verify.
            # 3-AP: a, a+d, a+2d.
            # If we add x:
            # 1. x = a+2d. Then a, a+d are in S. a+d = (a+x)/2.
            #    So for any a in S, if (a+x) even and (a+x)/2 in S, invalid.
            #    This corresponds to y=a in Case 1.
            # 2. x = a+d. Then a, a+2d are in S. a+2d = 2x-a.
            #    So for any a in S, if 2x-a in S, invalid.
            #    This corresponds to y=a in Case 2.
            # 3. x = a. Then a+d, a+2d are in S.
            #    Let y = a+d (in S). Then a+2d = 2y-x.
            #    So for any y in S with y > x, if 2y-x in S, invalid.
            #    OR let y = a+2d (in S). Then a+d = (x+y)/2.
            #    So for any y in S with y > x, if (x+y) even and (x+y)/2 in S, invalid.
            #    
            # My Case 1 and 2 only iterated y < x.
            # We need to handle y > x.
            
            if y > x:
                # Subcase 3a: y is the middle term (x, y, z) => z = 2y - x
                z = 2 * y - x
                if z <= N and in_set[z]:
                    return False
                # Subcase 3b: y is the end term (x, m, y) => m = (x+y)/2
                if (x + y) % 2 == 0:
                    m = (x + y) // 2
                    if in_set[m]:
                        return False
        return True

    def remove_element(x, in_set, S):
        in_set[x] = False
        S.remove(x)

    def add_element(x, in_set, S):
        in_set[x] = True
        S.append(x)

    def get_score(S):
        return sum(1.0 / x for x in S)

    # Initial construction: Greedy
    S = []
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        if is_valid_addition(x, in_set, S):
            add_element(x, in_set, S)
    
    current_score = get_score(S)
    best_S = S[:]
    best_score = current_score
    best_in_set = in_set[:]

    # Local Search
    # We will try to improve by swaps or additions
    # Since N is small, we can afford some iterations.
    
    # To speed up is_valid_addition, we can precompute conflicts or use bitsets, 
    # but with N=256 and |S|~100, O(|S|) check is ~100 ops. 
    # 50 seconds allows for millions of checks.
    
    # Strategy: 
    # 1. Try adding any valid x.
    # 2. Try swapping x in S with y not in S.
    # 3. Try removing x and adding y, z.
    
    # We'll use a simple hill climbing with random restarts if stuck.
    
    # To make it robust, we'll just do a few passes of improvement.
    
    # Optimization: 
    # Instead of full restart, we can do a "shake-up" if no improvement for a while.
    
    # Let's try a more aggressive local search:
    # For each x in S, try removing it. If we can add y, z with higher sum, do it.
    
    # However, a simpler and effective approach for this specific objective (sum of reciprocals)
    # is that small numbers are king. The greedy set is already quite good.
    # The main loss is if greedy blocks a small number later? No, greedy adds small first.
    # Greedy adds 1. Then 2. 1,2,3 is AP, so 3 is blocked.
    # 1,2,4? 1,2,3 no. 1,3,5 no. 2,4,6 no.
    # Greedy: 1, 2, 4. 
    # Next 3: 1,2,3 AP? Yes (1,2,3). Blocked.
    # Next 4: 2,3,4? 3 not in S. 1,? 1,2.5 no. 4,? 
    # Check 4: 
    # y=1: (1+4)/2=2.5 no. 2*4-1=7 no.
    # y=2: (2+4)/2=3 no. 2*4-2=6 no.
    # So 4 is added.
    # Next 5:
    # y=1: (1+5)/2=3 no. 2*5-1=9 no.
    # y=2: (2+5) odd. 2*5-2=8 no.
    # y=4: (4+5) odd. 2*5-4=6 no.
    # So 5 is added.
    # S = [1, 2, 4, 5]
    # Next 6:
    # y=2: (2+6)/2=4. 4 in S. AP 2,4,6. Blocked.
    # Next 7:
    # y=1: (1+7)/2=4. 4 in S. AP 1,4,7. Blocked.
    # Next 8:
    # y=4: (4+8)/2=6 no. 2*8-4=12 no.
    # y=1: (1+8) odd. 2*8-1=15 no.
    # y=2: (2+8)/2=5. 5 in S. AP 2,5,8. Blocked.
    # Next 9:
    # y=5: (5+9)/2=7 no. 2*9-5=13 no.
    # y=1: (1+9)/2=5. 5 in S. AP 1,5,9. Blocked.
    # Next 10:
    # y=2: (2+10)/2=6 no. 2*10-2=18 no.
    # y=4: (4+10)/2=7 no. 2*10-4=16 no.
    # y=5: (5+10)/2=7.5 no. 2*10-5=15 no.
    # y=1: (1+10) odd. 2*10-1=19 no.
    # So 10 is added.
    # S = [1, 2, 4, 5, 10]
    
    # The greedy set is deterministic.
    # Can we improve it?
    # What if we remove 2?
    # S = [1, 4, 5, 10]
    # Can we add 3? 
    # Check 3 with [1, 4, 5, 10]:
    # y=1: (1+3)/2=2 no. 2*3-1=5. 5 in S. AP 1,3,5. Blocked.
    # So removing 2 doesn't help add 3.
    
    # What if we remove 1?
    # S = [2, 4, 5, 10]
    # Can we add 1? No, we just removed it.
    # Can we add 3?
    # y=2: (2+3) odd. 2*3-2=4. 4 in S. AP 2,3,4. Blocked.
    
    # It seems the greedy set is locally optimal for single swaps.
    # We need to look for 2-swaps or additions.
    
    # Let's run a local search that tries:
    # 1. Add any valid x.
    # 2. Swap x (in S) with y (not in S) if score improves.
    # 3. Remove x, add y, z if score improves.
    
    # To keep it simple and fast, we'll iterate through possible moves.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try additions
        for x in range(1, N + 1):
            if not in_set[x] and is_valid_addition(x, in_set, S):
                add_element(x, in_set, S)
                current_score += 1.0 / x
                improved = True
                # Update best if better
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]
                    best_in_set = in_set[:]
        
        # Try swaps
        # For each x in S, for each y not in S
        # This is O(|S| * N * |S|) which is ~ 100 * 256 * 100 = 2.5M ops per pass.
        # Might be too slow if we do many passes.
        # We can restrict y to be "close" to x or just try a few random swaps.
        
        # Let's try random swaps to escape local optima
        for _ in range(1000): # Limit swaps per pass
            if time.time() > deadline: break
            x = random.choice(S)
            y = random.randint(1, N)
            if y == x or in_set[y]:
                continue
            
            # Try swapping x with y
            # Remove x
            remove_element(x, in_set, S)
            # Check if y can be added
            if is_valid_addition(y, in_set, S):
                add_element(y, in_set, S)
                new_score = current_score - 1.0/x + 1.0/y
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_in_set = in_set[:]
                else:
                    # Revert
                    remove_element(y, in_set, S)
                    add_element(x, in_set, S)
            else:
                # Revert
                add_element(x, in_set, S)
                
    return best_S
