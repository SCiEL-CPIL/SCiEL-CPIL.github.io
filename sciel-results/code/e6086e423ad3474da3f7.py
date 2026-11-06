import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 256
    
    def is_valid_3ap_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b: continue
                # 3-AP: a, b, c => 2b = a + c => c = 2b - a
                c = 2*b - a
                if c in S_set and c != a and c != b:
                    return False
        return True

    def check_add_validity(S_set, x):
        # Check if x can be added to S (represented as set)
        # Conflict if there exists a in S such that 2a - x in S (and != x)
        # OR there exists b in S such that 2x - b in S (and != x)
        # Note: 2a-x in S implies a, x, 2a-x is AP? No.
        # AP is (u, v, w) with 2v = u+w.
        # If x is the middle term: u, x, w => 2x = u+w => w = 2x-u.
        #   Check if u in S and w in S.
        # If x is the first term: x, u, w => 2u = x+w => w = 2u-x.
        #   Check if u in S and w in S.
        # If x is the last term: u, w, x => 2w = u+x => u = 2w-x.
        #   Check if w in S and u in S.
        
        # So:
        # 1. x is middle: for u in S, if (2x-u) in S and (2x-u) != x: return False
        # 2. x is first: for u in S, if (2u-x) in S and (2u-x) != x: return False
        # 3. x is last: for w in S, if (2w-x) in S and (2w-x) != x: return False
        
        # Note: Cases 2 and 3 are symmetric if we iterate over all elements.
        # Actually, if we iterate u in S:
        #   c1 = 2x - u. If c1 in S and c1 != x, conflict (x is middle).
        #   c2 = 2u - x. If c2 in S and c2 != x, conflict (x is last, u is middle).
        # This covers all cases where x is involved.
        
        for u in S_set:
            # x is middle term: u, x, 2x-u
            v = 2*x - u
            if v in S_set and v != x:
                return False
            # x is last term: 2u-x, u, x  => middle is u, first is 2u-x
            # Wait, if x is last, the AP is (a, b, x) with 2b = a+x => a = 2b-x.
            # So for each b in S, a = 2b-x. If a in S and a != x, conflict.
            # This is the same loop variable u acting as b.
            w = 2*u - x
            if w in S_set and w != x:
                return False
        return True

    def check_swap_validity(S_set, y, x):
        # Check if (S - {y}) U {x} is valid.
        # S is valid. S - {y} is valid.
        # Only need to check if x forms AP with S - {y}.
        S_minus_y = S_set - {y}
        return check_add_validity(S_minus_y, x)

    # Initial construction: Greedy
    S = []
    S_set = set()
    for i in range(1, N+1):
        if check_add_validity(S_set, i):
            S.append(i)
            S_set.add(i)
    
    current_sum = sum(1.0/x for x in S)
    best_S = list(S)
    best_sum = current_sum
    
    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find best swap
        best_swap_gain = 0
        best_x = None
        best_y = None
        
        # Optimization: Only consider x < y
        # We can iterate x from 1 to N
        for x in range(1, N+1):
            if x in S_set:
                continue
            for y in S:
                if y <= x:
                    continue
                gain = 1.0/x - 1.0/y
                if gain <= best_swap_gain:
                    continue
                if check_swap_validity(S_set, y, x):
                    best_swap_gain = gain
                    best_x = x
                    best_y = y
        
        if best_x is not None:
            # Perform swap
            S_set.remove(best_y)
            S_set.add(best_x)
            S = list(S_set)
            current_sum += best_swap_gain
            improved = True
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_S = list(S)
        
        # If no improvement, try a kick (random restart from current state)
        if not improved:
            # Simple kick: remove 5 random elements
            if len(S) > 10:
                to_remove = random.sample(S, 5)
                for r in to_remove:
                    S_set.remove(r)
                    S = list(S_set)
                current_sum = sum(1.0/x for x in S)
                # Try to re-add small numbers greedily?
                # Or just let the next hill climbing phase handle it.
                # Actually, removing elements makes the set "looser", so hill climbing might find new swaps.
                # But we need to ensure we don't lose too much.
                # Maybe just restart the hill climbing loop.
                # To avoid infinite loop of kicks, we can limit kicks or just rely on the fact that
                # the first pass found a local optimum, so a kick is needed to escape.
                # Let's do a few kicks.
                pass 

    return best_S
