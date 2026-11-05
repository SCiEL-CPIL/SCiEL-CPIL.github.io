import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute divisibility table for speed
    # divs[a][s] is True if a divides s
    max_sum = 2 * n
    divs = [[False] * (max_sum + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for multiple in range(a, max_sum + 1, a):
            divs[a][multiple] = True

    def can_add(x, S):
        """
        Check if x can be added to set S.
        S is a set of integers.
        Condition: For all a, b, c in S U {x} with a < b <= c, a !| (b+c).
        We only need to check triples involving x.
        1. x is a: x < b <= c in S. Check x !| (b+c).
        2. x is b: a < x <= c. a in S, c in S. Check a !| (x+c).
        3. x is c: a < b <= x. a in S, b in S. Check a !| (b+x).
        """
        S_list = list(S)
        
        # 1. x is a
        # b, c in S such that x < b <= c
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                if c < b:
                    continue
                if divs[x][b + c]:
                    return False
        
        # 2. x is b
        # a, c in S such that a < x <= c
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if divs[a][x + c]:
                    return False
        
        # 3. x is c
        # a, b in S such that a < b <= x
        for a in S_list:
            if a >= x:
                continue
            for b in S_list:
                if b <= a or b > x:
                    continue
                if divs[a][b + x]:
                    return False
        
        return True

    def get_blockers(x, S):
        """
        Return set of elements in S that prevent x from being added.
        """
        blockers = set()
        S_list = list(S)
        
        # 1. x is a
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                if c < b:
                    continue
                if divs[x][b + c]:
                    blockers.add(b)
                    blockers.add(c)
        
        # 2. x is b
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if divs[a][x + c]:
                    blockers.add(a)
                    blockers.add(c)
        
        # 3. x is c
        for a in S_list:
            if a >= x:
                continue
            for b in S_list:
                if b <= a or b > x:
                    continue
                if divs[a][b + x]:
                    blockers.add(a)
                    blockers.add(b)
        
        return blockers

    # Strategy 1: Greedy from top
    def greedy_top():
        S = set(range(n // 2 + 1, n + 1))
        for x in range(n // 2, 0, -1):
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    # Strategy 2: Greedy from bottom (building up)
    def greedy_bottom():
        S = set()
        for x in range(n, 0, -1):
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, time_limit):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        # Pre-compute list of all numbers
        all_nums = list(range(1, n + 1))
        
        # Simple local search: try to add any number, or swap
        while time.time() < time_limit:
            improved = False
            
            # Try adding any number from 1 to n not in S
            # Randomize order to avoid bias
            candidates = [x for x in all_nums if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > time_limit:
                    break
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    break # Restart scan after improvement
            
            if not improved:
                # Try swaps: remove y, add x
                # Only if it maintains size (to explore) or increases size (handled above)
                # We focus on increasing size primarily.
                # If stuck, try 1-for-1 swaps to escape local minima?
                # Given the constraint structure, 1-for-1 swaps rarely help increase size directly
                # unless they clear a path for a 1-for-2 later.
                # Let's try a limited number of random swaps.
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    if not S: break
                    y = random.choice(list(S))
                    candidates_swap = [x for x in all_nums if x not in S]
                    if not candidates_swap: break
                    x = random.choice(candidates_swap)
                    
                    # Check if S - {y} + {x} is valid
                    S_new = (S - {y}) | {x}
                    # We need to check validity of S_new
                    # This is expensive. 
                    # Instead, check if x can be added to S - {y}
                    if can_add(x, S - {y}):
                        # Accept swap (size remains same)
                        # This might help later
                        S = S_new
                        break # Restart scan

        return best_S

    # Initial constructions
    S1 = greedy_top()
    S2 = greedy_bottom()
    
    best_S = S1 if len(S1) > len(S2) else S2
    best_size = len(best_S)
    
    # Local search on the best one
    # Use remaining time
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 5:
        S_improved = local_search(best_S, start_time + remaining_time)
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            
    # Try local search on the other one if time permits
    if time.time() < deadline - 10:
        S_improved2 = local_search(S1 if best_S is S2 else S2, time.time() + 10)
        if len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    return sorted(best_S)

if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    # Verify
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
