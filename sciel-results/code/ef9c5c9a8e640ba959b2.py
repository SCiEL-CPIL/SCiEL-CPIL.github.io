import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Precompute nothing heavy, just use the direct check for clarity and speed with small N
    
    def is_valid_with_x(S, x):
        """
        Check if S U {x} is valid, assuming S is valid.
        Only checks triples involving x.
        """
        S_list = list(S)
        
        # Case 1: x is the smallest (a)
        # Need b, c in S such that x < b <= c and x | (b+c)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                # b, c are in S, so b != x and c != x
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle (b)
        # Need a, c in S such that a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                # a, c in S, so a != x and c != x
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest (c)
        # Need a, b in S such that a < b <= x and a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a:
                    continue
                if b > x:
                    continue
                # a, b in S, so a != x and b != x
                if (b + x) % a == 0:
                    return False
        
        return True

    def get_best_set():
        # Start with Upper Tail
        S = set(range(n // 2 + 1, n + 1))
        
        # Greedy Add
        for x in range(1, n + 1):
            if x in S:
                continue
            if is_valid_with_x(S, x):
                S.add(x)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding any element
            for x in range(1, n + 1):
                if x in S:
                    continue
                if is_valid_with_x(S, x):
                    S.add(x)
                    improved = True
                    break # Restart loop to try adding more
            
            if not improved:
                # Try swaps: remove y, add x
                # We want to find a swap that might enable future additions
                # Or just a swap that keeps size but changes structure
                # A better move: Remove y, then greedily add as many as possible.
                # If size increases, keep.
                
                # Pick a random y to remove
                if S:
                    y = random.choice(list(S))
                    S.remove(y)
                    
                    # Try to add elements greedily
                    added_any = False
                    for x in range(1, n + 1):
                        if x == y:
                            continue
                        if x in S:
                            continue
                        if is_valid_with_x(S, x):
                            S.add(x)
                            added_any = True
                    
                    if len(S) < n // 2 + 1: # Heuristic: if we lost too much, revert? 
                        # Actually, we just want max size. 
                        # If len(S) is same as before, it's a neutral swap.
                        # If len(S) > before, it's an improvement.
                        pass
                    
                    # We need to track if this was an improvement.
                    # Since we don't know the previous size easily in this block without storing it, 
                    # let's restructure.
                    
        return S

    # Better Local Search Implementation
    best_set = set(range(n // 2 + 1, n + 1))
    best_size = len(best_set)
    
    # Initial Greedy
    for x in range(1, n + 1):
        if x in best_set:
            continue
        if is_valid_with_x(best_set, x):
            best_set.add(x)
            best_size = len(best_set)
            
    current_set = set(best_set)
    
    while time.time() < deadline - 2:
        # Try to add
        added = False
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if is_valid_with_x(current_set, x):
                current_set.add(x)
                added = True
                break
        
        if added:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            continue
        
        # Try a 1-1 swap
        # Remove a random element, try to add a random element
        if not current_set:
            break
            
        y = random.choice(list(current_set))
        x = random.choice([i for i in range(1, n + 1) if i not in current_set])
        
        # Check if swap is valid
        temp_set = set(current_set)
        temp_set.remove(y)
        
        if is_valid_with_x(temp_set, x):
            temp_set.add(x)
            # Check if this swap enables any additions
            # Try to add more
            for z in range(1, n + 1):
                if z in temp_set:
                    continue
                if is_valid_with_x(temp_set, z):
                    temp_set.add(z)
            
            if len(temp_set) > len(current_set):
                current_set = temp_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
            else:
                # Neutral swap, might help later? 
                # For now, only accept if size increases to avoid getting stuck in loops.
                # Or accept with probability?
                # Let's only accept if size increases.
                pass
        else:
            # Try to find a valid x for this y
            pass

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
