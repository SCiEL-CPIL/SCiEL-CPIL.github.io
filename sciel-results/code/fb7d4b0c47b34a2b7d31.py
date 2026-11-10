import time
import random

def construct():
    n = 512
    # Helper to check if x can be added to S without forming a 4-AP
    def can_add(x, S):
        # S is a set
        # Check if adding x creates a 4-AP
        # x can be part of AP:
        # 1. x, x+d, x+2d, x+3d
        # 2. x-d, x, x+d, x+2d
        # 3. x-2d, x-d, x, x+d
        # 4. x-3d, x-2d, x-d, x
        
        # 1. x is start
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # 2. x is 2nd term (a+d)
        # a = x-d
        max_d2 = (x - 1) # a >= 1 => x-d >= 1 => d <= x-1
        # also x+2d <= n => d <= (n-x)//2
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # 3. x is 3rd term (a+2d)
        # a = x-2d
        max_d3 = (x - 2) // 2 # x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
        # also x+d <= n => d <= n-x
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # 4. x is 4th term (a+3d)
        # a = x-3d
        max_d4 = (x - 3) // 3 # x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        
        return True

    # 1. Generate initial greedy set
    S = set()
    for x in range(1, n + 1):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # 2. Local Search
    start_time = time.time()
    deadline = 55.0 # 5s margin
    
    # List of elements not in S
    # We'll use random sampling for moves
    
    while time.time() < deadline:
        # Try to add a random element
        added = False
        for _ in range(100): # Try a few random adds
            x = random.randint(1, n)
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    added = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    break
        
        if not added:
            # Try a swap: remove one, try to add two? Or just swap to change configuration
            # Simple swap: remove random x, add random y
            if S:
                x_to_remove = random.choice(list(S))
                S.remove(x_to_remove)
                
                # Try to fill the gap or just add random y
                improved = False
                for _ in range(100):
                    y = random.randint(1, n)
                    if y not in S:
                        if can_add(y, S):
                            S.add(y)
                            improved = True
                            if len(S) > best_size:
                                best_S = set(S)
                                best_size = len(S)
                            break
                
                # If we didn't add anything, revert the removal to maintain size
                # Or keep the removal if it allows future adds? 
                # For simplicity, if we didn't add, put x back.
                if not improved:
                    S.add(x_to_remove)
                    
        # Occasionally reset to best to avoid local minima
        if random.random() < 0.1:
            S = set(best_S)

    return sorted(best_S)
