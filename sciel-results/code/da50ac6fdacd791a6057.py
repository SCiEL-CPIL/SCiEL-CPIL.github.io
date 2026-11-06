import random
import time

N = 512

def construct():
    # Precompute nothing, just use sets for O(1) lookup.
    # can_add(x, S): checks if adding x to S creates a 4-AP.
    def can_add(x, S):
        # x is in [1, N]
        # Check 4 positions of x in a 4-AP: a, a+d, a+2d, a+3d
        # 1. x = a+3d => a = x-3d. Need x-3d, x-2d, x-d in S.
        # d can be 1 to (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        # 2. x = a+2d => a = x-2d. Need x-2d, x-d, x+d in S.
        # d can be 1 to min((x-1)//2, (N-x)//1)
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        # 3. x = a+d => a = x-d. Need x-d, x+d, x+2d in S.
        # d can be 1 to min(x-1, (N-x)//2)
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        # 4. x = a => Need x+d, x+2d, x+3d in S.
        # d can be 1 to (N-x)//3
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        return True

    # Initial Greedy Construction
    # Random order often yields better initial sizes than sorted.
    # We'll do a few random greedy runs and pick the best.
    
    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Generate initial best set
    best_set = set()
    best_size = 0
    
    # Run a few initial greedy passes with different random seeds/orders
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    current_set = set(best_set)
    current_size = best_size
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Local Search: Delete-Insert (Swap)
    # We remove one element and try to repack with random unselected elements.
    
    while time.time() < deadline:
        # Pick a random element to remove
        if not current_set:
            break
        s = random.choice(list(current_set))
        current_set.remove(s)
        
        # Collect unselected elements
        unselected = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(unselected)
        
        # Try to add them back in random order
        # This is the "repack" phase
        added_count = 0
        for x in unselected:
            if can_add(x, current_set):
                current_set.add(x)
                added_count += 1
                
        new_size = len(current_set)
        
        if new_size > best_size:
            best_size = new_size
            best_set = set(current_set)
        elif new_size < current_size:
            # Revert if we lost elements
            # This is a bit expensive to revert if we don't save state.
            # Better: Save state before modification.
            pass # We will handle state saving below
            
    # The above loop is inefficient because reverting is hard.
    # Let's restructure to save state.
    
    current_set = set(best_set)
    current_size = best_size
    start_time = time.time()
    deadline = start_time + 85
    
    while time.time() < deadline:
        # Save state
        prev_set = set(current_set)
        prev_size = current_size
        
        if not current_set:
            break
            
        s = random.choice(list(current_set))
        current_set.remove(s)
        
        unselected = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(unselected)
        
        for x in unselected:
            if can_add(x, current_set):
                current_set.add(x)
                
        new_size = len(current_set)
        
        if new_size > best_size:
            best_size = new_size
            best_set = set(current_set)
            # Accept move
            current_size = new_size
        elif new_size == current_size:
            # Accept move to walk on plateau
            # current_size remains same
            pass
        else:
            # Reject move, revert
            current_set = prev_set
            current_size = prev_size
            
    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    import random
    import time

    N = 512

    def can_add(x, S):
        # 1. x is 4th term: x-3d, x-2d, x-d
        for d in range(1, (x - 1) // 3 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        # 2. x is 3rd term: x-2d, x-d, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        # 3. x is 2nd term: x-d, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        # 4. x is 1st term: x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Initial Greedy Construction
    best_set = set()
    best_size = 0
    
    # Run a few initial greedy passes with different random seeds/orders
    for _ in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    current_set = set(best_set)
    current_size = best_size
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Local Search: Delete-Insert (Swap)
    while time.time() < deadline:
        # Save state
        prev_set = set(current_set)
        prev_size = current_size
        
        if not current_set:
            break
            
        s = random.choice(list(current_set))
        current_set.remove(s)
        
        # Create list of unselected elements
        # Optimization: We can just iterate 1..N and check if not in set
        # But shuffling a list of ~400 elements is fast.
        unselected = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(unselected)
        
        for x in unselected:
            if can_add(x, current_set):
                current_set.add(x)
                
        new_size = len(current_set)
        
        if new_size > best_size:
            best_size = new_size
            best_set = set(current_set)
            # Accept move
            current_size = new_size
        elif new_size == current_size:
            # Accept move to walk on plateau
            # current_size remains same
            pass
        else:
            # Reject move, revert
            current_set = prev_set
            current_size = prev_size
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
