import time
import random

def construct():
    """
    Constructs a B_3[1] set (unique triple sums) of maximum size.
    Uses a greedy initialization followed by randomized local search (swaps)
    to maximize cardinality within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    def can_add(x, selected, sums):
        if 3*x in sums:
            return False
        for s in selected:
            if 2*x + s in sums:
                return False
        n_sel = len(selected)
        for i in range(n_sel):
            s = selected[i]
            for j in range(i, n_sel):
                t = selected[j]
                if x + s + t in sums:
                    return False
        return True

    def add_x(x, selected, sums):
        selected.append(x)
        sums.add(3*x)
        for s in selected[:-1]:
            sums.add(2*x + s)
        n_sel = len(selected)
        for i in range(n_sel-1):
            s = selected[i]
            for j in range(i, n_sel-1):
                t = selected[j]
                sums.add(x + s + t)

    def remove_x(x, selected, sums):
        # Remove x from selected and sums
        idx = selected.index(x)
        selected.pop(idx)
        # Remove sums involving x
        sums.discard(3*x)
        for s in selected:
            sums.discard(2*x + s)
        n_sel = len(selected)
        for i in range(n_sel):
            s = selected[i]
            for j in range(i, n_sel):
                t = selected[j]
                sums.discard(x + s + t)

    # 1. Greedy Initialization
    selected = []
    sums = set()
    for x in range(1, 4097):
        if time.time() > deadline - 5: # Reserve time for search
            break
        if can_add(x, selected, sums):
            add_x(x, selected, sums)
            
    best_size = len(selected)
    best_selected = selected[:]
    best_sums = sums.copy()
    
    # 2. Randomized Local Search
    # Try to improve by adding random elements or swapping
    # We maintain the current state and try moves.
    
    # To speed up can_add, we could use a boolean array for sums, but set is fine for k~40.
    
    while time.time() < deadline:
        # Try to add a random element
        x = random.randint(1, 4096)
        if x not in selected:
            if can_add(x, selected, sums):
                add_x(x, selected, sums)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = selected[:]
                    best_sums = sums.copy()
        
        # Try a swap: remove y, add x
        # Only if we haven't grown
        if len(selected) < 40: # Max possible
            y = random.choice(selected)
            x = random.randint(1, 4096)
            if x != y and x not in selected:
                # Tentatively remove y
                remove_x(y, selected, sums)
                if can_add(x, selected, sums):
                    add_x(x, selected, sums)
                    # If we swapped, size is same. 
                    # We might want to accept swaps that allow future growth?
                    # For now, only accept if it leads to immediate growth or is "better" (e.g. smaller values)?
                    # Since we want max size, and size didn't change, we revert unless we think it helps.
                    # Actually, random walks can get stuck. 
                    # Let's try to accept swaps that replace a large number with a small one?
                    # Or just revert.
                    remove_x(x, selected, sums)
                    add_x(y, selected, sums)
                else:
                    add_x(y, selected, sums)
                    
        # Try 2-for-1 swap? (Remove 2, add 1) - reduces size, bad.
        # Try 1-for-2 swap? (Remove 1, add 2) - increases size!
        # This is the most promising move.
        if len(selected) < 40:
            y = random.choice(selected)
            # Find two candidates x1, x2 that can be added if y is removed
            remove_x(y, selected, sums)
            # Try to add two random elements
            # This is expensive. 
            # Simplified: Try to add one random element. If successful, we are at same size.
            # Then try to add another?
            
            # Let's stick to simple add/swap for now to ensure speed.
            add_x(y, selected, sums)

    return best_selected
