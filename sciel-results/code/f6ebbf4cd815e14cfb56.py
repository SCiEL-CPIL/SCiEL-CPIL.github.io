import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    N = 1000
    max_sum = 2 * N
    
    # State
    active = set()
    counts = [0] * (max_sum + 1)
    
    # Helper to add element
    def add_elem(x):
        active.add(x)
        # Update counts for pairs (x, y) with y in active, y >= x
        # And (x, x)
        for y in active:
            if y >= x:
                s = x + y
                counts[s] += 1
        # Note: The loop above includes y=x, so counts[2*x] is incremented once.
        # Wait, if y=x, s=2x. If y!=x, s=x+y.
        # The condition y>=x ensures we count each pair {x,y} exactly once when adding x?
        # If we add x, and y is already in active.
        # If y > x, pair is {x,y}. Sum x+y.
        # If y == x, pair is {x,x}. Sum 2x.
        # If y < x, pair {y,x} was already counted when y was added? No, y was added earlier.
        # When y was added, it iterated over active elements z >= y.
        # If x was not in active then, the pair {x,y} was NOT counted.
        # So when adding x, we must count pairs with all existing y.
        # But we must be careful not to double count.
        # Standard way: When adding x, iterate all y in active.
        # If we do this, we count {x,y} for all y.
        # But when y was added, it counted {y,x} only if x was in active? No, x was not.
        # So {x,y} is counted exactly once, when the later of the two is added?
        # No, this logic is flawed.
        
        # Let's restart the update logic.
        # Maintain counts of sums for all pairs {a,b} in active.
        # When adding x:
        # New pairs are {x, y} for all y in active (before adding x) AND {x, x}.
        # So we should iterate over a snapshot of active BEFORE adding x.
        
        pass

    # Corrected Update Logic
    def try_add(x):
        if x in active:
            return True
        # Check validity
        # New sums: x+x and x+y for y in active
        # We need to ensure counts[s] < 2 for all new s.
        # Actually, if counts[s] is 2, adding one more makes it 3 (invalid).
        # So we need counts[s] <= 1 for all s involved?
        # No, if counts[s] is 1, adding makes it 2 (valid).
        # If counts[s] is 0, adding makes it 1 (valid).
        # If counts[s] is 2, adding makes it 3 (invalid).
        
        # So, check if any s in {2x} U {x+y for y in active} has counts[s] == 2.
        
        # Optimization: Pre-check
        if counts[2*x] == 2:
            return False
        for y in active:
            s = x + y
            if counts[s] == 2:
                return False
        
        # If valid, add
        active.add(x)
        counts[2*x] += 1
        for y in active:
            if y != x:
                s = x + y
                counts[s] += 1
        return True

    def try_remove(x):
        if x not in active:
            return True
        active.remove(x)
        counts[2*x] -= 1
        for y in active:
            s = x + y
            counts[s] -= 1
        return True

    # Greedy Initialization with Randomization
    def run_greedy(order):
        nonlocal active, counts
        active = set()
        counts = [0] * (max_sum + 1)
        
        for x in order:
            if try_add(x):
                pass
            # If time is up, break
            if time.time() > deadline:
                break
        return len(active)

    # Local Search: Try to improve by swapping
    def local_search():
        nonlocal active, counts
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any element
            for x in range(1, N+1):
                if x not in active:
                    if try_add(x):
                        improved = True
                        break # Restart loop to check for more additions
            if improved:
                continue
                
            # If no additions, try a swap: remove one, add one
            # Or try to remove one and see if we can add more than 1
            # Let's try: Pick random y in active, remove it, try to add random x's
            # This is a perturbation.
            if active:
                y = random.choice(list(active))
                try_remove(y)
                # Try to add elements greedily again
                # We can just call a limited greedy or try all x
                added_any = False
                for x in range(1, N+1):
                    if x not in active:
                        if try_add(x):
                            added_any = True
                            # Don't break, try to add as many as possible
                # If we ended up with a smaller set, we might want to revert?
                # But maybe it unlocks a better configuration later.
                # For now, we accept the state. The outer loop (ILS) will manage best solution.
                improved = True # Force another iteration to see if we can add more

    best_size = 0
    best_active = set()
    best_counts = [0] * (max_sum + 1)
    
    # Run multiple restarts
    while time.time() < deadline:
        # Random order
        order = list(range(1, N+1))
        random.shuffle(order)
        
        # Reset state
        active = set()
        counts = [0] * (max_sum + 1)
        
        # Greedy
        for x in order:
            if try_add(x):
                pass
            if time.time() > deadline:
                break
                
        # Local Search
        local_search()
        
        if len(active) > best_size:
            best_size = len(active)
            best_active = set(active)
            best_counts = counts[:]
            
        # If we found a very large set, we can tighten the time check
        # Or just keep going.
        
    return list(best_active)
