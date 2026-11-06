"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55
    
    TARGET = 200
    ALL = list(range(1, 101))
    
    def can_add(S, x, reachable):
        """Check if we can add x to S without creating sum TARGET."""
        if x >= TARGET:
            return False
        return not reachable[TARGET - x]
    
    def update_reachable(reachable, x):
        """Update reachable sums after adding x. Returns new reachable."""
        new_reachable = reachable[:]
        for s in range(TARGET - x, -1, -1):
            if reachable[s]:
                ns = s + x
                if ns < TARGET:
                    new_reachable[ns] = True
        return new_reachable
    
    def greedy(order):
        """Greedy construction based on a given order of elements."""
        S = []
        reachable = [False] * (TARGET + 1)
        reachable[0] = True
        
        for x in order:
            if x >= TARGET:
                continue
            if not reachable[TARGET - x]:
                S.append(x)
                reachable = update_reachable(reachable, x)
        
        return S, reachable
    
    def local_search(S, reachable, max_iters=5000):
        """Try to improve the solution by adding excluded elements or swapping."""
        S_set = set(S)
        excluded = [x for x in ALL if x not in S_set]
        
        best_S = S[:]
        best_reachable = reachable[:]
        best_len = len(S)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Try to add an excluded element
            added = False
            for x in excluded:
                if x >= TARGET:
                    continue
                if not best_reachable[TARGET - x]:
                    # We can add x
                    new_S = best_S + [x]
                    new_reachable = update_reachable(best_reachable, x)
                    best_S = new_S
                    best_reachable = new_reachable
                    best_len += 1
                    excluded.remove(x)
                    added = True
                    break
            
            if added:
                continue
            
            # Try swaps: remove one element, add one excluded element
            # Only if it doesn't reduce size (we're trying to maintain or improve)
            # Actually, let's try to find a swap that allows adding more later
            # For now, just try random swaps
            if excluded and len(S) > 1:
                # Try a random swap
                if random.random() < 0.3:
                    remove_idx = random.randint(0, len(best_S) - 1)
                    add_x = random.choice(excluded)
                    
                    if add_x >= TARGET:
                        continue
                    
                    # Remove best_S[remove_idx]
                    removed_x = best_S[remove_idx]
                    new_S = [x for i, x in enumerate(best_S) if i != remove_idx]
                    
                    # Recompute reachable from scratch (expensive but correct)
                    new_reachable = [False] * (TARGET + 1)
                    new_reachable[0] = True
                    for y in new_S:
                        if y < TARGET:
                            new_reachable = update_reachable(new_reachable, y)
                    
                    # Try to add add_x
                    if add_x < TARGET and not new_reachable[TARGET - add_x]:
                        new_S.append(add_x)
                        new_reachable = update_reachable(new_reachable, add_x)
                        
                        # Try to add more excluded elements
                        for y in excluded:
                            if y == add_x or y >= TARGET:
                                continue
                            if y in new_S:
                                continue
                            if not new_reachable[TARGET - y]:
                                new_S.append(y)
                                new_reachable = update_reachable(new_reachable, y)
                        
                        if len(new_S) > best_len:
                            best_S = new_S
                            best_reachable = new_reachable
                            best_len = len(new_S)
                            excluded = [x for x in ALL if x not in set(best_S)]
        
        return best_S, best_reachable
    
    best_set = []
    best_len = 0
    best_reachable = None
    
    # Initial greedy with different orderings
    static_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(1, 51)) + list(range(100, 50, -1)),
    ]
    
    for o in static_orders:
        if time.time() > deadline - 5:
            break
        S, reachable = greedy(o)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            best_reachable = reachable
    
    # Local search to improve
    if best_reachable is not None:
        best_set, best_reachable = local_search(best_set, best_reachable)
        best_len = len(best_set)
    
    # More greedy attempts with random orderings, then local search
    while time.time() < deadline - 5:
        if random.random() < 0.2:
            order = ALL[:]
            random.shuffle(order)
        else:
            # Weighted random: favor larger elements first (they're harder to fit)
            order = sorted(ALL, key=lambda x: (-x, random.random()))
        
        S, reachable = greedy(order)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            best_reachable = reachable
            # Do a quick local search
            best_set, best_reachable = local_search(best_set, best_reachable, max_iters=500)
            best_len = len(best_set)
        
        if best_len == 100:
            break
    
    if not best_set:
        return [100]
        
    return best_set

# EVOLVE-BLOCK-END
