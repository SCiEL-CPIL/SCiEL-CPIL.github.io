<<<<<<< SEARCH
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin for 90s kill
    
    TARGET = 200
    ALL = list(range(1, 101))
    
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
            if excluded and len(best_S) > 1:
                if random.random() < 0.3:
                    remove_idx = random.randint(0, len(best_S) - 1)
                    add_x = random.choice(excluded)
                    
                    if add_x >= TARGET:
                        continue
                    
                    # Remove best_S[remove_idx]
                    removed_x = best_S[remove_idx]
                    new_S = [x for i, x in enumerate(best_S) if i != remove_idx]
                    
                    # Recompute reachable from scratch
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
    
    def backtrack_construct(elements, reachable, S, depth, max_depth, time_limit):
        """
        Backtracking search with forward checking to maximize cardinality.
        Tries to include elements in a smart order, pruning when infeasible.
        """
        if time.time() > time_limit:
            return S[:]
        
        if depth >= max_depth or not elements:
            return S[:]
        
        best_S = S[:]
        
        # Heuristic: sort remaining elements by descending value (larger first)
        # This tends to find solutions with more elements since large numbers
        # are more constrained
        sorted_elements = sorted(elements, key=lambda x: -x)
        
        for i, x in enumerate(sorted_elements):
            if time.time() > time_limit:
                break
            
            if x >= TARGET:
                continue
            
            # Forward check: can we add x?
            if reachable[TARGET - x]:
                continue
            
            # Try including x
            new_reachable = update_reachable(reachable, x)
            new_S = S + [x]
            
            # Recursively try to add more from remaining elements
            remaining = [e for e in elements if e != x]
            result = backtrack_construct(remaining, new_reachable, new_S, depth + 1, max_depth, time_limit)
            
            if len(result) > len(best_S):
                best_S = result
            
            # Early termination if we've included everything possible
            if len(best_S) == len(elements):
                break
        
        return best_S
    
    def improved_backtrack(elements, time_limit):
        """
        Improved backtracking that uses a smarter heuristic:
        Process elements in descending order, and at each step decide include/exclude.
        Uses forward checking to prune infeasible branches.
        """
        if time.time() > time_limit:
            return []
        
        # Start with all elements sorted descending
        elements = sorted(elements, key=lambda x: -x)
        
        def dfs(idx, reachable, S):
            if time.time() > time_limit:
                return S[:]
            
            if idx >= len(elements):
                return S[:]
            
            x = elements[idx]
            best_S = S[:]
            
            # Option 1: Try to include x
            if x < TARGET and not reachable[TARGET - x]:
                new_reachable = update_reachable(reachable, x)
                new_S = S + [x]
                result = dfs(idx + 1, new_reachable, new_S)
                if len(result) > len(best_S):
                    best_S = result
            
            # Option 2: Exclude x
            result = dfs(idx + 1, reachable, S)
            if len(result) > len(best_S):
                best_S = result
            
            return best_S
        
        return dfs(0, [False] * (TARGET + 1), [])
    
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
        if time.time() > deadline - 10:
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
    
    # Try backtracking construction with time-limited search
    # Use a shorter time limit for the backtrack attempt
    bt_time_limit = min(deadline - 5, time.time() + 20)
    bt_result = improved_backtrack(ALL, bt_time_limit)
    
    if len(bt_result) > best_len:
        best_len = len(bt_result)
        best_set = bt_result
        # Compute reachable for local search
        best_reachable = [False] * (TARGET + 1)
        best_reachable[0] = True
        for x in best_set:
            if x < TARGET:
                best_reachable = update_reachable(best_reachable, x)
        # Polish with local search
        best_set, best_reachable = local_search(best_set, best_reachable, max_iters=2000)
        best_len = len(best_set)
    
    # More greedy attempts with random orderings, then local search
    while time.time() < deadline - 5:
        if random.random() < 0.2:
            order = ALL[:]
            random.shuffle(order)
        else:
            # Weighted random: favor larger elements first
            order = sorted(ALL, key=lambda x: (-x, random.random()))
        
        S, reachable = greedy(order)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            best_reachable = reachable
            best_set, best_reachable = local_search(best_set, best_reachable, max_iters=500)
            best_len = len(best_set)
        
        if best_len == 100:
            break
    
    if not best_set:
        return [100]
        
    return best_set

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
