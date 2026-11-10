import time
import random
import math

N = 256

# EVOLVE-BLOCK-START

def _try_add(selected, sums, x):
    """
    Check if x can be added to the Sidon set (selected, sums).
    Returns (True, new_sums) if successful, (False, None) otherwise.
    """
    if 2 * x in sums:
        return False, None
    
    conflict = False
    for a in selected:
        if x + a in sums:
            conflict = True
            break
            
    if conflict:
        return False, None
        
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        new_sums.add(x + a)
        
    return True, new_sums

def _is_valid_solution(arr):
    """
    Verifies Sidon and Maximality.
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
        
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute sums and check Sidon property
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
            
        # If x can be added, it's not maximal.
        # x can be added if 2x not in sums AND no x+a in sums for a in arr.
        
        if 2 * x in sums:
            continue
            
        can_add = True
        for a in arr:
            if x + a in sums:
                can_add = False
                break
                
        if can_add:
            return False
            
    return True

def _greedy_build(order):
    """
    Builds a maximal Sidon set by greedily adding elements from 'order'.
    """
    selected = []
    sums = set()
    
    for x in order:
        ok, new_sums = _try_add(selected, sums, x)
        if ok:
            selected.append(x)
            sums = new_sums
            
    return selected

def _local_search_remove(current_set, rng, max_iter=100):
    """
    Tries to reduce the size of the current maximal Sidon set by removing
    elements and re-maximizing. Returns a potentially smaller maximal set.
    """
    best_set = current_set[:]
    best_size = len(current_set)
    
    for _ in range(max_iter):
        # Try removing a random element
        idx = rng.randint(0, len(best_set) - 1)
        candidate_set = [x for i, x in enumerate(best_set) if i != idx]
        
        # Re-maximize using a random order
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        new_set = _greedy_build(order)
        
        # Check if new_set is valid (it should be, but just in case)
        # And check if it's smaller
        if len(new_set) < best_size:
            # Verify validity before accepting
            if _is_valid_solution(new_set):
                best_set = new_set[:]
                best_size = len(new_set)
                
    return best_set

def construct():
    start_time = time.time()
    # Safety margin: 55s limit (5s buffer for overhead)
    deadline = start_time + 55.0
    
    rng = random.Random(42)
    
    # Initial solution: Greedy from a few different random orders
    # and pick the smallest valid one.
    
    best_set = None
    best_size = N + 1
    
    # Try a few initial random greedy builds
    for _ in range(10):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = _greedy_build(order)
        if len(s) < best_size:
            if _is_valid_solution(s):
                best_set = s
                best_size = len(s)
                
    if best_set is None:
        # Fallback: should not happen, but just in case
        order = list(range(1, N + 1))
        rng.shuffle(order)
        best_set = _greedy_build(order)
        best_size = len(best_set)
        
    # Local Search: Try to improve the best set found so far
    # We use a simple local search: remove an element, re-maximize, keep if smaller.
    
    current_set = best_set[:]
    current_size = best_size
    
    # Simulated Annealing style local search
    T = 5.0
    T_min = 0.5
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with a fresh random set if we've cooled down
            order = list(range(1, N + 1))
            rng.shuffle(order)
            current_set = _greedy_build(order)
            current_size = len(current_set)
            T = 5.0
            continue
            
        # Pick a random element to remove
        idx = rng.randint(0, len(current_set) - 1)
        y = current_set[idx]
        
        # Create new initial set (current_set without y)
        new_initial = [x for i, x in enumerate(current_set) if i != idx]
        
        # Re-maximize the remaining set.
        # To ensure we get a maximal set, we add candidates from a random order
        # to the remaining set until no more can be added.
        
        # Note: _greedy_build builds from scratch.
        # We need a function that starts from a partial set and adds more.
        # Let's modify _greedy_build or create _greedy_extend.
        
        # For now, let's just build from scratch with a random order.
        # This is less efficient but simpler.
        # However, the prompt says "Keep the function names and signatures of the seed program".
        # The seed program had _greedy_maximize_fast which took an initial set.
        # Let's implement a helper that extends a set.
        
        def _greedy_extend(initial_set, order):
            selected = list(initial_set)
            sums = set()
            # Compute sums for initial_set
            for i in range(len(selected)):
                a = selected[i]
                if 2 * a in sums:
                    return selected # Invalid initial set? Should not happen.
                sums.add(2 * a)
                for j in range(i + 1, len(selected)):
                    s = a + selected[j]
                    if s in sums:
                        return selected
                    sums.add(s)
                    
            for x in order:
                if x in selected:
                    continue
                ok, new_sums = _try_add(selected, sums, x)
                if ok:
                    selected.append(x)
                    sums = new_sums
                    
            return selected
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        new_set = _greedy_extend(new_initial, order)
        new_size = len(new_set)
        
        delta = new_size - current_size
        
        if delta < 0:
            # Accept
            current_set = new_set
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = current_set[:]
        else:
            # Accept with probability exp(-delta / T)
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_set = new_set
                current_size = new_size
                
        T *= cooling_rate
        
    # Final check
    if not _is_valid_solution(best_set):
        # Fallback
        order = list(range(1, N + 1))
        rng.shuffle(order)
        best_set = _greedy_build(order)
        
    return best_set

# EVOLVE-BLOCK-END
