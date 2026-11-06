import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N  # 2000

    def build_counts_numpy(chosen):
        """Build sum counts as a numpy array for speed."""
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        arr = np.array(sorted(chosen), dtype=np.int32)
        n = len(arr)
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                b = arr[j]
                counts[a + b] += 1
        return counts

    def is_valid_add_numpy(counts, chosen_set, x):
        """Check if x can be added using numpy array counts."""
        for y in chosen_set:
            s = x + y
            if counts[s] >= 2:
                return False
        if counts[2 * x] >= 2:
            return False
        return True

    def add_element_numpy(counts, chosen_set, x):
        """Add x to chosen_set and update counts."""
        for y in chosen_set:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen_set.add(x)

    def remove_element_numpy(counts, chosen_set, x):
        """Remove x from chosen_set and update counts."""
        for y in chosen_set:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen_set.discard(x)

    # Phase 1: Multiple greedy passes with numpy
    best_chosen = set()
    best_size = 0
    
    # Forward greedy
    chosen_set = set()
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if is_valid_add_numpy(counts, chosen_set, x):
            add_element_numpy(counts, chosen_set, x)
    best_chosen = chosen_set.copy()
    best_size = len(chosen_set)
    
    # Backward greedy
    if time.time() < deadline - 10:
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in range(N, 0, -1):
            if is_valid_add_numpy(counts, chosen_set, x):
                add_element_numpy(counts, chosen_set, x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Random greedy passes - more passes since we'll have faster local search
    random.seed(99999)
    for _ in range(500):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in order:
            if is_valid_add_numpy(counts, chosen_set, x):
                add_element_numpy(counts, chosen_set, x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Phase 2: Try to add more elements to the best solution
    chosen_set = best_chosen.copy()
    counts = build_counts_numpy(list(chosen_set))
    
    # Try to add any element that can be added
    added = True
    while added and time.time() < deadline - 5:
        added = False
        for x in range(1, N + 1):
            if x not in chosen_set and is_valid_add_numpy(counts, chosen_set, x):
                add_element_numpy(counts, chosen_set, x)
                added = True
                break
    
    best_size = len(chosen_set)
    best_chosen = chosen_set.copy()
    
    # Phase 3: Fast swap-based local search
    if time.time() < deadline - 2:
        counts = build_counts_numpy(list(best_chosen))
        best_size = len(best_chosen)
        
        # Fast local search: remove one, try to add many candidates
        for iteration in range(2000):
            if time.time() > deadline - 2:
                break
            
            if len(best_chosen) == 0:
                break
            
            # Pick a random element to remove
            old_x = random.choice(list(best_chosen))
            
            # Temporarily remove it
            remove_element_numpy(counts, best_chosen, old_x)
            
            # Try to add a different element - quick check
            found = False
            # Try a random sample of candidates first (fast)
            candidates = [x for x in range(1, N + 1) if x != old_x and x not in best_chosen]
            random.shuffle(candidates)
            
            for x in candidates[:50]:
                if is_valid_add_numpy(counts, best_chosen, x):
                    add_element_numpy(counts, best_chosen, x)
                    found = True
                    break
            
            if not found:
                # Put old_x back
                add_element_numpy(counts, best_chosen, old_x)
            
            # Update best if improved
            if len(best_chosen) > best_size:
                best_size = len(best_chosen)
    
    # Phase 4: 2-opt local search - remove two, add two
    if time.time() < deadline - 2:
        counts = build_counts_numpy(list(best_chosen))
        best_size = len(best_chosen)
        
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            
            if len(best_chosen) < 2:
                break
            
            # Pick two random elements to remove
            elems = list(best_chosen)
            random.shuffle(elems)
            old_x, old_y = elems[0], elems[1]
            
            # Temporarily remove both
            remove_element_numpy(counts, best_chosen, old_x)
            remove_element_numpy(counts, best_chosen, old_y)
            
            # Try to add two different elements
            found_x = False
            found_y = False
            added_x = None
            added_y = None
            
            candidates = [x for x in range(1, N + 1) if x not in (old_x, old_y) and x not in best_chosen]
            random.shuffle(candidates)
            
            for x in candidates[:30]:
                if is_valid_add_numpy(counts, best_chosen, x):
                    add_element_numpy(counts, best_chosen, x)
                    found_x = True
                    added_x = x
                    break
            
            if found_x:
                candidates2 = [x for x in range(1, N + 1) if x not in (old_x, old_y, added_x) and x not in best_chosen]
                random.shuffle(candidates2)
                for x in candidates2[:30]:
                    if is_valid_add_numpy(counts, best_chosen, x):
                        add_element_numpy(counts, best_chosen, x)
                        found_y = True
                        added_y = x
                        break
            
            if not (found_x and found_y):
                # Roll back
                if found_x:
                    remove_element_numpy(counts, best_chosen, added_x)
                add_element_numpy(counts, best_chosen, old_x)
                add_element_numpy(counts, best_chosen, old_y)
            
            # Update best if improved
            if len(best_chosen) > best_size:
                best_size = len(best_chosen)
    
    # Phase 5: Final polish - try to add any remaining valid elements
    if time.time() < deadline - 1:
        counts = build_counts_numpy(list(best_chosen))
        
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for x in range(1, N + 1):
                if x not in best_chosen and is_valid_add_numpy(counts, best_chosen, x):
                    add_element_numpy(counts, best_chosen, x)
                    added = True
                    break
        
        best_size = len(best_chosen)
        best_chosen = best_chosen.copy()
    
    return list(best_chosen)
