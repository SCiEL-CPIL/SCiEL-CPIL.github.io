import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin (150s limit, leave 35s)
    best_set = []
    best_size = 0
    
    max_n = 1000
    sum_size = 2 * max_n + 1
    
    def run_greedy(order):
        chosen = []
        counts = np.zeros(sum_size, dtype=np.int8)
        
        for c in order:
            # Check 2*c
            if counts[2 * c] >= 2:
                continue
            
            # Check c + x for x in chosen
            if chosen:
                sums = c + np.array(chosen)
                if np.any(counts[sums] >= 2):
                    continue
            
            # Update counts
            counts[2 * c] += 1
            chosen.append(c)
            if len(chosen) > 1:
                sums = c + np.array(chosen[:-1])
                counts[sums] += 1
        
        return chosen

    def run_greedy_with_prefilter(order, preselected):
        """Greedy that starts with preselected elements."""
        chosen = list(preselected)
        counts = np.zeros(sum_size, dtype=np.int8)
        
        # Build counts from preselected
        for i, a in enumerate(chosen):
            counts[2*a] += 1
            for b in chosen[i+1:]:
                counts[a+b] += 1
        
        for c in order:
            if c in preselected:
                continue
            # Check 2*c
            if counts[2 * c] >= 2:
                continue
            
            # Check c + x for x in chosen
            if chosen:
                sums = c + np.array(chosen)
                if np.any(counts[sums] >= 2):
                    continue
            
            # Update counts
            counts[2 * c] += 1
            chosen.append(c)
            if len(chosen) > 1:
                sums = c + np.array(chosen[:-1])
                counts[sums] += 1
        
        return chosen

    # 1. Deterministic Ascending
    asc_order = list(range(1, max_n + 1))
    current_set = run_greedy(asc_order)
    best_set = current_set
    best_size = len(current_set)

    # 2. Random Restarts
    while time.time() - start_time < deadline - 30.0:  # Leave 30s for local search
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_set = run_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # 3. Local Search on Best Set
    if time.time() - start_time < deadline - 5.0:
        best_set = local_search(best_set, start_time, deadline - 5.0)
        best_size = len(best_set)

    return best_set

def local_search(A, start_time, deadline):
    A = set(A)
    max_n = 1000
    sum_size = 2 * max_n + 1
    counts = np.zeros(sum_size, dtype=np.int8)
    
    # Build initial counts
    A_list = list(A)
    for i, a in enumerate(A_list):
        counts[2*a] += 1
        for b in A_list[i+1:]:
            counts[a+b] += 1
    
    def is_valid_insert(b, A_set, cnts):
        """Check if b can be added to A_set with given counts."""
        if cnts[2*b] >= 2:
            return False
        if A_set:
            sums = b + np.array(list(A_set))
            if np.any(cnts[sums] >= 2):
                return False
        return True

    def do_insert(b, A_set, cnts):
        """Insert b into A_set and update counts."""
        A_set.add(b)
        cnts[2*b] += 1
        other = [x for x in A_set if x != b]
        if other:
            sums = b + np.array(other)
            cnts[sums] += 1

    def remove_element(a, A_set, cnts):
        """Remove a from A_set and update counts."""
        A_set.discard(a)
        cnts[2*a] -= 1
        for b in A_set:
            cnts[a+b] -= 1

    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        
        if no_improve_count > 5:
            # Try a more aggressive restart with random swaps
            if time.time() > deadline - 10:
                break
            
        # Try Insert
        candidates = [x for x in range(1, max_n + 1) if x not in A]
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() > deadline: break
            if is_valid_insert(b, A, counts):
                do_insert(b, A, counts)
                improved = True
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try Swap: remove a, add b
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() > deadline: break
            
            # Temporarily remove a
            old_counts = counts.copy()
            remove_element(a, A, counts)
            
            # Try to find a b that can be added
            candidates = [x for x in range(1, max_n + 1) if x not in A]
            random.shuffle(candidates)
            
            found = False
            for b in candidates:
                if time.time() > deadline: break
                if is_valid_insert(b, A, counts):
                    do_insert(b, A, counts)
                    found = True
                    break
            
            if not found:
                # Restore
                A.add(a)
                counts = old_counts
                # Rebuild counts properly
                counts = np.zeros(sum_size, dtype=np.int8)
                A_list2 = list(A)
                for i, x in enumerate(A_list2):
                    counts[2*x] += 1
                    for y in A_list2[i+1:]:
                        counts[x+y] += 1
            
            if found:
                improved = True
                break
        
        if improved:
            no_improve_count = 0

    return list(A)
