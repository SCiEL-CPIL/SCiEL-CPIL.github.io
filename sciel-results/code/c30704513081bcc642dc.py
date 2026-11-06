import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    
    def get_sums(arr):
        """Return set of all pairwise sums a+b where a<=b, or None if invalid."""
        sums = set()
        n = len(arr)
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                b = arr[j]
                s = a + b
                if s in sums:
                    return None
                sums.add(s)
        return sums
    
    def can_add(arr, sums, x):
        """Check if x can be added to arr without creating duplicate sums."""
        for y in arr:
            s = x + y
            if s in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def greedy_expand(arr, sums, remaining_candidates):
        """Greedily add elements from remaining_candidates to arr."""
        arr = arr[:]
        sums = sums.copy()
        remaining = set(remaining_candidates)
        
        while remaining:
            # Try to find a candidate that can be added
            added = False
            for x in sorted(remaining):
                if can_add(arr, sums, x):
                    arr.append(x)
                    arr.sort()
                    # Update sums
                    for y in arr[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    remaining.discard(x)
                    added = True
                    break
            if not added:
                break
        
        return arr, sums
    
    def kick(arr, sums):
        """Perform a random kick: remove 1-3 elements and try to re-add them or new ones."""
        arr = arr[:]
        sums = sums.copy()
        n = len(arr)
        
        if n <= 4:
            return arr, sums
        
        # Remove 1-2 random non-base elements
        base_set = {1, 2, 5, 11}
        non_base_indices = [i for i, x in enumerate(arr) if x not in base_set]
        
        if not non_base_indices:
            return arr, sums
        
        kick_size = min(random.randint(1, 2), len(non_base_indices))
        to_remove = random.sample(non_base_indices, kick_size)
        to_remove.sort(reverse=True)
        
        removed = []
        for idx in to_remove:
            x = arr[idx]
            arr.pop(idx)
            removed.append(x)
        
        # Recompute sums
        sums = get_sums(arr)
        if sums is None:
            # Shouldn't happen, but restore
            return arr, get_sums(arr)
        
        # Try to add back removed elements and new candidates
        candidates = set(removed)
        candidates.update(range(1, 285))
        candidates.difference_update(set(arr))
        
        # Greedily try to add candidates
        for x in sorted(candidates):
            if can_add(arr, sums, x):
                arr.append(x)
                arr.sort()
                for y in arr[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        
        return arr, sums
    
    def local_search(arr, sums, max_iters=50):
        """Local search: try swapping elements to improve size."""
        best_arr = arr[:]
        best_sums = sums.copy()
        best_size = len(arr)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Try removing one element and adding a better one
            base_set = {1, 2, 5, 11}
            non_base = [x for x in arr if x not in base_set]
            
            if not non_base:
                break
            
            for x in random.sample(non_base, min(len(non_base), 5)):
                # Remove x
                arr2 = [y for y in arr if y != x]
                sums2 = get_sums(arr2)
                if sums2 is None:
                    continue
                
                # Try to add a new element
                for y in range(1, 285):
                    if y in arr2:
                        continue
                    if can_add(arr2, sums2, y):
                        arr2.append(y)
                        arr2.sort()
                        for z in arr2[:-1]:
                            sums2.add(y + z)
                        sums2.add(2 * y)
                        
                        if len(arr2) > best_size:
                            best_arr = arr2[:]
                            best_sums = sums2.copy()
                            best_size = len(arr2)
                        break
        
        return best_arr, best_sums
    
    # Initial greedy expansion
    current_arr, current_sums = greedy_expand(base, get_sums(base), range(12, 285))
    best_arr = current_arr[:]
    best_sums = current_sums.copy()
    best_size = len(current_arr)
    
    # Multiple restarts with randomized strategies
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Start from base with some random additions
        arr = base[:]
        sums = get_sums(base)
        
        # Randomly add some elements first
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        for x in candidates[:random.randint(0, 5)]:
            if can_add(arr, sums, x):
                arr.append(x)
                arr.sort()
                for y in arr[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        
        # Greedy expansion
        arr, sums = greedy_expand(arr, sums, range(1, 285))
        
        if len(arr) > best_size:
            best_arr = arr[:]
            best_sums = sums.copy()
            best_size = len(arr)
        
        # Local search
        arr, sums = local_search(arr, sums, max_iters=20)
        
        if len(arr) > best_size:
            best_arr = arr[:]
            best_sums = sums.copy()
            best_size = len(arr)
        
        # Kick and expand
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            arr, sums = kick(arr, sums)
            arr, sums = greedy_expand(arr, sums, range(1, 285))
            
            if len(arr) > best_size:
                best_arr = arr[:]
                best_sums = sums.copy()
                best_size = len(arr)
    
    # Final local search on best solution
    if time.time() < deadline - 5.0:
        best_arr, best_sums = local_search(best_arr, best_sums, max_iters=30)
    
    return best_arr
