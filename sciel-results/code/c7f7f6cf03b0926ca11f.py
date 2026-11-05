import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    # Precompute reciprocals as floats for fast comparison
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # State representation
    in_set = [False] * (n + 1)
    active_list = []
    active_set = set()
    
    def get_sum():
        s = 0.0
        for x in active_list:
            s += recip[x]
        return s

    def can_add(x):
        """
        Check if x can be added to the current set.
        Uses active_list for iteration and active_set for lookup.
        """
        # Case 1: x is middle. a + c = 2x => c = 2x - a
        # Case 2: x is endpoint. a + x = 2m => m = (a+x)/2 or x + a = 2m
        # The condition "No 3-AP" means:
        # 1. No a, c in S such that a+c=2x
        # 2. No m in S such that 2m-x in S (which implies x, m, 2m-x is AP? No.
        #    If m in S and y=2m-x in S, then x, m, y is AP?
        #    m-x = y-m => 2m = x+y. Yes.
        #    So we check if 2m-x is in S.
        
        for a in active_list:
            # Check if x is the middle term: a, x, c => c = 2x - a
            c = 2 * x - a
            if c != x and 1 <= c <= n:
                if c in active_set:
                    return False
            
            # Check if x is an endpoint: a, m, x => m = (a+x)/2
            # This is equivalent to checking if there exists m in S such that 2m - a = x?
            # No, the AP is (u, v, w).
            # If x is u: x, v, w => 2v = x+w => w = 2v-x.
            # So for any v in S, check if 2v-x is in S.
            # This covers x being the first element.
            # By symmetry, x being the last element is covered by the same check?
            # If x is w: u, v, x => 2v = u+x => u = 2v-x.
            # So checking "for any v in S, is 2v-x in S?" covers x being an endpoint.
            # And checking "for any a in S, is 2x-a in S?" covers x being the middle.
            
            v = a # Reusing variable name for clarity in loop
            w = 2 * v - x
            if w != x and 1 <= w <= n:
                if w in active_set:
                    return False
                    
        return True

    def add_element(x):
        in_set[x] = True
        active_list.append(x)
        active_set.add(x)

    def remove_element(x):
        in_set[x] = False
        active_set.remove(x)
        # Remove from list
        # List removal is O(N), but N is small.
        # We can just filter or use a set for list?
        # For speed, we might just rebuild list in greedy_fill if needed, 
        # or use a linked list? 
        # Given N=256, list.remove is fine.
        try:
            active_list.remove(x)
        except ValueError:
            pass

    def greedy_fill():
        """
        Add all valid elements in increasing order.
        """
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not in_set[x]:
                    if can_add(x):
                        add_element(x)
                        changed = True

    def local_search():
        """
        Perform local search by trying to remove each element and refilling.
        Returns the best set found.
        """
        improved = True
        current_sum = get_sum()
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_trial_list = active_list[:]
            best_trial_sum = current_sum
            
            # Iterate over a copy of the list as we might modify it
            # Actually, we should try removing each element from the CURRENT best set
            # So we iterate over the elements of the current best set.
            # We need to be careful: if we accept a move, the set changes.
            # Standard approach: iterate 1..n, if element is in set, try removing it.
            
            for r in range(1, n + 1):
                if not in_set[r]:
                    continue
                if time.time() > deadline - 2.0:
                    break
                
                # Save state
                orig_list = active_list[:]
                orig_set = active_set.copy()
                orig_in_set = in_set[:]
                orig_sum = current_sum

                # Remove r
                remove_element(r)
                
                # Refill
                greedy_fill()
                
                new_sum = get_sum()
                
                if new_sum > orig_sum + 1e-12:
                    # Accept
                    current_sum = new_sum
                    improved = True
                    # State is already updated to the new better one
                    # Break to restart the loop with the new set? 
                    # Or continue?
                    # If we continue, we are removing from the NEW set.
                    # That's fine.
                    # But we should probably restart to ensure we check all removals from the new base.
                    # However, for speed, just continuing is okay, but might miss some.
                    # Let's break to be safe and thorough.
                    break
                else:
                    # Revert
                    active_list[:] = orig_list
                    active_set.clear()
                    active_set.update(orig_set)
                    in_set[:] = orig_in_set

        return current_sum

    # 1. Greedy Initialization
    greedy_fill()
    best_sum = get_sum()
    best_state = {
        'in_set': in_set[:],
        'active_list': active_list[:],
        'active_set': active_set.copy()
    }
    
    # 2. Initial Local Search
    current_sum = local_search()
    if current_sum > best_sum:
        best_sum = current_sum
        best_state = {
            'in_set': in_set[:],
            'active_list': active_list[:],
            'active_set': active_set.copy()
        }

    # 3. Iterated Local Search (ILS)
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Perturb the current best state
        # We operate on the current active_list/set which represents the best state
        # Make sure we start from best_state
        
        # Restore best state to current
        in_set[:] = best_state['in_set']
        active_list[:] = best_state['active_list']
        active_set.clear()
        active_set.update(best_state['active_set'])
        
        # Random Perturbation: Remove k random elements
        # k depends on time left? Or fixed?
        # Let's vary k.
        k = random.randint(2, 5)
        
        # Select k random elements to remove
        # Use a copy to avoid modification during selection
        current_elements = active_list[:]
        if len(current_elements) <= k:
            continue # Can't remove more than available
            
        # Shuffle and take first k
        random.shuffle(current_elements)
        to_remove = current_elements[:k]
        
        for r in to_remove:
            if in_set[r]: # Might have been removed by previous logic? No, we just copied.
                remove_element(r)
                
        # Refill
        greedy_fill()
        
        # Local Search on the perturbed set
        new_sum = local_search()
        
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_state = {
                'in_set': in_set[:],
                'active_list': active_list[:],
                'active_set': active_set.copy()
            }

    # Restore best state for final result
    in_set[:] = best_state['in_set']
    active_list[:] = best_state['active_list']
    active_set.clear()
    active_set.update(best_state['active_set'])
    
    # Final greedy fill just in case
    greedy_fill()
    
    result = active_list[:]
    result.sort() # Ensure sorted output, though not strictly required, good for determinism
    return result
