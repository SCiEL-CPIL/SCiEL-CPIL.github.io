import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 120s. We stop at 115s to be safe.
    deadline = start_time + 115.0
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Initial state: just the fixed elements
    # We could start with a greedy expansion to get a better starting point
    def greedy_expand(initial_list, fixed_set):
        a = list(initial_list)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        # Try to add elements greedily
        changed = True
        while changed:
            changed = False
            for x in range(1, 285):
                if x in set(a):
                    continue
                # Check if x can be added
                new_sums = {x + y for y in a} | {2 * x}
                if len(new_sums) == len(a) + 1 and not (new_sums & sums):
                    a.append(x)
                    sums |= new_sums
                    changed = True
                    # Restart loop to pick up new opportunities
                    break
        return a, sums

    # Get a good initial guess using greedy
    current_set, current_sums = greedy_expand(fixed_elements, fixed_set)
    best_set = current_set[:]
    best_size = len(best_set)
    
    # Pre-compute all possible sums for quick checking? 
    # Actually, maintaining the set of sums is efficient enough for N~20.
    
    # Helper to check if a candidate x can be added to the current set
    def can_add(x, a_list, sums_set):
        if x in set(a_list):
            return False
        new_sums = {x + y for y in a_list} | {2 * x}
        if len(new_sums) != len(a_list) + 1:
            return False
        if new_sums & sums_set:
            return False
        return True

    # Main search loop: Randomized Local Search with Kicks
    # We will try to add elements. If stuck, we kick (remove) some elements.
    
    # To speed up "find an element to add", we can maintain a list of candidates.
    # But since the range is small (284), iterating through 1..284 is fast (284 checks).
    
    while time.time() < deadline:
        # 1. Try to add elements
        added = True
        while added and time.time() < deadline:
            added = False
            # Randomize the order of candidates to avoid bias
            candidates = [x for x in range(1, 285) if x not in set(current_set)]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, current_set, current_sums):
                    # Add x
                    current_set.append(x)
                    # Update sums
                    for y in current_set[:-1]:
                        current_sums.add(x + y)
                    current_sums.add(2 * x)
                    added = True
                    # Update best if improved
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                    # Break to restart candidate shuffling
                    break
        
        # 2. If we couldn't add anything, we are in a local optimum.
        # Check if we are done or need to kick.
        if not added:
            # If we have no non-fixed elements to kick, we are stuck at the base size.
            # But usually we can kick.
            non_fixed = [x for x in current_set if x not in fixed_set]
            
            if not non_fixed:
                # We are at the base size. We can't kick. 
                # We might need to try different random seeds or restarts, 
                # but with a fixed deadline, we just exit or wait.
                # Since we can't improve, we break.
                break
            
            # Perform a kick: remove 1 or 2 random non-fixed elements
            # Removing too many might be too costly, removing too few might not help.
            # Let's try removing 1 element first. If that fails repeatedly, try 2.
            # Simple strategy: remove 1 random non-fixed element.
            
            # To avoid oscillating between the same states, we can track recent states
            # but for now, simple random kick is effective.
            
            k = 1
            if random.random() < 0.2:
                k = 2
            if random.random() < 0.05:
                k = 3
            
            # Select elements to remove
            to_remove = random.sample(non_fixed, min(k, len(non_fixed)))
            
            # Remove them
            for x in to_remove:
                current_set.remove(x)
                # Remove sums involving x
                # This is O(N) to filter sums, which is fine.
                # Sums to remove: x+y for all y in current_set (after removal? No, before removal includes x)
                # Actually, it's easier to rebuild sums if N is small, or carefully subtract.
                # Let's rebuild sums for simplicity and correctness, N is small (<20).
            
            # Rebuild sums
            current_sums = set()
            for i, x in enumerate(current_set):
                for y in current_set[i:]:
                    current_sums.add(x + y)
                    
    return best_set

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found size: {len(res)}")
    # Verify
    s = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            sm = res[i] + res[j]
            if sm in s:
                valid = False
            s.add(sm)
    print(f"Valid: {valid}")
    print(f"Contains fixed: {set([1,2,5,11]).issubset(set(res))}")
