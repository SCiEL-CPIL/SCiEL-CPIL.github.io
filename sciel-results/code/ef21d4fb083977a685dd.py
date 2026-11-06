import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute LCM of 1..22
    # LCM(1..22) = 232792560
    L = 232792560
    weights = [L // i for i in range(1, 23)]
    
    # indices 0..21 correspond to numbers 1..22
    all_indices = list(range(22))
    
    def is_dissociated(indices):
        """
        Returns True if all subset sums of weights[i] for i in indices are distinct.
        Uses incremental set update to fail fast.
        """
        current_sums = {0}
        for i in indices:
            w = weights[i]
            # Generate new sums
            # To avoid creating a huge list if we already know it's bad,
            # we can check intersection
            new_sums = set()
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    return False
                new_sums.add(ns)
            current_sums.update(new_sums)
        return True

    def get_collision_count(indices):
        """
        Returns the number of duplicate sums.
        0 means dissociated.
        """
        current_sums = set([0])
        total_expected = 0
        for i in indices:
            w = weights[i]
            new_sums = set()
            for s in current_sums:
                ns = s + w
                new_sums.add(ns)
            # If there are duplicates, len(current_sums | new_sums) < len(current_sums) + len(new_sums)
            # Actually, we just need to count how many new sums are already in current_sums
            # Or simpler: 
            union = current_sums | new_sums
            if len(union) < len(current_sums) + len(new_sums):
                 # There are collisions
                 pass
            current_sums = union
        # If dissociated, len(current_sums) should be 2^len(indices)
        return (1 << len(indices)) - len(current_sums)

    best_set = []
    best_size = 0
    
    # 1. Greedy Initialization with randomization
    # Try a few random orderings
    for _ in range(50):
        random.shuffle(all_indices)
        current_set = []
        for i in all_indices:
            test_set = current_set + [i]
            if is_dissociated(test_set):
                current_set.append(i)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # 2. Local Search to improve best_set
    # We try to add elements or swap elements to increase size
    # Since we want max size, we primarily try to ADD.
    # If we can't add, we try SWAP to see if we can reach a state where we CAN add,
    # but swapping doesn't increase size. So we should only swap if it helps us add later?
    # Actually, just trying to add is a local optimum. 
    # To escape, we can do a "kick": remove a random element, try to add a different one.
    
    # Let's try to find sets of size K > best_size using random restarts + local repair
    # But local repair for "maximize size" is tricky if we start with a set of size K and it's invalid.
    # Better: Start with a valid set (from greedy) and try to expand.
    
    # Let's refine the expansion:
    # While time remains:
    #   Try to add a random element not in set. If valid, accept.
    #   If no element can be added (all 22-|S| fail), try a swap:
    #      Pick a to remove, b to add. If valid, accept.
    #      This keeps size same, but might allow future additions.
    
    current_set = best_set[:]
    current_size = best_size
    
    while time.time() < deadline:
        # Try to add
        added = False
        candidates_to_add = [i for i in all_indices if i not in current_set]
        random.shuffle(candidates_to_add)
        for i in candidates_to_add:
            test_set = current_set + [i]
            if is_dissociated(test_set):
                current_set.append(i)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                added = True
                break
        
        if not added:
            # Try to swap to escape local optimum
            # Pick a random element to remove
            if current_size > 0:
                remove_idx = random.choice(range(current_size))
                a = current_set[remove_idx]
                candidates_to_add = [i for i in all_indices if i not in current_set]
                random.shuffle(candidates_to_add)
                swapped = False
                for b in candidates_to_add:
                    # Remove a, add b
                    new_set = current_set.copy()
                    new_set[remove_idx] = b
                    # Check if new_set is dissociated
                    if is_dissociated(new_set):
                        current_set = new_set
                        # Size is same, so best_size doesn't change unless we add later
                        swapped = True
                        break
                if not swapped:
                    # If no swap worked, maybe kick harder?
                    # Or just restart the local search on a new random greedy set?
                    # Let's do a quick random restart of the greedy phase
                    random.shuffle(all_indices)
                    current_set = []
                    current_size = 0
                    for i in all_indices:
                        test_set = current_set + [i]
                        if is_dissociated(test_set):
                            current_set.append(i)
                            current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]

    return [i + 1 for i in best_set]
# EVOLVE-BLOCK-END
