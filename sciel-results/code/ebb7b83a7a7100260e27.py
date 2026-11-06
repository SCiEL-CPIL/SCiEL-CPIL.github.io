import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    def is_valid_addition(reachable, x):
        """Check if x can be added given current reachable sums."""
        return (200 - x) not in reachable

    def update_reachable(reachable, x):
        """Update reachable sums after adding x."""
        new_sums = set()
        for s in reachable:
            ns = s + x
            if ns <= 200:
                new_sums.add(ns)
        return reachable | new_sums

    def greedy_build(order, reachable_init=None):
        """Build a set using greedy addition in given order."""
        if reachable_init is None:
            reachable = {0}
        else:
            reachable = set(reachable_init)
        selected = []
        for x in order:
            if is_valid_addition(reachable, x):
                selected.append(x)
                reachable = update_reachable(reachable, x)
        return selected, reachable

    def local_search_improve(S, reachable):
        """Improve set S by adding elements from complement."""
        S_set = set(S)
        all_elements = set(range(1, 101))
        complement = list(all_elements - S_set)
        improved = True
        current_S = list(S)
        current_reachable = set(reachable)
        
        while improved:
            improved = False
            random.shuffle(complement)
            for x in complement:
                if is_valid_addition(current_reachable, x):
                    current_S.append(x)
                    S_set.add(x)
                    current_reachable = update_reachable(current_reachable, x)
                    complement.remove(x)
                    improved = True
        return current_S, current_reachable

    def local_search_with_swaps(S, reachable, num_swaps=50):
        """Try to improve by swapping elements."""
        S_set = set(S)
        all_elements = set(range(1, 101))
        complement = list(all_elements - S_set)
        current_S = list(S)
        current_reachable = set(reachable)
        improved = True
        
        while improved and len(complement) > 0:
            improved = False
            # Try to find a swap that increases size or maintains size but allows future additions
            # Simple heuristic: Remove one element, try to add two from complement.
            # If we can add 2, size increases by 1.
            
            # To keep it fast, we just try random removals and see if we can add more.
            # Actually, a simpler and effective strategy is "Kick-out":
            # Pick a random element to remove, then greedily add from complement.
            
            if time.time() > deadline - 5:
                break
                
            # Pick a candidate to remove
            # Prefer removing smaller elements? Or random?
            # Removing larger elements frees up "sum space" for more small elements.
            # But we want to maximize cardinality, so small elements are valuable.
            # Let's try removing a random element.
            
            if not current_S:
                break
                
            remove_idx = random.randint(0, len(current_S) - 1)
            elem_to_remove = current_S[remove_idx]
            
            # Temporarily remove
            current_S.pop(remove_idx)
            S_set.discard(elem_to_remove)
            
            # Recalculate reachable? 
            # Removing an element from reachable set is hard (not invertible easily).
            # We must rebuild reachable from scratch for the new set.
            # This is O(N*T) = 100*200 = 20000 ops. Fast enough.
            
            new_reachable = {0}
            for x in current_S:
                new_reachable = update_reachable(new_reachable, x)
            
            # Now try to add elements from complement (which now includes elem_to_remove)
            # We want to see if we can get a larger set.
            # Greedily add from complement.
            
            new_S, new_reachable = greedy_build(complement, new_reachable)
            
            # Combine
            final_S = current_S + new_S
            final_reachable = new_reachable
            
            # Check if final_S is larger
            if len(final_S) > len(S):
                S = final_S
                reachable = final_reachable
                S_set = set(S)
                complement = list(all_elements - S_set)
                improved = True
            else:
                # Revert removal if it didn't help immediately
                # We don't strictly need to revert if we track state, 
                # but for simplicity in this loop, we just don't update S if not improved.
                # The next iteration will pick a different removal.
                pass

        return S, reachable

    best_set = []
    best_size = -1
    best_reachable = {0}

    # Strategy 1: Standard Greedy (Descending)
    order = list(range(100, 0, -1))
    S, reachable = greedy_build(order)
    S, reachable = local_search_improve(S, reachable)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()
        best_reachable = reachable.copy()

    if time.time() > deadline:
        return best_set if best_set else [100]

    # Strategy 2: Randomized Greedy + Local Search
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        S, reachable = greedy_build(order)
        S, reachable = local_search_improve(S, reachable)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_reachable = reachable.copy()

    if time.time() > deadline:
        return best_set if best_set else [100]

    # Strategy 3: Local Search with Swaps on Best Set
    # Try to improve the best set found so far
    if best_set:
        S, reachable = local_search_with_swaps(best_set, best_reachable)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_reachable = reachable.copy()

    # Strategy 4: More aggressive random restarts with swaps
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        S, reachable = greedy_build(order)
        S, reachable = local_search_improve(S, reachable)
        S, reachable = local_search_with_swaps(S, reachable)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_reachable = reachable.copy()

    if not best_set:
        best_set = [100]
        
    return best_set
# EVOLVE-BLOCK-END
