import time
import random
import sys

# Set recursion limit just in case
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    """
    
    # 1. Precompute a high-quality seed set using a fast greedy approach.
    # We try multiple orderings (ascending, descending, random) to find a large valid set.
    # This set will serve as the "core" for local search.
    
    def get_sums(subset):
        sums = {0.0}
        for x in subset:
            recip = 1.0 / x
            new_sums = set()
            for s in sums:
                new_sums.add(s + recip)
            sums = sums.union(new_sums)
        return sums

    def is_valid(subset):
        if not subset:
            return True
        sums = get_sums(subset)
        return len(sums) == (1 << len(subset))

    def greedy_find_seed(order):
        """Greedy addition based on a specific element order."""
        current_set = []
        current_sums = {0.0}
        for elem in order:
            recip = 1.0 / elem
            new_part = set()
            collision = False
            for s in current_sums:
                ns = s + recip
                if ns in current_sums:
                    collision = True
                    break
                new_part.add(ns)
            if not collision:
                current_set.append(elem)
                current_sums = current_sums.union(new_part)
        return current_set, current_sums

    # Generate seeds from different orderings
    seeds = []
    
    # Ascending
    s_asc, sums_asc = greedy_find_seed(list(range(1, 23)))
    seeds.append((s_asc, sums_asc))
    
    # Descending
    s_desc, sums_desc = greedy_find_seed(list(range(22, 0, -1)))
    seeds.append((s_desc, sums_desc))
    
    # Random shuffles
    for _ in range(20):
        order = list(range(1, 23))
        random.shuffle(order)
        s_rand, sums_rand = greedy_find_seed(order)
        seeds.append((s_rand, sums_rand))
        
    # Pick the largest seed
    best_seed = max(seeds, key=lambda x: len(x[0]))
    current_set, current_sums = best_seed
    current_set_set = set(current_set)
    
    if not current_set:
        return [1]

    # 2. Local Search: Swap-based optimization
    # We attempt to swap an element in the current set with one outside the set.
    # A swap (remove r, add a) is valid if:
    #   1. r is in current_set
    #   2. a is NOT in current_set
    #   3. The new set of sums (after removing r's contribution and adding a's) has no collisions.
    
    # To efficiently check validity of a swap:
    # Let S be the set of sums for current_set.
    # Let r be the element to remove.
    # Let a be the element to add.
    # The sums without r are obtained by "un-adding" r.
    # If S is the set of sums for {e1, ..., ek}, and we remove ek (recip rk),
    # the remaining sums S' = S \ {s + rk | s in S'}.
    # This is a bit tricky to compute directly in O(|S|).
    
    # Alternative:
    # We can maintain the sums for the set WITHOUT the last added element? No, swaps are arbitrary.
    
    # Let's use a simpler but slightly slower validity check for swaps, 
    # relying on the fact that |S| is at most 2^12 ~ 4096.
    
    def get_sums_without_and_add(sums, remove_elem, add_elem):
        """
        Given the full set of sums `sums` for a set containing `remove_elem`,
        compute the set of sums for the set with `remove_elem` replaced by `add_elem`.
        
        This function assumes `sums` contains all 2^k sums.
        It reconstructs the sums for the set without `remove_elem`, then adds `add_elem`.
        """
        # Step 1: Recover sums without remove_elem
        # Let R = 1/remove_elem
        # Let S_full be the input set.
        # Let S_no_r be the set of sums for the set without remove_elem.
        # S_full = S_no_r U (S_no_r + R)
        # Since the set is dissociated, S_no_r and S_no_r + R are disjoint.
        # We can identify S_no_r by picking any element x in S_full.
        # If x - R is in S_full, then x is in S_no_r + R, so x - R is in S_no_r.
        # If x - R is NOT in S_full, then x is in S_no_r.
        
        # Actually, a simpler way:
        # Pick an arbitrary element from the set of elements that formed `sums`.
        # But we don't have the list of elements easily accessible in this helper.
        
        # Let's just recompute from scratch for the candidate set.
        # Since |S| is small, recomputing is O(|S| * k).
        # For |S|=4096, k=12, this is ~50k ops.
        # We might do 100 swaps per iteration. 5M ops. Fast enough.
        pass

    # Better approach for local search:
    # Maintain the list of elements in `current_set`.
    # For each candidate swap (r, a):
    #   new_set = current_set - {r} + {a}
    #   Check if new_set is valid.
    
    # To speed up validity check, we can use the property:
    # A set is valid iff all subset sums are distinct.
    
    def check_validity_fast(subset):
        """
        Checks if all subset sums of reciprocals in `subset` are distinct.
        Returns True if valid, False otherwise.
        """
        sums = {0.0}
        for x in subset:
            recip = 1.0 / x
            new_sums = set()
            for s in sums:
                ns = s + recip
                new_sums.add(ns)
            # Check collision
            if new_sums & sums:
                return False
            sums = sums.union(new_sums)
        return True

    # Local Search Loop
    # We try to improve the size of the set, or maintain size while exploring the space.
    # Since we start with a large seed, we primarily want to find a set of size K+1.
    # If we can't find K+1, we return K.
    
    # Let's try to find a set of size len(current_set) + 1.
    # We can do this by trying to add an element? No, adding an element increases size by 1.
    # If we just add an element, it might collide.
    # So we need to swap: remove one, add two? No, that changes size by +1.
    # Swap: remove r, add a (size stays same).
    # To increase size: add a. If collision, try to fix by swapping?
    
    # Strategy:
    # 1. Try to add any element from [1..22] \ current_set.
    #    If valid, return the new set (size + 1).
    # 2. If no single addition works, try swaps to reach a state where an addition works?
    #    This is complex.
    
    # Simpler Local Search:
    # Iterate through all possible pairs (r in current_set, a in [1..22]\current_set).
    # If swapping r with a results in a valid set, perform the swap.
    # This explores the space of sets of the same size.
    # Periodically, try to add an element. If successful, we've increased size.
    # If we find a valid set of size K+1, we can stop (or continue to find K+2).
    
    # Let's implement a hill-climbing / simulated annealing style search.
    # Objective: Maximize size.
    # Moves:
    #   1. Add an element (if valid).
    #   2. Remove an element (if valid, though reduces size - only use if stuck or for diversification).
    #   3. Swap an element (if valid).
    
    # We will prioritize moves that increase size.
    # If no size-increasing move is found, we do random swaps to escape local minima.
    
    current_size = len(current_set)
    best_size_found = current_size
    best_set_found = list(current_set)
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    
    # To speed up collision checks, we maintain the set of sums.
    # But updating sums after a swap is hard.
    # Given the small size (<=12), recomputing sums for a candidate set is fast.
    
    # Let's define a function to try a specific move.
    
    def try_add(elem):
        """Try to add elem to current_set. Return new set if valid, else None."""
        if elem in current_set_set:
            return None
        new_set = list(current_set) + [elem]
        if check_validity_fast(new_set):
            return new_set
        return None

    def try_swap(r, a):
        """Try to swap r (in current_set) with a (not in current_set). Return new set if valid, else None."""
        if r not in current_set_set or a in current_set_set:
            return None
        new_set = [x for x in current_set if x != r]
        new_set.append(a)
        if check_validity_fast(new_set):
            return new_set
        return None

    # Main search loop
    iterations = 0
    max_iterations = 50000 # Cap on iterations
    
    while _time_left() > 2.0 and iterations < max_iterations:
        # 1. Try to add an element
        added = False
        candidates_to_add = [i for i in range(1, 23) if i not in current_set_set]
        random.shuffle(candidates_to_add)
        
        for a in candidates_to_add:
            new_set = try_add(a)
            if new_set is not None:
                current_set = new_set
                current_set_set = set(current_set)
                current_size += 1
                if current_size > best_size_found:
                    best_size_found = current_size
                    best_set_found = list(current_set)
                added = True
                break # Restart the loop with the new set
                
        if added:
            continue
            
        # 2. Try to swap (to escape local minima or find a better configuration for future additions)
        # We pick a random element to remove and a random element to add.
        # Or we try all swaps?
        # With size ~12, there are ~12 * 10 = 120 possible swaps.
        # Checking 120 swaps, each taking ~50k ops, is ~6M ops.
        # This might be too slow per iteration if we do many iterations.
        # Let's try a random subset of swaps.
        
        swaps_tried = 0
        swap_found = False
        
        # Try a few random swaps
        for _ in range(10):
            if _time_left() < 2.0:
                break
            r = random.choice(current_set)
            candidates_to_add = [i for i in range(1, 23) if i not in current_set_set]
            if not candidates_to_add:
                break
            a = random.choice(candidates_to_add)
            
            new_set = try_swap(r, a)
            if new_set is not None:
                current_set = new_set
                current_set_set = set(current_set)
                swap_found = True
                break
        
        if not swap_found:
            # If no random swap worked, try a systematic sweep of all swaps?
            # This is expensive. Let's just do a few more random restarts of the swap logic.
            # Or break if we've been stuck for a while?
            # For now, just continue with random swaps.
            pass
        
        iterations += 1
        
    # Return the best set found
    return best_set_found

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    sums = {0.0}
    for x in s:
        recip = 1.0 / x
        new_sums = {ss + recip for ss in sums}
        if new_sums & sums:
            print("INVALID!")
        sums = sums.union(new_sums)
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")
