import time
import random
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if len(sums) < 2:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        for u in sum_list:
            if u * 2 > max_sum:
                break
            for v in range(u * 2, max_sum + 1, u):
                if v in sums:
                    return False
        return True

    def try_add_element(current_sums, c):
        """Try adding element c. Returns new_sums if valid, else None."""
        # Quick collision check
        # We want to know if any (s + c) is already in current_sums.
        # This is equivalent to checking if (current_sums - c) intersects current_sums.
        # But we can't just subtract a set.
        # Let's just build the new set. It's fast enough.
        
        new_sums = set(current_sums)
        for s in current_sums:
            ns = s + c
            if ns in new_sums:
                return None # Collision: two subsets have same sum
            new_sums.add(ns)
        new_sums.add(c)
        
        if not is_valid(new_sums):
            return None
            
        return new_sums

    def get_subset_sums(elements):
        """Return set of all nonempty subset sums."""
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def compute_conflict_matrix(elements):
        """
        Pre-compute which pairs of elements in the current set cannot coexist 
        in a superset with a new element? No, that's not right.
        
        Instead, let's look at the structure of the problem.
        The conflict is between the SET of sums.
        If we have a set S, and we want to add x, we fail if:
        1. Collision: (S + x) intersects S.
        2. Divisibility: New sums create a divisibility chain.
        
        For a more structural approach, consider the "Conflict Graph" of the *candidates*.
        Two candidates a, b conflict if they cannot be in the same set?
        No, that's not static. The validity depends on the whole set.
        
        However, we can use a "Bitset Accelerated" approach for the search.
        Let's stick to the current greedy + local search but make it smarter.
        
        The directive suggests: "Model the problem as a maximum independent set in a conflict graph... solve it using a branch-and-bound".
        This is a huge structural change. 
        Let's try a hybrid: 
        1. Use the current fast greedy to get a good seed.
        2. Use a specific local search move: "Swap one for one" but with a lookahead.
        
        Actually, let's look at the "Inspiration 3" which had a similar score.
        The key might be in the *order* of addition or the *choice* of candidates.
        
        Let's try a different heuristic for the greedy phase:
        Sort candidates by "flexibility" or "constraint".
        Large numbers are better.
        
        Let's refine the local search:
        Instead of random swaps, try to swap an element with another that is "closer" in value or has a similar divisibility profile?
        
        Let's try to implement a simple Branch and Bound for a small subset of candidates?
        No, 256 is too large for naive B&B.
        
        Let's focus on optimizing the existing loop.
        The current code does 2000 random restarts.
        Maybe we can reduce the number of restarts and spend more time on local search for the best found?
        
        Let's try a "Simulated Annealing" style local search on the set composition.
        State: Set of elements.
        Energy: -Size (we want to maximize size) + Penalty (if invalid, huge penalty).
        But we always keep the set valid.
        So we just want to maximize size.
        
        Move: Remove one, add one (size same).
        Acceptance: Always accept if valid? Or accept with probability?
        Since size is same, we need a reason to accept.
        Maybe "potential" = number of candidates that could be added?
        
        Let's try to calculate the "addable candidates" count for a set.
        If we swap, and the new set has more addable candidates, we accept.
        
        Let's implement this.
        """
        pass

    def count_addable(current, current_sums, all_candidates):
        """Count how many candidates can be added to current."""
        count = 0
        current_set = set(current)
        for c in all_candidates:
            if c in current_set:
                continue
            # Quick check: collision
            # We can approximate or do a fast check.
            # For now, just try to add.
            # This is expensive if done for all candidates.
            # But we only do this for local search steps, not every iteration.
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                count += 1
        return count

    best_set = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Large to Small)
    orderings = [
        list(range(256, 0, -1)),
        list(range(256, 128, -1)) + list(range(127, 0, -1)),
        [x for x in range(1, 257) if x % 2 == 1],
        [x for x in range(1, 257) if x % 2 == 0],
        # New ordering: Primes?
        [x for x in range(1, 257) if all(x % i != 0 for i in range(2, int(x**0.5)+1))],
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 5.0:
            break
        current = []
        current_sums = set()
        for c in ordering:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 2: Randomized Greedy with Restart
    all_candidates = list(range(1, 257))
    
    # Reduce restarts to allow more local search time?
    # Let's do 500 restarts, then spend the rest on local search.
    for attempt in range(500):
        if time.time() > deadline - 10.0: # Leave 10s for local search
            break
        
        current = []
        current_sums = set()
        
        candidates = all_candidates[:]
        random.shuffle(candidates)
        
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 3: Local Search with "Addable Candidates" Heuristic
    # We try to improve the best set found by swapping elements.
    # We accept a swap if it allows us to add more elements later.
    
    if best_size > 0 and time.time() < deadline - 2.0:
        current = best_set[:]
        current_sums = get_subset_sums(current)
        
        # Calculate initial addable count
        # This is expensive, but we only do it occasionally.
        # Let's just try to add first.
        
        # Loop for local search
        while time.time() < deadline - 1.0:
            # 1. Try to add any element
            added = False
            # Try candidates in random order? Or sorted by value?
            # Let's try a few random candidates.
            for _ in range(20):
                c = random.choice(all_candidates)
                if c in current:
                    continue
                new_sums = try_add_element(current_sums, c)
                if new_sums is not None:
                    current.append(c)
                    current_sums = new_sums
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = current[:]
                    added = True
                    break
            
            if added:
                continue
            
            # 2. If we couldn't add, try a swap
            # Pick a random element to remove
            idx = random.randint(0, len(current) - 1)
            old_elem = current[idx]
            reduced_set = current[:idx] + current[idx+1:]
            red_sums = get_subset_sums(reduced_set)
            
            # Try to replace old_elem with a new_elem
            # We want a new_elem that is "better".
            # How to define better?
            # Maybe one that allows adding another element?
            
            # Let's try a few random new elements.
            swapped = False
            for _ in range(10):
                new_elem = random.choice(all_candidates)
                if new_elem == old_elem or new_elem in current:
                    continue
                
                new_sums = try_add_element(red_sums, new_elem)
                if new_sums is not None:
                    # Valid swap
                    # Check if this new state allows adding an element
                    # We can do a quick check: try to add a random element to this new state
                    # If we can, accept the swap.
                    # If not, maybe accept with low probability?
                    # For now, accept if it leads to an add.
                    
                    # Try to add to the swapped state
                    can_add = False
                    for _ in range(5):
                        c2 = random.choice(all_candidates)
                        if c2 in current: # Note: current still has old_elem, but we are checking against the *hypothetical* new state
                             # Actually, we should check against the hypothetical new set.
                             # Let's just try to add to new_sums.
                             pass
                        # We need to check if c2 is in the hypothetical new set.
                        # Hypothetical set is reduced_set + [new_elem].
                        if c2 in reduced_set or c2 == new_elem:
                            continue
                        new_sums2 = try_add_element(new_sums, c2)
                        if new_sums2 is not None:
                            can_add = True
                            break
                    
                    if can_add:
                        # Accept swap and the add
                        current = reduced_set + [new_elem]
                        current_sums = new_sums
                        # Find the element to add
                        # We didn't store which one worked. Let's just try to add again in the next loop iteration.
                        # Or we can just accept the swap and let the next loop's "try to add" handle it.
                        swapped = True
                        break
            
            if not swapped:
                # If no good swap found, maybe try a random valid swap to escape local minimum?
                # Or just break?
                # Let's try a random valid swap.
                for _ in range(5):
                    new_elem = random.choice(all_candidates)
                    if new_elem == old_elem or new_elem in current:
                        continue
                    new_sums = try_add_element(red_sums, new_elem)
                    if new_sums is not None:
                        current = reduced_set + [new_elem]
                        current_sums = new_sums
                        swapped = True
                        break

            if not swapped:
                # Stuck?
                # Try a different element to remove?
                # We just looped and tried.
                # Maybe we should break out of this local search phase?
                # Or try a different strategy.
                # Let's just continue the loop, it will pick a different idx next time.
                pass

    # Ensure types are correct
    return [int(x) for x in best_set]

if __name__ == "__main__":
    # Quick self-test
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {result}")
    
    # Verify
    if result:
        sums = set()
        n = len(result)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += result[i]
            if s in sums:
                print("INVALID: Duplicate sum")
                break
            sums.add(s)
        
        sum_list = sorted(sums)
        valid = True
        for u in sum_list:
            if u * 2 > sum_list[-1]:
                break
            for v in range(u * 2, sum_list[-1] + 1, u):
                if v in sums:
                    print(f"INVALID: {u} divides {v}")
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            print("VALID")
        else:
            print("INVALID")
    else:
        print("Empty set")
