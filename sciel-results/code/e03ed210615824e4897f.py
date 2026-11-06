import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    base = [1, 2, 5, 11]
    base_set = set(base)

    # Calculate initial sums
    initial_sums = set()
    for i in range(len(base)):
        for j in range(i, len(base)):
            initial_sums.add(base[i] + base[j])

    # Initial candidates: all numbers in 1..284 not in base
    all_numbers = list(range(1, 285))

    def is_valid_candidate(x, current_set, current_sums):
        """
        Check if x can be added to current_set without creating sum collisions.
        """
        new_sums = set()
        # Self sum
        s_self = 2 * x
        if s_self in current_sums:
            return False
        new_sums.add(s_self)

        # Sums with existing elements
        for y in current_set:
            s = x + y
            if s in current_sums:
                return False
            new_sums.add(s)

        return True

    def count_blocked(x, current_set, current_sums):
        """Count how many other candidates would be blocked if x is added."""
        blocked = 0
        for y in current_set:
            s = x + y
            if s not in current_sums:
                # Check if this sum would block any other candidate
                for z in range(1, 285):
                    if z not in current_set and z != x:
                        # z would be blocked if 2*z == s or z + y' == s for some y' in current_set
                        # Simplified: just count potential conflicts
                        pass
        # Simpler: just use a heuristic based on position
        return 0

    # Filter initial candidates
    candidates = []
    for x in all_numbers:
        if x in base_set:
            continue
        if is_valid_candidate(x, base_set, initial_sums):
            candidates.append(x)

    candidates.sort()

    best_solution = list(base)
    best_size = len(base)

    def greedy_fill(cands, current_set, current_sums, time_limit):
        """Greedily add candidates, with option to backtrack."""
        nonlocal best_solution, best_size
        
        i = 0
        while i < len(cands) and time.time() < time_limit:
            x = cands[i]
            if is_valid_candidate(x, current_set, current_sums):
                # Try adding x
                new_set = set(current_set)
                new_set.add(x)
                new_sums = set(current_sums)
                new_sums.add(2 * x)
                for y in current_set:
                    new_sums.add(x + y)
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_solution = list(new_set)
                
                # Continue with this choice
                current_set = new_set
                current_sums = new_sums
                
                # Filter out candidates that are no longer valid
                cands = [c for c in cands[i+1:] if is_valid_candidate(c, current_set, current_sums)]
                i = 0
            else:
                i += 1

        return current_set, current_sums

    def local_search(current_set, current_sums, time_limit):
        """Try to improve the solution by swapping elements."""
        nonlocal best_solution, best_size
        
        current_list = sorted(current_set)
        current_size = len(current_list)
        
        for _ in range(50):  # Limited number of iterations
            if time.time() >= time_limit:
                break
            
            # Try removing a random element and adding a different one
            if current_size <= 4:
                break
                
            idx = random.randint(0, current_size - 1)
            removed = current_list[idx]
            
            # Build set without removed
            test_set = set(current_list)
            test_set.remove(removed)
            test_sums = set()
            for i in range(len(test_set)):
                for j in range(i, len(test_set)):
                    test_sums.add(list(test_set)[i] + list(test_set)[j])
            
            # Try to add a better candidate
            added = None
            for x in candidates:
                if x in test_set or x == removed:
                    continue
                if is_valid_candidate(x, test_set, test_sums):
                    # Check if this improves something (for now, just accept if valid)
                    added = x
                    break
            
            if added is not None:
                test_set.add(added)
                test_sums.add(2 * added)
                for y in current_list:
                    if y != removed:
                        test_sums.add(added + y)
                
                if len(test_set) > best_size:
                    best_size = len(test_set)
                    best_solution = list(test_set)
                
                current_set = test_set
                current_sums = test_sums
                current_list = sorted(current_list)
                current_size = len(current_list)

        return current_set, current_sums

    # Run multiple strategies
    restart_count = 0
    while time.time() < deadline:
        if restart_count % 3 == 0:
            # Pure greedy
            cands = candidates[:]
            random.shuffle(cands)
            current_set = set(base)
            current_sums = set(initial_sums)
            greedy_fill(cands, current_set, current_sums, deadline)
        elif restart_count % 3 == 1:
            # Greedy then local search
            cands = candidates[:]
            random.shuffle(cands)
            current_set = set(base)
            current_sums = set(initial_sums)
            current_set, current_sums = greedy_fill(cands, current_set, current_sums, deadline - 10)
            if time.time() < deadline - 5:
                local_search(current_set, current_sums, deadline)
        else:
            # Sorted greedy (deterministic)
            cands = sorted(candidates)
            current_set = set(base)
            current_sums = set(initial_sums)
            greedy_fill(cands, current_set, current_sums, deadline)
        
        restart_count += 1
        
        # If we reached size 20, we can stop early
        if best_size >= 20:
            break

    return sorted(best_solution)

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
