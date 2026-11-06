"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # The prescribed elements
    prescribed = [1, 2, 5, 11]
    
    # Helper to check if a set is valid (pairwise sums distinct)
    def is_valid(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                total = s[i] + s[j]
                if total in sums:
                    return False
                sums.add(total)
        return True
    
    # Helper to compute sum set
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    # Helper to count conflicts (number of duplicate sums)
    def count_conflicts(s):
        sums = set()
        conflicts = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                total = s[i] + s[j]
                if total in sums:
                    conflicts += 1
                else:
                    sums.add(total)
        return conflicts

    # 1. Greedy initialization (fast, gives a decent starting point)
    # We use a randomized greedy to explore different paths
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        a = prescribed[:]
        sums = get_sums(a)
        # Try to add elements in a random order or sequential?
        # Sequential is deterministic. Randomized greedy can find different valid sets.
        # Let's do a mix: try sequential first, then if we want diversity, we can randomize candidates.
        # For now, standard sequential greedy is very fast and reliable.
        for x in range(12, 285):
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & sums:
                a.append(x)
                sums |= new
        return a

    current = greedy_init()
    best_set = current[:]
    best_size = len(current)
    
    # 2. Local Search with Simulated Annealing to improve the set
    # Goal: Find a valid set of size > best_size.
    # Strategy: 
    #   - Start with a set of size K (e.g., best_size + 1).
    #   - If invalid, use SA to minimize conflicts.
    #   - If we reach 0 conflicts, we have a valid set of size K.
    #   - Try to increase K further.
    
    # To speed up SA, we can use a "delta" evaluation for moves.
    # But for simplicity and robustness within the time limit, we'll use a simpler approach:
    # We will run SA on a set of fixed size N.
    # We will try N = best_size + 1, then best_size + 2, etc.
    
    def local_search_sa(initial_set, target_size, time_limit):
        """
        Run SA to find a valid set of size target_size.
        initial_set is a list of size target_size (may have conflicts).
        Returns the best set found (hopefully valid).
        """
        s = sorted(initial_set)
        current_conflicts = count_conflicts(s)
        best_s = s[:]
        best_conflicts = current_conflicts
        
        T = 5.0
        cooling_rate = 0.9995
        
        while time.time() < time_limit:
            if current_conflicts == 0:
                return s
            
            # Move: swap two elements? Or remove one and add another?
            # Since size is fixed, we must swap.
            # Pick two random indices i, j. Swap s[i] and s[j].
            # Or pick one index i and a random candidate c not in s, replace s[i] with c.
            # The latter allows more exploration.
            
            idx_remove = random.randint(0, len(s) - 1)
            removed = s[idx_remove]
            
            # Pick a random candidate from 1..284 not in s
            # To be efficient, we can pick from a precomputed list of candidates
            # or just random.randint and check membership.
            # Membership check in a set is O(1).
            s_set = set(s)
            
            # We want to avoid picking 'removed' again
            candidate = random.randint(1, 284)
            if candidate in s_set:
                # Retry a few times or just skip
                # For speed, just skip
                continue
            
            # Construct new set
            new_s = s[:]
            new_s[idx_remove] = candidate
            new_s.sort()
            
            new_conflicts = count_conflicts(new_s)
            
            delta = new_conflicts - current_conflicts
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                s = new_s
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_s = s[:]
                    if best_conflicts == 0:
                        return best_s
            
            T *= cooling_rate
            if T < 0.01:
                break
        
        return best_s

    # Main loop: Try to increase size incrementally
    # Start with the greedy result.
    # Try to find a valid set of size best_size + 1.
    # If successful, update best_set and best_size, and try best_size + 2.
    
    target_size = best_size + 1
    
    while time.time() < deadline - 10 and target_size <= 20:
        # Initialize a set of size target_size
        # We can start with the current best_set (size best_size) and add a random element.
        # This ensures we keep the good structure of the previous solution.
        
        init_s = best_set[:]
        # Add a random element not in best_set
        candidates = [x for x in range(1, 285) if x not in init_s]
        if not candidates:
            break
        
        # Try multiple random initializations for this target size
        success = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Pick a random candidate to add
            # We might want to bias towards candidates that cause fewer conflicts?
            # For now, random.
            cand = random.choice(candidates)
            init_s = best_set + [cand]
            random.shuffle(init_s)
            
            # Run SA for a short time
            sa_time_limit = time.time() + 2.0 # 2 seconds per attempt
            found_s = local_search_sa(init_s, target_size, sa_time_limit)
            
            if count_conflicts(found_s) == 0:
                # Found a valid set of target_size
                best_set = found_s
                best_size = target_size
                success = True
                break
        
        if success:
            target_size += 1
        else:
            # Failed to extend further with current method
            # We could try a different strategy, e.g., restart from scratch with a larger target?
            # Or just break and return the best we have.
            # Given the time, breaking is safer.
            break

    # Final check: try greedy expansion on the best_set found
    # This is fast and might pick up 1-2 more elements if SA left some room.
    final_set = best_set[:]
    sums = get_sums(final_set)
    for x in range(1, 285):
        if x in final_set:
            continue
        new = {x + y for y in final_set} | {2 * x}
        if len(new) == len(final_set) + 1 and not new & sums:
            final_set.append(x)
            sums |= new
    
    if len(final_set) > best_size:
        best_set = final_set
        best_size = len(final_set)

    return best_set
# EVOLVE-BLOCK-END
