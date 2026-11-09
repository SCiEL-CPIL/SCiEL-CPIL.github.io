import time
import random
import sys

N = 1000
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin before 150s kill
    
    # Data structures - use set for O(1) membership, list for iteration
    S_set = set()
    S_list = []
    counts = [0] * (MAX_SUM + 1)
    # Track which elements contribute to each sum for efficient blocker finding
    # sum_contributors[s] = set of elements in S that form a pair summing to s
    sum_contributors = [set() for _ in range(MAX_SUM + 1)]
    
    def add_element(x):
        """Add x to S, updating counts and contributors."""
        S_set.add(x)
        S_list.append(x)
        counts[2 * x] += 1
        sum_contributors[2 * x].add(x)
        for y in S_set:
            if y != x:
                s = x + y
                counts[s] += 1
                sum_contributors[s].add(x)
                sum_contributors[s].add(y)
                
    def remove_element(x):
        """Remove x from S, updating counts and contributors."""
        S_set.discard(x)
        S_list.remove(x)
        counts[2 * x] -= 1
        sum_contributors[2 * x].discard(x)
        for y in S_set:
            s = x + y
            counts[s] -= 1
            sum_contributors[s].discard(x)
            sum_contributors[s].discard(y)
                
    def try_add(x):
        """Try to add x to S. Returns True if successful."""
        if x in S_set:
            return False
        if counts[2 * x] >= 2:
            return False
        for y in S_set:
            s = x + y
            if counts[s] >= 2:
                return False
        add_element(x)
        return True

    def get_blockers(c):
        """Get elements in S that block adding c. Uses sum_contributors for efficiency."""
        blockers = set()
        
        # Check 2*c
        if counts[2 * c] >= 2:
            blockers.update(sum_contributors[2 * c])
                
        # Check c + y for all y in S
        for y in S_set:
            s = c + y
            if counts[s] >= 2:
                blockers.update(sum_contributors[s])
                    
        return blockers

    def is_valid_swap(a, c):
        """
        Check if swapping a (in S) with c (not in S) is valid.
        a is removed, c is added.
        """
        if a == c:
            return False
        if a not in S_set or c in S_set:
            return False
            
        # Build deltas efficiently
        deltas = {}
        
        # Subtract a's contributions
        deltas[2 * a] = deltas.get(2 * a, 0) - 1
        for y in S_set:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
                
        # Add c's contributions
        deltas[2 * c] = deltas.get(2 * c, 0) + 1
        for y in S_set:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
                
        # Check if any count exceeds 2
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
        
        return True

    def perform_swap(a, c):
        """Perform the swap a->c. Assumes is_valid_swap(a,c) is True."""
        remove_element(a)
        add_element(c)

    def greedy_build(order):
        """Greedy build given a specific order."""
        for x in order:
            if time.time() - start_time > time_limit:
                return
            try_add(x)

    def reset_state():
        """Reset all state to empty."""
        nonlocal S_set, S_list, counts, sum_contributors
        S_set = set()
        S_list = []
        counts = [0] * (MAX_SUM + 1)
        sum_contributors = [set() for _ in range(MAX_SUM + 1)]

    def save_state():
        """Save current state."""
        return (S_set.copy(), S_list.copy(), counts.copy(), 
                [s.copy() for s in sum_contributors])

    def restore_state(state):
        """Restore state from saved tuple."""
        nonlocal S_set, S_list, counts, sum_contributors
        S_set, S_list, counts, sum_contributors = state

    # --- Step 1: Initialization with multiple strategies ---
    rng = random.Random(42)
    best_size = 0
    best_state = None
    
    # Try multiple greedy orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: rng.random()),
        # Try a few more random orders
        sorted(range(1, N + 1), key=lambda x: rng.random()),
        sorted(range(1, N + 1), key=lambda x: rng.random()),
    ]
    
    for order in orders:
        if time.time() - start_time > time_limit - 60:
            break
        reset_state()
        greedy_build(order)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_state = save_state()
    
    # Restore best initialization
    if best_state is not None:
        restore_state(best_state)
    else:
        # Fallback: ascending greedy
        reset_state()
        greedy_build(list(range(1, N + 1)))
        best_state = save_state()
        best_size = len(S_set)

    # --- Step 2: Local Search ---
    no_improve_count = 0
    last_improve_time = start_time
    stagnation_threshold = 3.0  # seconds
    
    while time.time() - start_time < time_limit:
        # Check for stagnation
        if time.time() - last_improve_time > stagnation_threshold:
            # Perturb: remove 3-5 random elements to escape local optimum
            num_remove = rng.randint(3, 5)
            for _ in range(num_remove):
                if S_set:
                    a = rng.choice(list(S_set))
                    remove_element(a)
            last_improve_time = time.time()
            no_improve_count = 0
            continue
            
        # Pick a random candidate c not in S
        c = rng.randint(1, N)
        if c in S_set:
            continue
            
        # Try to add c
        if try_add(c):
            no_improve_count = 0
            last_improve_time = time.time()
            if len(S_set) > best_size:
                best_size = len(S_set)
                best_state = save_state()
            continue
            
        # If add failed, try targeted swap
        blockers = get_blockers(c)
        
        if blockers:
            # Try to swap with a random blocker
            blocker_list = list(blockers)
            rng.shuffle(blocker_list)
            
            swapped = False
            for a in blocker_list:
                if is_valid_swap(a, c):
                    perform_swap(a, c)
                    swapped = True
                    no_improve_count = 0
                    last_improve_time = time.time()
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                        best_state = save_state()
                    break
            
            if not swapped:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
    # Return the best found
    if best_state is not None:
        S_set, S_list, counts, sum_contributors = best_state
    
    return list(S_set)
