import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        Optimized: unroll loop in chunks for speed.
        """
        count = 0
        m = mask
        # Check d = 1..127
        for d in range(1, 128):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        for d in range(128, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Generate initial candidates
    candidates = []

    # Random starts
    for _ in range(20):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        candidates.append(lst)

    # Structured starts - various APs and dense blocks
    candidates.append(list(range(1, 1 + 31 * 16, 16)))  # AP with d=16
    candidates.append(list(range(1, 33)))  # Dense at start
    candidates.append(list(range(481, 513)))  # Dense at end
    candidates.append(list(range(1, 1 + 31 * 8, 8)))  # AP with d=8
    candidates.append(list(range(1, 1 + 31 * 4, 4)))  # AP with d=4
    # Mixed: two interleaved APs
    c1 = list(range(1, 1 + 16 * 32, 32))
    c2 = list(range(2, 2 + 16 * 32, 32))
    candidates.append(c1 + c2)
    # Block of 16 + block of 16
    candidates.append(list(range(1, 17)) + list(range(257, 273)))

    # Greedy construction: add elements that maximize new AP differences
    def greedy_construct():
        """Build a set greedily by adding elements that create new AP differences."""
        best_greedy = None
        best_greedy_score = -1
        
        for trial in range(6):
            cur_set = set()
            cur_mask = 0
            first = random.randint(1, MAX_VAL)
            cur_set.add(first)
            cur_mask |= (1 << first)
            
            for _ in range(N - 1):
                best_elem = None
                best_new_score = -1
                # Sample candidate elements
                for _ in range(60):
                    elem = random.randint(1, MAX_VAL)
                    if elem in cur_set:
                        continue
                    new_mask = cur_mask | (1 << elem)
                    s = score_mask(new_mask)
                    if s > best_new_score:
                        best_new_score = s
                        best_elem = elem
                if best_elem is not None:
                    cur_set.add(best_elem)
                    cur_mask |= (1 << best_elem)
            
            s_final = score_mask(cur_mask)
            if s_final > best_greedy_score:
                best_greedy_score = s_final
                best_greedy = list(cur_set)
        
        return best_greedy

    if time.time() < deadline - 20:
        g = greedy_construct()
        if g:
            candidates.append(g)

    # Evaluate initial candidates
    best_score = -1
    best_list = None
    best_mask = 0

    for lst in candidates:
        if lst is None:
            continue
        m = list_to_mask(lst)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    # Simulated Annealing with restart on stagnation
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask
    current_set = set(current_list)

    T = 8.0
    cooling_rate = 0.9999
    min_T = 0.05
    stagnation_limit = 1500
    stagnation_count = 0

    # Precompute bit shifts for common operations
    while time.time() < deadline:
        # Choose move type: single swap (75%) or remove-2-add-2 (25%)
        use_double_move = random.random() < 0.25
        
        if use_double_move:
            # Remove 2 elements, add 2 new ones
            idx1, idx2 = random.sample(range(N), 2)
            old1 = current_list[idx1]
            old2 = current_list[idx2]
            
            # Pick two new values not in current set
            new1 = random.randint(1, MAX_VAL)
            while new1 in current_set:
                new1 = random.randint(1, MAX_VAL)
            new2 = random.randint(1, MAX_VAL)
            while new2 in current_set or new2 == new1:
                new2 = random.randint(1, MAX_VAL)
            
            new_mask = current_mask ^ (1 << old1) ^ (1 << old2) ^ (1 << new1) ^ (1 << new2)
            new_score = score_mask(new_mask)
            delta = new_score - current_score
            
            if delta > 0:
                accept = True
                stagnation_count = 0
            else:
                if T < 1e-10:
                    accept = False
                else:
                    accept = random.random() < math.exp(delta / T)
            
            if accept:
                current_list[idx1] = new1
                current_list[idx2] = new2
                current_set.discard(old1)
                current_set.discard(old2)
                current_set.add(new1)
                current_set.add(new2)
                current_score = new_score
                current_mask = new_mask
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list[:]
                    best_mask = current_mask
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1
        else:
            # Single swap
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]

            # Pick a new value
            new_val = random.randint(1, MAX_VAL)
            while new_val == old_val or new_val in current_set:
                new_val = random.randint(1, MAX_VAL)

            new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
            new_score = score_mask(new_mask)
            delta = new_score - current_score

            if delta > 0:
                accept = True
                stagnation_count = 0
            else:
                if T < 1e-10:
                    accept = False
                else:
                    accept = random.random() < math.exp(delta / T)

            if accept:
                current_list[idx] = new_val
                current_set.discard(old_val)
                current_set.add(new_val)
                current_score = new_score
                current_mask = new_mask

                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list[:]
                    best_mask = current_mask
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1

        T *= cooling_rate
        if T < min_T:
            T = min_T

        # Restart if stuck
        if stagnation_count > stagnation_limit:
            # Try a new random start
            if time.time() < deadline - 5:
                new_lst = random.sample(range(1, MAX_VAL + 1), N)
                new_m = list_to_mask(new_lst)
                new_s = score_mask(new_m)
                if new_s >= best_score - 3:
                    current_list = new_lst[:]
                    current_score = new_s
                    current_mask = new_m
                    current_set = set(current_list)
                    T = 5.0
                    stagnation_count = 0

    return best_list
# EVOLVE-BLOCK-END
