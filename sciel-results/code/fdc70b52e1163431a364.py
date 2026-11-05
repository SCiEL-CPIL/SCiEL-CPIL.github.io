import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 84.0  # Safety margin before 90s limit

    # Precompute all 4-APs: list of tuples (a, b, c, d) with 1-indexed values
    # For each element, store the indices of APs it participates in
    all_aps = []
    element_aps = [[] for _ in range(N)]  # 0-indexed element -> list of AP indices

    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(all_aps)
            all_aps.append((a, b, c, e))
            element_aps[a - 1].append(ap_idx)
            element_aps[b - 1].append(ap_idx)
            element_aps[c - 1].append(ap_idx)
            element_aps[e - 1].append(ap_idx)

    num_aps = len(all_aps)
    
    # For fast conflict checking: for each element, store the 3 other elements in each AP
    # conflicts[i] = list of (other1, other2, other3) for each AP containing element i (0-indexed)
    conflicts = [[] for _ in range(N)]
    for ap_idx, (a, b, c, e) in enumerate(all_aps):
        # a's others: b, c, e
        conflicts[a - 1].append((b - 1, c - 1, e - 1))
        # b's others: a, c, e
        conflicts[b - 1].append((a - 1, c - 1, e - 1))
        # c's others: a, b, e
        conflicts[c - 1].append((a - 1, b - 1, e - 1))
        # e's others: a, b, c
        conflicts[e - 1].append((a - 1, b - 1, c - 1))

    def count_violations(state, x_idx):
        """
        Count how many 4-APs are fully satisfied if we add element x_idx (0-indexed)
        to the current state (numpy boolean array).
        Returns the number of new 4-APs that would be completed.
        """
        count = 0
        for (o1, o2, o3) in conflicts[x_idx]:
            if state[o1] and state[o2] and state[o3]:
                count += 1
        return count

    def count_removed_violations(state, x_idx):
        """
        Count how many 4-APs are fully satisfied ONLY because of element x_idx (0-indexed).
        i.e., if we remove x_idx, how many APs become broken.
        """
        count = 0
        for (o1, o2, o3) in conflicts[x_idx]:
            if state[o1] and state[o2] and state[o3]:
                count += 1
        return count

    # Initial state: greedy construction in natural order
    state = np.zeros(N, dtype=bool)
    for x in range(N):
        # Check if adding x creates any 4-AP
        can_add = True
        for (o1, o2, o3) in conflicts[x]:
            if state[o1] and state[o2] and state[o3]:
                can_add = False
                break
        if can_add:
            state[x] = True

    best_state = state.copy()
    best_size = int(np.sum(best_state))

    # Simulated annealing
    T_start = 2.0
    T_end = 0.01
    num_iterations = 500000
    
    # We'll do multiple restarts if time permits
    max_restarts = 10
    rng = random.Random(42)

    for restart in range(max_restarts):
        # Reset state with some randomness
        if restart == 0:
            # Use the greedy result
            state = best_state.copy()
        else:
            # Perturb the best state
            state = best_state.copy()
            # Randomly flip some bits
            num_flips = rng.randint(10, 50)
            for _ in range(num_flips):
                idx = rng.randint(0, N - 1)
                state[idx] = not state[idx]
            # Fix any violations by removing elements
            # Simple approach: rebuild from scratch with greedy + random
            state = np.zeros(N, dtype=bool)
            order = list(range(N))
            rng.shuffle(order)
            for x in order:
                can_add = True
                for (o1, o2, o3) in conflicts[x]:
                    if state[o1] and state[o2] and state[o3]:
                        can_add = False
                        break
                if can_add:
                    state[x] = True

        current_size = int(np.sum(state))
        T = T_start

        for it in range(num_iterations):
            if time.time() > deadline - 2.0:
                break

            # Pick a random element to flip
            x_idx = rng.randint(0, N - 1)

            if state[x_idx]:
                # Remove element
                # How many APs are broken by removing x_idx?
                removed_aps = count_removed_violations(state, x_idx)
                # Delta in size: -1
                # Delta in violations: -removed_aps (we remove those APs)
                # But we want to minimize violations, so removing an element that's part of
                # 'removed_aps' APs reduces violations by that amount.
                # Actually, for 4-AP-free, we want 0 violations. The "cost" is the number
                # of 4-APs present.
                # Current cost = total 4-APs in state
                # If we remove x_idx, cost decreases by removed_aps (those APs are no longer complete)
                # Size decreases by 1.
                # We want to maximize size with 0 cost.
                # Let's use energy = -size + penalty * cost
                # For simplicity, let's use: if current state has 0 violations, 
                # removing an element always worsens size.
                # Better: use energy = -size + 100 * num_violations
                # But computing total violations is expensive.
                
                # Alternative: track violations incrementally.
                # For now, let's use a simpler approach:
                # If state is 4-AP-free (0 violations), only accept removals with high T.
                # If state has violations, removing an element that breaks APs is good.
                
                # Let's just try the move and check if it's valid
                state[x_idx] = False
                new_size = current_size - 1
                
                # Check if this removal helps (reduces violations)
                # We need to know if x_idx was part of any complete 4-AP
                # We already computed removed_aps above
                if removed_aps > 0:
                    # This removal breaks some APs, so it's an improvement in terms of validity
                    # Accept with high probability
                    if rng.random() < 0.9:
                        current_size = new_size
                    else:
                        state[x_idx] = True
                else:
                    # This removal doesn't break any APs, so it's just a size reduction
                    # Accept with low probability (annealing)
                    delta_E = 1.0  # energy increases by 1 (size decreases)
                    if rng.random() < np.exp(-delta_E / T):
                        current_size = new_size
                    else:
                        state[x_idx] = True
            else:
                # Add element
                new_aps = count_violations(state, x_idx)
                if new_aps == 0:
                    # Safe addition
                    state[x_idx] = True
                    current_size += 1
                else:
                    # This addition creates new_aps 4-APs
                    # Energy increase = new_aps * penalty - 1 (size gain)
                    # Let's use penalty = 5
                    delta_E = 5.0 * new_aps - 1.0
                    if rng.random() < np.exp(-delta_E / T):
                        state[x_idx] = True
                        current_size += 1

            # Cool down
            T = T_start * (T_end / T_start) ** (it / num_iterations)

            # Track best
            if current_size > best_size:
                best_state = state.copy()
                best_size = current_size

            if time.time() > deadline - 2.0:
                break

        if time.time() > deadline - 2.0:
            break

    # Final: try to add any remaining elements greedily
    for x in range(N):
        if not best_state[x]:
            can_add = True
            for (o1, o2, o3) in conflicts[x]:
                if best_state[o1] and best_state[o2] and best_state[o3]:
                    can_add = False
                    break
            if can_add:
                best_state[x] = True

    # Convert to list
    result = [i + 1 for i in range(N) if best_state[i]]

    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False

    if has_4ap(result):
        return [1]

    return sorted(result)
