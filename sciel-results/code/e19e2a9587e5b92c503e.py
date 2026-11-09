import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    N = 256

    # Precompute 3-APs as bitmasks
    three_ap_masks = []
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            m = (1 << a) | (1 << b) | (1 << c)
            three_ap_masks.append(m)

    # Precompute 4-APs as bitmasks
    four_ap_masks = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            m = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            four_ap_masks.append(m)

    # For fast 3-AP count: precompute for each element, the list of 3-AP indices it participates in
    elem_3ap_indices = [[] for _ in range(N + 1)]
    for idx, m in enumerate(three_ap_masks):
        for x in range(1, N + 1):
            if (m & (1 << x)) != 0:
                elem_3ap_indices[x].append(idx)

    # For fast 4-AP check: precompute for each element, the list of 4-AP indices it participates in
    elem_4ap_indices = [[] for _ in range(N + 1)]
    for idx, m in enumerate(four_ap_masks):
        for x in range(1, N + 1):
            if (m & (1 << x)) != 0:
                elem_4ap_indices[x].append(idx)

    def count_3aps(mask):
        count = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                count += 1
        return count

    def get_3ap_gain(mask, x):
        """Number of new 3-APs formed by adding x to mask."""
        bit_x = 1 << x
        gain = 0
        for idx in elem_3ap_indices[x]:
            m = three_ap_masks[idx]
            if (mask & (m ^ bit_x)) == (m ^ bit_x):
                gain += 1
        return gain

    def get_3ap_loss(mask, x):
        """Number of 3-APs lost by removing x from mask."""
        loss = 0
        for idx in elem_3ap_indices[x]:
            m = three_ap_masks[idx]
            if (mask & m) == m:
                loss += 1
        return loss

    def can_add(mask, x):
        """Check if adding x creates any 4-AP."""
        bit_x = 1 << x
        new_mask = mask | bit_x
        for idx in elem_4ap_indices[x]:
            m = four_ap_masks[idx]
            if (new_mask & m) == m:
                return False
        return True

    def can_swap(mask, remove_x, add_y):
        """Check if swapping remove_x for add_y maintains 4-AP-free."""
        bit_x = 1 << remove_x
        bit_y = 1 << add_y
        new_mask = (mask & ~bit_x) | bit_y
        for idx in elem_4ap_indices[add_y]:
            m = four_ap_masks[idx]
            if (new_mask & m) == m:
                return False
        return True

    def weighted_greedy(ordering):
        """Greedy that picks the element with highest marginal gain among those that don't create 4-APs."""
        mask = 0
        available = list(ordering)
        
        while available:
            best_x = None
            best_gain = -1
            for x in available:
                if can_add(mask, x):
                    gain = get_3ap_gain(mask, x)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            if best_x is None:
                break
            mask |= (1 << best_x)
            available.remove(best_x)
        
        return mask

    def local_search(mask):
        """Hill-climbing with single additions and swaps."""
        current_score = count_3aps(mask)
        improved = True

        while improved and time.time() < deadline - 3.0:
            improved = False
            best_score = current_score
            best_new_mask = mask

            # Try additions
            for y in range(1, N + 1):
                if (mask & (1 << y)) == 0:
                    if can_add(mask, y):
                        gain = get_3ap_gain(mask, y)
                        new_score = current_score + gain
                        if new_score > best_score:
                            best_score = new_score
                            best_new_mask = mask | (1 << y)

            # Try swaps
            elems_in_mask = [x for x in range(1, N + 1) if (mask & (1 << x)) != 0]
            for x in elems_in_mask:
                loss_x = get_3ap_loss(mask, x)
                for y in range(1, N + 1):
                    if (mask & (1 << y)) != 0 or y == x:
                        continue
                    if not can_swap(mask, x, y):
                        continue
                    gain_y = get_3ap_gain(mask & ~(1 << x), y)
                    new_score = current_score - loss_x + gain_y
                    if new_score > best_score:
                        best_score = new_score
                        best_new_mask = (mask & ~(1 << x)) | (1 << y)

            if best_new_mask != mask:
                mask = best_new_mask
                current_score = best_score
                improved = True

        return mask, current_score

    def simulated_annealing(mask, current_score, iterations=5000):
        """Simulated annealing to escape local optima."""
        T = 5.0
        T_min = 0.01
        cooling = 0.995

        best_mask = mask
        best_score = current_score

        for it in range(iterations):
            if time.time() > deadline - 2.0:
                break

            # Choose a random move
            move_type = random.random()
            if move_type < 0.3:
                # Try adding an element
                candidates = [x for x in range(1, N + 1) if (mask & (1 << x)) == 0]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if not can_add(mask, x):
                    continue
                new_mask = mask | (1 << x)
                delta = get_3ap_gain(mask, x)
            elif move_type < 0.7:
                # Try removing an element
                elems_in_mask = [x for x in range(1, N + 1) if (mask & (1 << x)) != 0]
                if not elems_in_mask:
                    continue
                x = random.choice(elems_in_mask)
                new_mask = mask & ~(1 << x)
                delta = -get_3ap_loss(mask, x)
            else:
                # Try swapping
                elems_in_mask = [x for x in range(1, N + 1) if (mask & (1 << x)) != 0]
                candidates_out = [x for x in range(1, N + 1) if (mask & (1 << x)) == 0]
                if not elems_in_mask or not candidates_out:
                    continue
                x = random.choice(elems_in_mask)
                y = random.choice(candidates_out)
                if not can_swap(mask, x, y):
                    continue
                new_mask = (mask & ~(1 << x)) | (1 << y)
                delta = -get_3ap_loss(mask, x) + get_3ap_gain(mask & ~(1 << x), y)

            new_score = current_score + delta

            # Accept with probability exp(-delta/T) if delta < 0
            if delta >= 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)

            if accept:
                mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask

            T = max(T_min, T * cooling)

        return best_mask, best_score

    best_mask = 0
    best_score = 0

    # Run multiple restarts with weighted greedy
    num_restarts = 30

    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        # Create a random ordering
        elems = list(range(1, N + 1))
        random.shuffle(elems)

        # Weighted greedy seed
        seed_mask = weighted_greedy(elems)
        seed_score = count_3aps(seed_mask)

        # Local search to polish
        refined_mask, refined_score = local_search(seed_mask)

        # Simulated annealing to explore further
        sa_mask, sa_score = simulated_annealing(refined_mask, refined_score, iterations=2000)

        # Final local search on SA result
        final_mask, final_score = local_search(sa_mask)

        if final_score > best_score:
            best_score = final_score
            best_mask = final_mask

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask & (1 << x)) != 0]

    return result

# EVOLVE-BLOCK-END
