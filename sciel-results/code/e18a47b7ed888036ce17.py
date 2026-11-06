import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def calc_score(mask):
        """Count distinct d > 0 where a, a+d, a+2d all in set (bitmask)."""
        s = 0
        m = mask
        for d in range(1, 256):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    def get_elements(mask):
        """Convert bitmask to list of elements."""
        elems = []
        for i in range(1, 513):
            if mask & (1 << i):
                elems.append(i)
        return elems

    def count_bits(mask):
        """Count set bits."""
        c = 0
        m = mask
        while m:
            c += 1
            m &= m - 1
        return c

    def get_set_list(mask):
        """Get list of elements in mask."""
        return [i for i in range(1, 513) if mask & (1 << i)]

    def greedy_max_coverage(mask, max_iter=200):
        """
        Greedy heuristic: repeatedly find the element with minimal 'coverage'
        (number of distinct d it participates in as part of a 3-AP) and swap
        it with a candidate that covers the most new differences.
        """
        current = mask
        current_score = calc_score(mask)
        best_local = mask
        best_local_score = current_score

        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break

            elems = get_set_list(current)
            if len(elems) != 32:
                break

            # For each element, count how many distinct d it's involved in
            elem_d_count = {e: 0 for e in elems}
            # For each d that works, find which elements participate
            for d in range(1, 256):
                if current & (current >> d) & (current >> (2 * d)):
                    # Find all a such that a, a+d, a+2d in set
                    for a in range(1, 513):
                        if (current >> a) & 1 and (current >> (a+d)) & 1 and (current >> (a+2*d)) & 1:
                            elem_d_count[a] = elem_d_count.get(a, 0) + 1
                            elem_d_count[a+d] = elem_d_count.get(a+d, 0) + 1
                            elem_d_count[a+2*d] = elem_d_count.get(a+2*d, 0) + 1

            # Find element with minimum coverage (tie-break: smallest value)
            min_elem = min(elems, key=lambda e: (elem_d_count.get(e, 0), e))

            # Find best replacement: try all non-elements, pick one that maximizes new score
            unset = [i for i in range(1, 513) if not (current & (1 << i))]
            if not unset:
                break

            best_new_score = current_score - 1  # Must improve
            best_replacement = None

            # Sample candidates if too many
            if len(unset) > 50:
                candidates = random.sample(unset, 50)
            else:
                candidates = unset

            for y in candidates:
                new_mask = current ^ (1 << min_elem) ^ (1 << y)
                ns = calc_score(new_mask)
                if ns > best_new_score:
                    best_new_score = ns
                    best_replacement = y

            if best_replacement is None:
                break

            new_mask = current ^ (1 << min_elem) ^ (1 << best_replacement)
            if best_new_score > current_score:
                current = new_mask
                current_score = best_new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = current
            else:
                break

        return best_local, best_local_score

    best_mask = 0
    best_score = -1

    # Strategy: multiple simulated annealing restarts with different inits
    # Use bitmasks for fast scoring

    # Initial configurations
    inits = []

    # Init 1: contiguous block 1..32
    mask1 = 0
    for i in range(1, 33):
        mask1 |= (1 << i)
    inits.append(mask1)

    # Init 2: random 32 elements
    elems = random.sample(range(1, 513), 32)
    mask2 = 0
    for e in elems:
        mask2 |= (1 << e)
    inits.append(mask2)

    # Init 3: arithmetic progression with step 16: 1, 17, 33, ...
    mask3 = 0
    for i in range(32):
        val = 1 + i * 16
        if val <= 512:
            mask3 |= (1 << val)
    inits.append(mask3)

    # Init 4: two interleaved APs
    mask4 = 0
    for i in range(16):
        mask4 |= (1 << (1 + i * 32))
        mask4 |= (1 << (2 + i * 32))
    inits.append(mask4)

    # Init 5: geometric-ish spread
    mask5 = 0
    vals = sorted(random.sample(range(1, 513), 32))
    for v in vals:
        mask5 |= (1 << v)
    inits.append(mask5)

    # Init 6: structured - every other number in a range
    mask6 = 0
    for i in range(1, 65, 2):
        mask6 |= (1 << i)
    inits.append(mask6)

    # Init 7: 3-cluster structure
    mask7 = 0
    # Three clusters of ~11 elements each, spread out
    for base in [1, 180, 360]:
        for i in range(11):
            val = base + i * 5
            if val <= 512:
                mask7 |= (1 << val)
    # Fill remaining slots
    remaining = 32 - count_bits(mask7)
    used = set(get_set_list(mask7))
    for i in range(1, 513):
        if remaining <= 0:
            break
        if i not in used:
            mask7 |= (1 << i)
            used.add(i)
            remaining -= 1
    inits.append(mask7)

    # Init 8: geometric spread
    mask8 = 0
    vals = []
    v = 1
    while len(vals) < 32 and v <= 512:
        vals.append(v)
        v = int(v * 1.15) + 1
    # Fill if needed
    while len(vals) < 32:
        vals.append(random.randint(1, 512))
    # Ensure distinct
    seen = set()
    final_vals = []
    for v in vals:
        if v not in seen and 1 <= v <= 512:
            final_vals.append(v)
            seen.add(v)
    while len(final_vals) < 32:
        v = random.randint(1, 512)
        if v not in seen:
            final_vals.append(v)
            seen.add(v)
    for v in final_vals[:32]:
        mask8 |= (1 << v)
    inits.append(mask8)

    # Run simulated annealing for each init
    num_restarts = len(inits)
    time_per_restart = (deadline - start_time) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # Simulated annealing parameters
        T = 1.0
        cool_rate = 0.9995
        min_T = 0.005

        restart_deadline = start_time + (r + 1) * time_per_restart

        no_improve = 0
        max_no_improve = 2000

        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            # Generate move: swap a random element with a random non-element
            # Pick random x in 1..512, random y in 1..512
            # If x in mask and y not in mask: remove x, add y
            x = random.randint(1, 512)
            y = random.randint(1, 512)

            if x == y:
                continue

            bx = 1 << x
            by = 1 << y

            # x must be in mask, y must NOT be in mask
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                # Accept if better, or with probability exp(-delta/T) for worse
                if delta >= 0:
                    accept = True
                else:
                    # delta is negative, so -delta/T is positive
                    # Accept with probability exp(delta/T) where delta < 0
                    threshold = math.exp(delta / T)
                    accept = random.random() < threshold

                if accept:
                    current_mask = new_mask
                    current_score = new_score
                    no_improve = 0

                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    no_improve += 1

                T *= cool_rate

                if no_improve > max_no_improve:
                    break

            # Occasionally try a "kick" - swap 2 elements at once
            if random.random() < 0.05:
                # Pick 2 elements to remove, 2 to add
                # Find set bits and unset bits
                set_list = []
                unset_list = []
                for i in range(1, 513):
                    if current_mask & (1 << i):
                        set_list.append(i)
                    else:
                        unset_list.append(i)

                if len(set_list) >= 2 and len(unset_list) >= 2:
                    x1, x2 = random.sample(set_list, 2)
                    y1, y2 = random.sample(unset_list, 2)

                    new_mask = current_mask
                    new_mask ^= (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)

                    if count_bits(new_mask) == 32:
                        new_score = calc_score(new_mask)
                        delta = new_score - current_score

                        if delta >= 0 or random.random() < math.exp(delta / T):
                            current_mask = new_mask
                            current_score = new_score
                            no_improve = 0

                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask

    # Apply greedy max coverage to best found
    if time.time() < deadline - 10 and best_mask != 0:
        g_mask, g_score = greedy_max_coverage(best_mask, max_iter=100)
        if g_score > best_score:
            best_score = g_score
            best_mask = g_mask

    # Final: try a quick local search from the best found
    if time.time() < deadline - 5 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        T = 0.05
        cool_rate = 0.999
        min_T = 0.001

        while time.time() < deadline - 1:
            if T < min_T:
                break

            x = random.randint(1, 512)
            y = random.randint(1, 512)

            if x == y:
                continue

            bx = 1 << x
            by = 1 << y

            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score

                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask

                T *= cool_rate

    # Final polish: one more greedy pass
    if time.time() < deadline - 3 and best_mask != 0:
        g_mask, g_score = greedy_max_coverage(best_mask, max_iter=50)
        if g_score > best_score:
            best_score = g_score
            best_mask = g_mask

    # Convert best_mask to list
    result = get_elements(best_mask)

    # Verify length
    if len(result) != 32:
        # Fallback
        return list(range(1, 33))

    return result
