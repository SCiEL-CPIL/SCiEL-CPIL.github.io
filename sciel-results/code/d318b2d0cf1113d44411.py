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

    best_mask = 0
    best_score = -1

    # Strategy: Multiple restarts with diverse initializations + simulated annealing
    # Plus a greedy construction phase to seed good starting points

    # Helper: greedy construction trying to maximize distinct differences
    def greedy_construct(target_diffs=None):
        """Build a set greedily, preferring elements that create new 3-AP differences."""
        if target_diffs is None:
            target_diffs = list(range(1, 200))
        
        S = set()
        # Start with some base elements
        base_candidates = list(range(1, 513))
        random.shuffle(base_candidates)
        
        # Add first element
        S.add(1)
        
        for _ in range(31):
            best_elem = None
            best_gain = -1
            # Sample a subset of candidates for speed
            candidates = random.sample([x for x in base_candidates if x not in S], min(100, len(base_candidates) - len(S)))
            
            for elem in candidates:
                new_S = S | {elem}
                # Count how many new differences this creates
                new_diffs = 0
                for d in range(1, 256):
                    if elem in new_S:
                        # Check if a, a+d, a+2d all in new_S for some a
                        # Quick check: does adding elem complete any 3-AP?
                        for a in range(1, 513 - 2*d + 1):
                            if a in new_S and (a+d) in new_S and (a+2*d) in new_S:
                                # Check if this difference was already achievable without elem
                                old_S = S
                                old_ok = False
                                for a2 in range(1, 513 - 2*d + 1):
                                    if a2 in old_S and (a2+d) in old_S and (a2+2*d) in old_S:
                                        old_ok = True
                                        break
                                if not old_ok:
                                    new_diffs += 1
                                break
                if new_diffs > best_gain:
                    best_gain = new_diffs
                    best_elem = elem
            
            if best_elem is not None:
                S.add(best_elem)
        
        mask = 0
        for e in S:
            mask |= (1 << e)
        return mask

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

    # Init 3: arithmetic progression with step 16
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
    vals = sorted(random.sample(range(1, 513), 32))
    mask5 = 0
    for v in vals:
        mask5 |= (1 << v)
    inits.append(mask5)

    # Init 6: structured - every other number in a range
    mask6 = 0
    for i in range(1, 65, 2):
        mask6 |= (1 << i)
    inits.append(mask6)

    # Init 7: Greedy construction
    if time.time() < deadline - 30:
        mask7 = greedy_construct()
        inits.append(mask7)

    # Init 8: Another greedy with different seed
    if time.time() < deadline - 20:
        mask8 = greedy_construct()
        inits.append(mask8)

    # Run simulated annealing for each init
    num_restarts = len(inits)
    time_per_restart = (deadline - start_time) / max(num_restarts, 1)

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

                if delta >= 0:
                    accept = True
                else:
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

    # Convert best_mask to list
    result = get_elements(best_mask)

    if len(result) != 32:
        return list(range(1, 33))

    return result
