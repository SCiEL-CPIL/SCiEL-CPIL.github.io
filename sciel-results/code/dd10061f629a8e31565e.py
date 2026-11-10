import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    def can_add_mask(x, m):
        """Check if x can be added to set represented by bitmask m without creating a 3-AP."""
        if m & (1 << x):
            return False
        # For each a in S, check if 2x-a is in S (x is middle) or 2a-x is in S (x is end)
        mm = m
        while mm:
            lsb = mm & -mm
            a = lsb.bit_length() - 1
            mm ^= lsb
            # x is middle: a, x, c where c = 2x - a
            c = 2 * x - a
            if 1 <= c <= N and c != a:
                if m & (1 << c):
                    return False
            # x is end: a, b, x or x, b, a where b is middle
            # If b in S, check if 2b - x is in S
            c2 = 2 * a - x
            if 1 <= c2 <= N and c2 != a and c2 != x:
                if m & (1 << c2):
                    return False
        return True

    def score_of_mask(m):
        """Compute sum of reciprocals for set represented by bitmask m."""
        s = 0.0
        mm = m
        while mm:
            lsb = mm & -mm
            x = lsb.bit_length() - 1
            s += recip[x]
            mm ^= lsb
        return s

    def get_set_bits(m):
        """Return list of set bits in bitmask m."""
        bits = []
        mm = m
        while mm:
            lsb = mm & -mm
            x = lsb.bit_length() - 1
            bits.append(x)
            mm ^= lsb
        return bits

    # Greedy initialization: add numbers in order of decreasing reciprocal weight (smallest first)
    current_mask = 0
    for x in range(1, N + 1):
        if time.time() - start_time > time_limit - 5:
            break
        if can_add_mask(x, current_mask):
            current_mask |= (1 << x)

    current_score = score_of_mask(current_mask)
    best_mask = current_mask
    best_score = current_score

    all_numbers = list(range(1, N + 1))
    set_bits = get_set_bits(current_mask)

    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    step_count = 0
    no_improve_steps = 0
    max_no_improve = 5000

    while time.time() - start_time < time_limit and T > min_T:
        step_count += 1

        # Choose move type with bias toward swaps (more likely to improve)
        r = random.random()
        if r < 0.4:
            move_type = 'swap'
        elif r < 0.7:
            move_type = 'add'
        else:
            move_type = 'remove'

        new_mask = current_mask

        if move_type == 'add':
            # Find candidates not in mask, biased toward smaller numbers (higher reciprocal)
            candidates = [x for x in all_numbers if not (current_mask & (1 << x))]
            if not candidates:
                continue
            # Bias toward smaller numbers
            weights = [recip[x] for x in candidates]
            total_w = sum(weights)
            if total_w > 0:
                r2 = random.random() * total_w
                cum = 0
                x = candidates[-1]
                for i, w in enumerate(weights):
                    cum += w
                    if r2 <= cum:
                        x = candidates[i]
                        break
            else:
                x = random.choice(candidates)
            if can_add_mask(x, current_mask):
                new_mask |= (1 << x)
            else:
                continue

        elif move_type == 'remove':
            if not set_bits:
                continue
            x = random.choice(set_bits)
            new_mask &= ~(1 << x)

        elif move_type == 'swap':
            if not set_bits:
                continue
            x = random.choice(set_bits)
            candidates = [y for y in all_numbers if not (current_mask & (1 << y))]
            if not candidates:
                continue
            # Bias toward smaller numbers (higher reciprocal)
            weights = [recip[y] for y in candidates]
            total_w = sum(weights)
            if total_w > 0:
                r2 = random.random() * total_w
                cum = 0
                y = candidates[-1]
                for i, w in enumerate(weights):
                    cum += w
                    if r2 <= cum:
                        y = candidates[i]
                        break
            else:
                y = random.choice(candidates)

            temp_mask = current_mask & ~(1 << x)
            if can_add_mask(y, temp_mask):
                new_mask = temp_mask | (1 << y)
            else:
                continue

        new_score = score_of_mask(new_mask)
        delta = new_score - current_score

        # Accept if better, or with probability exp(-delta/T) if worse
        if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_mask = new_mask
            current_score = new_score
            set_bits = get_set_bits(current_mask)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T *= cooling

        # Periodic perturbation if stuck
        if no_improve_steps >= max_no_improve and time.time() - start_time < time_limit - 10:
            # Perturb: randomly remove 2-4 elements and re-add greedily
            perturb_mask = current_mask
            if set_bits:
                num_remove = min(random.randint(2, 4), len(set_bits))
                to_remove = random.sample(set_bits, num_remove)
                for x in to_remove:
                    perturb_mask &= ~(1 << x)
            # Re-add greedily
            for x in range(1, N + 1):
                if time.time() - start_time > time_limit - 5:
                    break
                if not (perturb_mask & (1 << x)) and can_add_mask(x, perturb_mask):
                    perturb_mask |= (1 << x)
            perturb_score = score_of_mask(perturb_mask)
            if perturb_score > current_score * 0.9:  # Accept if not too bad
                current_mask = perturb_mask
                current_score = perturb_score
                set_bits = get_set_bits(current_mask)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            no_improve_steps = 0
            T = max(T, 0.1)  # Reset temperature

    # Extract best set
    result = get_set_bits(best_mask)
    return result
