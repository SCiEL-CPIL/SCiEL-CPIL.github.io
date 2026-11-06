import time
import random
import numpy as np

def _score_bitmask(mask):
    """
    Fast score using a 512-bit integer bitmask.
    mask: int with bit i set iff (i+1) is in the set (i ranges 0..511).
    Returns number of distinct d>0 such that a, a+d, a+2d all in set.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _set_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_set(mask):
    result = []
    temp = mask
    idx = 0
    while temp:
        if temp & 1:
            result.append(idx + 1)
        temp >>= 1
        idx += 1
    return result

def _get_set_bits(mask):
    bits = []
    temp = mask
    idx = 0
    while temp:
        if temp & 1:
            bits.append(idx)
        temp >>= 1
        idx += 1
    return bits

def _make_initial_set(rng, style):
    """Create diverse initial sets."""
    if style == 0:
        # Random
        return list(rng.sample(range(1, 513), 32))
    elif style == 1:
        # Arithmetic progression
        start = rng.randint(1, 30)
        step = rng.randint(10, 15)
        s = set()
        x = start
        while len(s) < 32 and x <= 512:
            s.add(x)
            x += step
        remaining = [v for v in range(1, 513) if v not in s]
        if len(s) < 32:
            s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)
    elif style == 2:
        # Two interleaved APs
        start1 = rng.randint(1, 20)
        step1 = rng.randint(12, 18)
        start2 = rng.randint(21, 40)
        step2 = rng.randint(12, 18)
        s = set()
        x = start1
        while len(s) < 16 and x <= 512:
            s.add(x)
            x += step1
        x = start2
        while len(s) < 32 and x <= 512:
            s.add(x)
            x += step2
        remaining = [v for v in range(1, 513) if v not in s]
        if len(s) < 32:
            s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)
    elif style == 3:
        # Clustered: groups in different regions
        regions = [(1, 128), (129, 256), (257, 384), (385, 512)]
        s = set()
        for lo, hi in regions:
            count = 8
            s.update(rng.sample(range(lo, hi + 1), count))
        return list(s)
    elif style == 4:
        # Grid-like: 4 rows x 8 cols with spacing
        row_step = rng.randint(14, 20)
        col_step = rng.randint(1, 4)
        s = set()
        for r in range(4):
            for c in range(8):
                v = 1 + r * row_step + c * col_step
                if v <= 512:
                    s.add(v)
        remaining = [v for v in range(1, 513) if v not in s]
        if len(s) < 32:
            s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)
    else:
        # Mixed: half AP, half random
        s = set()
        start = rng.randint(1, 20)
        step = rng.randint(14, 18)
        x = start
        while len(s) < 16 and x <= 512:
            s.add(x)
            x += step
        remaining = [v for v in range(1, 513) if v not in s]
        s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    rng = random.Random(42)

    best_mask = 0
    best_score = -1

    # Evaluate diverse initial structures
    for style in range(6):
        if time.time() > deadline:
            break
        s = _make_initial_set(rng, style)
        m = _set_to_mask(s)
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Multi-restart: simulated annealing + local search
    num_restarts = 10
    restart_budget = 12.0  # seconds per restart

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbation
        if restart == 0:
            current_mask = best_mask
            current_score = best_score
        else:
            current_mask = best_mask
            # Perturb: remove 3-6 random elements, add new ones
            num_perturb = rng.randint(3, 6)
            for _ in range(num_perturb):
                bits_set = _get_set_bits(current_mask)
                if not bits_set:
                    break
                remove_idx = rng.choice(bits_set)
                current_mask &= ~(1 << remove_idx)
                while True:
                    add_idx = rng.randint(0, 511)
                    if not (current_mask & (1 << add_idx)):
                        current_mask |= (1 << add_idx)
                        break
            current_score = _score_bitmask(current_mask)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask

        # Simulated annealing phase
        T = 2.0
        T_min = 0.01
        cooling = 0.9998
        restart_deadline = start_time + (time.time() - start_time) + restart_budget
        if restart_deadline > deadline:
            restart_deadline = deadline

        while time.time() < restart_deadline and T > T_min:
            # Generate move: remove one element, add another
            bits_set = _get_set_bits(current_mask)
            if not bits_set:
                break
            remove_idx = rng.choice(bits_set)

            # Choose add position: biased toward nearby values
            if rng.random() < 0.4:
                # Local: near a random existing element
                ref = rng.choice(bits_set)
                delta = rng.randint(-30, 30)
                add_idx = ref + delta
                if add_idx < 0 or add_idx > 511:
                    add_idx = rng.randint(0, 511)
            else:
                add_idx = rng.randint(0, 511)

            if add_idx == remove_idx:
                continue
            if current_mask & (1 << add_idx):
                continue

            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)

            new_score = _score_bitmask(new_mask)
            delta_score = new_score - current_score

            if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                pass  # reject

            T *= cooling

        # Local search phase: greedy single-swap improvements
        local_deadline = min(restart_deadline + 3.0, deadline)
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            bits_set = _get_set_bits(current_mask)
            if not bits_set:
                break

            # Try several random swaps
            for _ in range(80):
                if time.time() >= local_deadline:
                    break
                remove_idx = rng.choice(bits_set)
                add_idx = rng.randint(0, 511)
                if add_idx == remove_idx:
                    continue
                if current_mask & (1 << add_idx):
                    continue

                new_mask = current_mask
                new_mask &= ~(1 << remove_idx)
                new_mask |= (1 << add_idx)

                new_score = _score_bitmask(new_mask)
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                    bits_set = _get_set_bits(current_mask)

    # Convert best mask back to list
    result = _mask_to_set(best_mask)

    # Ensure exactly 32 elements (safety)
    if len(result) != 32:
        # Fallback: fill with random
        remaining = [v for v in range(1, 513) if v not in set(result)]
        result.extend(rng.sample(remaining, 32 - len(result)))
        result = result[:32]

    return sorted(result)
