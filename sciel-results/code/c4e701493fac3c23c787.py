import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute masks for all k-subsets and reverse mapping
    masks = [0] * n_k
    coverers = [[] for _ in range(n_t)]

    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = m

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')

    # Helper: Simulated Annealing with penalty
    def simulated_annealing(initial_indices, seed, max_time=8.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S

        # counts[t_idx] = number of blocks in S covering t_idx
        counts = [0] * n_t
        for b in S:
            m = masks[b]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb

        # Count uncovered
        uncovered_count = sum(1 for c in counts if c == 0)

        sa_start = time.time()
        current_len = len(S)
        best_local = current_len
        best_local_set = set(S)

        # Temperature schedule: start high, decay
        T = 50.0
        cooling = 0.995

        while (time.time() - sa_start) < max_time:
            # Try a random move
            move_type = rng.random()

            if move_type < 0.3:
                # Move 1: Remove a block
                if not S:
                    continue
                b_out = rng.choice(list(S))
                m_out = masks[b_out]

                # Check if removal keeps coverage valid
                valid = True
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        valid = False
                        break
                    temp ^= lsb

                if valid:
                    # Apply removal
                    S.discard(b_out)
                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb

                    new_len = len(S)
                    if new_len < best_local:
                        best_local = new_len
                        best_local_set = set(S)

            elif move_type < 0.6:
                # Move 2: Add a block that covers uncovered items
                if len(S) > 160:
                    continue

                # Find uncovered items
                uncovered = [i for i in range(n_t) if counts[i] == 0]
                if not uncovered:
                    continue

                # Pick a random uncovered item and a random block covering it
                target = rng.choice(uncovered)
                candidates = coverers[target]
                if not candidates:
                    continue

                b_in = rng.choice(candidates)
                if b_in in S:
                    continue

                # Add it
                S.add(b_in)
                m_in = masks[b_in]
                temp = m_in
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    old_count = counts[idx]
                    counts[idx] += 1
                    temp ^= lsb

                # Count new uncovered
                new_uncovered = sum(1 for c in counts if c == 0)
                delta = 1 + 100 * (new_uncovered - uncovered_count)

                if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                    uncovered_count = new_uncovered
                    if uncovered_count == 0:
                        new_len = len(S)
                        if new_len < best_local:
                            best_local = new_len
                            best_local_set = set(S)
                else:
                    # Revert
                    S.discard(b_in)
                    temp = m_in
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb

            elif move_type < 0.85:
                # Move 3: Swap - remove b_out, add b_in
                if not S:
                    continue
                b_out = rng.choice(list(S))
                m_out = masks[b_out]

                # Find t-subsets exclusively covered by b_out
                U = []
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        U.append(idx)
                    temp ^= lsb

                if not U:
                    # b_out is redundant, just remove it
                    S.discard(b_out)
                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                    new_len = len(S)
                    if new_len < best_local:
                        best_local = new_len
                        best_local_set = set(S)
                    continue

                # Find candidate b_in that covers all t in U
                candidates = set(coverers[U[0]])
                for t_idx in U[1:]:
                    candidates.intersection_update(coverers[t_idx])
                candidates -= S

                if not candidates:
                    continue

                b_in = rng.choice(list(candidates))
                m_in = masks[b_in]

                # Perform swap
                S.discard(b_out)
                S.add(b_in)

                # Update counts
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    temp ^= lsb

                temp = m_in
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    temp ^= lsb

                new_len = len(S)
                if new_len < best_local:
                    best_local = new_len
                    best_local_set = set(S)

            else:
                # Move 4: Random swap
                if not S:
                    continue
                b_out = rng.choice(list(S))
                m_out = masks[b_out]

                b_in = rng.randint(0, n_k - 1)
                if b_in in S:
                    continue
                m_in = masks[b_in]

                # Perform swap
                S.discard(b_out)
                S.add(b_in)

                # Update counts
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    temp ^= lsb

                temp = m_in
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    temp ^= lsb

                # Count new uncovered
                new_uncovered = sum(1 for c in counts if c == 0)
                delta = 100 * (new_uncovered - uncovered_count)

                if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                    uncovered_count = new_uncovered
                    if uncovered_count == 0:
                        new_len = len(S)
                        if new_len < best_local:
                            best_local = new_len
                            best_local_set = set(S)
                else:
                    # Revert
                    S.discard(b_in)
                    S.add(b_out)

                    temp = m_in
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb

                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        temp ^= lsb

            # Cool down
            T *= cooling

        return best_local_set

    # Run multiple randomized greedy attempts to get initial solutions
    attempts = 15
    for attempt in range(attempts):
        if time.time() > deadline - 25:
            break

        rng = random.Random(attempt * 12345 + 67890)

        current_uncovered = (1 << n_t) - 1
        selected = set()

        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0

            for i in range(n_k):
                if i in selected:
                    continue
                gain_mask = current_uncovered & masks[i]
                if not gain_mask:
                    continue
                gain = gain_mask.bit_count()

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    if rng.random() < 0.2:
                        best_gain = gain
                        best_idx = i
                        best_mask = gain_mask

            if best_idx == -1:
                break

            selected.add(best_idx)
            current_uncovered &= ~best_mask

            if time.time() > deadline - 30:
                break

        if current_uncovered == 0:
            indices = list(selected)

            # Apply Simulated Annealing
            sa_indices = simulated_annealing(indices, attempt, max_time=7.0)
            sa_len = len(sa_indices)

            if sa_len < best_len:
                best_len = sa_len
                best_solution = get_blocks(sa_indices)

            if time.time() > deadline - 25:
                break

    if best_solution is None:
        # Fallback: simple greedy
        current_uncovered = (1 << n_t) - 1
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            for i in range(n_k):
                if i in selected: continue
                gain_mask = current_uncovered & masks[i]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
            if best_idx == -1: break
            selected.append(best_idx)
            current_uncovered &= ~best_mask
        best_solution = get_blocks(selected)

    return best_solution
