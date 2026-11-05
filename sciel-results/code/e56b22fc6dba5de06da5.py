import time
import random
import bisect


def _check_extension(seq, sums, x):
    """
    Check if appending x to the end of seq is valid.
    Returns the new sums to add if valid, or None if invalid.
    """
    if not seq:
        if x in sums:
            return None
        return [x]

    tails = []
    total = x
    tails.append(total)
    for a in reversed(seq):
        total += a
        tails.append(total)

    for s in tails:
        if s in sums:
            return None

    return tails


def _check_insertion(seq, sums, x, pos):
    """
    Check if inserting x at position pos (0..len(seq)) keeps seq strictly
    increasing and all consecutive-block sums distinct.
    Returns the set of new sums to add if valid, else None.
    """
    L = len(seq)
    if pos < 0 or pos > L:
        return None
    if pos > 0 and seq[pos - 1] >= x:
        return None
    if pos < L and x >= seq[pos]:
        return None

    # New sums that involve x. A block containing x spans a left run of length
    # l (0..pos) and a right run of length r (0..L-pos), with l+r >= 1.
    # sum = (sum of last l elements of seq[:pos]) + x + (sum of first r elements of seq[pos:])
    new_sums = set()
    # left partial sums: Ls[l] = sum of last l elements before pos
    left = [0] * (pos + 1)
    s = 0
    for i in range(pos - 1, -1, -1):
        s += seq[i]
        left[pos - i] = s
    # right partial sums: Rs[r] = sum of first r elements after pos
    right = [0] * (L - pos + 1)
    s = 0
    for i in range(pos, L):
        s += seq[i]
        right[i - pos + 1] = s

    for l in range(pos + 1):
        ls = left[l]
        for r in range(L - pos + 1):
            if l == 0 and r == 0:
                continue
            val = ls + x + right[r]
            if val in sums:
                return None
            new_sums.add(val)

    return new_sums


def _compute_sums(seq):
    L = len(seq)
    sums = set()
    for i in range(L):
        s = 0
        for j in range(i, L):
            s += seq[j]
            sums.add(s)
    return sums


def _greedy_appending(order, max_val=256):
    """
    Greedy construction following a given order of candidate values.
    Only appends (keeps strictly increasing), using the fast incremental check.
    """
    seq = []
    sums = set()
    for x in order:
        if x < 1 or x > max_val:
            continue
        if seq and x <= seq[-1]:
            continue
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            seq.append(x)
            sums.update(new_tails)
    return seq


def _greedy_randomized(k=1, seed=None, max_val=256):
    """
    Greedy construction. k=1 is standard (smallest valid). k>1 picks randomly
    from the last k valid candidates seen before committing.
    """
    if seed is not None:
        random.seed(seed)

    seq = []
    sums = set()
    buffer = []

    for x in range(1, max_val + 1):
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            buffer.append((x, new_tails))
            if len(buffer) >= k:
                idx = random.randint(0, len(buffer) - 1)
                x_val, tails = buffer[idx]
                seq.append(x_val)
                sums.update(tails)
                buffer = []

    if buffer:
        idx = random.randint(0, len(buffer) - 1)
        x_val, tails = buffer[idx]
        seq.append(x_val)
        sums.update(tails)

    return seq


def _random_skip_greedy(seed=None, max_val=256, skip_prob=0.1):
    """Randomized greedy that sometimes skips a valid candidate."""
    if seed is not None:
        random.seed(seed)

    seq = []
    sums = set()

    for x in range(1, max_val + 1):
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            if random.random() > skip_prob:
                seq.append(x)
                sums.update(new_tails)

    return seq


def _local_search_insert(seq, max_val=256, deadline=None, max_rounds=40):
    """
    Local search: repeatedly try to insert any unused value at any valid
    position to grow the sequence. Uses the fast incremental check.
    """
    current = list(seq)
    sums = _compute_sums(current)
    used = set(current)
    available = [v for v in range(1, max_val + 1) if v not in used]
    best = current[:]
    best_len = len(current)

    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        if len(current) >= max_val:
            break

        improved = False
        # shuffle available for diversity
        random.shuffle(available)
        for x in available:
            if deadline is not None and time.time() > deadline:
                break
            L = len(current)
            # try positions in a mixed order (center-out for balance)
            positions = list(range(L + 1))
            random.shuffle(positions)
            for pos in positions:
                new_sums = _check_insertion(current, sums, x, pos)
                if new_sums is not None:
                    # commit
                    current = current[:pos] + [x] + current[pos:]
                    sums.update(new_sums)
                    used.add(x)
                    available.remove(x)
                    if len(current) > best_len:
                        best_len = len(current)
                        best = current[:]
                    improved = True
                    break  # restart position scan for this x done; move on
        if not improved:
            break

    return best


def _beam_search(seed=None, beam_width=10, max_val=256, deadline=None):
    """Beam search over partial sequences, keeping top beams by length/room."""
    if seed is not None:
        random.seed(seed)

    beams = [([], set())]

    for _ in range(max_val):
        if deadline is not None and time.time() > deadline:
            break

        new_beams = []
        for seq, sums in beams:
            last = seq[-1] if seq else 0
            for x in range(last + 1, max_val + 1):
                new_tails = _check_extension(seq, sums, x)
                if new_tails is not None:
                    new_beams.append((seq + [x], sums.union(new_tails)))

        if not new_beams:
            break

        new_beams.sort(key=lambda b: (-len(b[0]), b[0][-1] if b[0] else 0))
        beams = new_beams[:beam_width]

    best = max(beams, key=lambda b: len(b[0]))
    return best[0]


def _dfs_extend(seq, sums, start_val, max_val, deadline, memo, best_seq, best_len):
    """DFS with memoization to extend a prefix."""
    if time.time() > deadline:
        return best_seq[:]

    last = seq[-1] if seq else 0
    state_key = (frozenset(sums), last)

    if state_key in memo:
        if memo[state_key] > best_len[0]:
            best_len[0] = memo[state_key]
        return best_seq[:]

    best_local = best_seq[:]
    best_local_len = best_len[0]

    for x in range(start_val, max_val + 1):
        if x <= last:
            continue
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            new_seq = seq + [x]
            new_sums = sums | set(new_tails)

            if len(new_seq) > best_local_len:
                best_local = new_seq[:]
                best_local_len = len(new_seq)
                best_seq[:] = new_seq[:]
                best_len[0] = best_local_len

            _dfs_extend(new_seq, new_sums, x + 1, max_val, deadline, memo, best_seq, best_len)

            if time.time() > deadline:
                break

    memo[state_key] = max(best_local_len, len(seq))
    return best_seq[:]


def _try_extend_from_prefix(prefix, prefix_sums, max_val=256, deadline=None):
    """Extend a prefix via DFS with memoization."""
    if deadline is not None and time.time() > deadline:
        return prefix[:]

    best_seq = prefix[:]
    best_len = [len(prefix)]
    memo = {}

    start_val = (prefix[-1] + 1) if prefix else 1
    _dfs_extend(prefix, set(prefix_sums), start_val, max_val, deadline, memo, best_seq, best_len)

    return best_seq


def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    best_seq = []
    best_len = 0

    def consider(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Phase 1: baselines
    consider(_greedy_appending(list(range(1, 257))))
    consider(_greedy_appending(list(range(256, 0, -1))))
    consider(_greedy_randomized(k=1, seed=0))

    # Phase 2: randomized-order greedy baselines (diverse)
    rng = random.Random(7)
    for trial in range(40):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        rng.shuffle(order)
        consider(_greedy_appending(order))

    # Phase 3: randomized greedy with varying k and seeds
    for k in [2, 3, 5, 10, 20]:
        for seed in range(15):
            if time.time() > deadline:
                break
            consider(_greedy_randomized(k=k, seed=seed))
        if time.time() > deadline:
            break

    # Phase 4: random skip greedy
    for skip_prob in [0.05, 0.1, 0.2, 0.3]:
        for seed in range(10):
            if time.time() > deadline:
                break
            consider(_random_skip_greedy(seed=seed, skip_prob=skip_prob))
        if time.time() > deadline:
            break

    # Phase 5: position-aware insertion local search on best (and a few others)
    random.seed(123)
    if time.time() < deadline - 15:
        improved = _local_search_insert(best_seq, deadline=deadline - 10)
        consider(improved)

    # Phase 6: beam search
    if time.time() < deadline - 10:
        for bw in [8, 15]:
            if time.time() > deadline - 5:
                break
            consider(_beam_search(seed=0, beam_width=bw, deadline=deadline - 5))

    # Phase 7: another insertion local search pass on the (possibly) improved best
    if time.time() < deadline - 5:
        random.seed(456)
        improved = _local_search_insert(best_seq, deadline=deadline - 2, max_rounds=25)
        consider(improved)

    # Phase 8: targeted DFS polish from short prefixes of the best
    if time.time() < deadline - 3 and best_len > 0:
        for prefix_len in range(min(4, best_len), 0, -1):
            if time.time() > deadline - 2:
                break
            prefix = best_seq[:prefix_len]
            prefix_sums = _compute_sums(prefix)
            consider(_try_extend_from_prefix(prefix, prefix_sums, deadline=deadline - 2))

    return best_seq
