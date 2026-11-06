"""Snake-in-the-box Q_9: optimized beam search with parameter-tuned scoring and multi-phase strategy."""
import time
import random

N = 9
SIZE = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m


def _greedy_extend(start, bit_order=None):
    """Greedy extension from a single start vertex using forbidden_mask."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0

    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break

        best_v = None
        best_score = SIZE + 1
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
            new_used = used_mask | (1 << v)
            next_opts = NEIGHBOR_MASKS[v] & ~(new_used | new_forbidden)
            score = next_opts.bit_count()
            if score < best_score:
                best_score = score
                best_v = v

        if best_v is None:
            break
        path.append(best_v)
        used_mask |= (1 << best_v)
        forbidden_mask |= NEIGHBOR_MASKS[cur]

    return path


def _lookahead_estimate(cur, used_mask, forbidden_mask, depth=3):
    """
    Estimate the potential length of the path by greedily extending
    for 'depth' steps and counting the remaining options.
    Returns a score: (depth_reached, remaining_options)
    """
    cur_forb = forbidden_mask
    cur_used = used_mask
    steps = 0
    for _ in range(depth):
        candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)
        if candidates == 0:
            break
        
        # Pick the candidate with the most future options (greedy)
        best_v = None
        best_score = -1
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            new_forb = cur_forb | NEIGHBOR_MASKS[cur]
            new_used = cur_used | (1 << v)
            next_opts = (NEIGHBOR_MASKS[v] & ~(new_used | new_forb)).bit_count()
            if next_opts > best_score:
                best_score = next_opts
                best_v = v
        
        if best_v is None:
            break
            
        cur = best_v
        cur_used |= (1 << cur)
        cur_forb |= NEIGHBOR_MASKS[cur] # Note: This logic in lookahead is slightly off for 'forbidden' update, 
        # actually forbidden_mask should include neighbors of the *previous* vertex.
        # In the main search, forbidden_mask is updated by OR-ing NEIGHBOR_MASKS[prev_cur].
        # Here we are simulating.
        # Let's just return the number of steps we could take.
        steps += 1
        
    # Estimate: steps taken + log of remaining options?
    # Simpler: just return steps as a bonus.
    return steps

def _beam_search(start, beam_width, max_time, rng, alpha=1.0, beta=0.5):
    """
    Beam search with tunable scoring parameters.
    alpha: weight for path length in sorting
    beta: weight for next_options (lower is better)
    """
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = 1
    t0 = time.time()

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []

        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)

            if candidates == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb

                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        def _score(entry):
            p, used, forb = entry
            cur = p[-1]
            next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
            # Lookahead adds a bonus for how far we can greedily go
            la = _lookahead_estimate(cur, used, forb, depth=2)
            return (-len(p) * alpha - next_opts * beta - la * 1.5)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _beam_search_from_path(prefix, beam_width, max_time, rng, alpha=1.0, beta=0.5):
    """Beam search continuing from an existing path prefix."""
    if not prefix:
        return _beam_search(0, beam_width, max_time, rng, alpha, beta)

    start = prefix[-1]
    used_mask = 0
    for v in prefix:
        used_mask |= 1 << v
    forbidden_mask = 0
    for v in prefix[:-1]:
        forbidden_mask |= NEIGHBOR_MASKS[v]

    path = list(prefix)
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = len(path)
    t0 = time.time()

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []

        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)

            if candidates == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb

                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        def _score(entry):
            p, used, forb = entry
            cur = p[-1]
            next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
            la = _lookahead_estimate(cur, used, forb, depth=2)
            return (-len(p) * alpha - next_opts * beta - la * 1.5)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _local_improve(path, deadline, rng, alpha=1.0, beta=0.5):
    """Try to improve path by re-extending from various cut points."""
    best_path = list(path)
    best_len = len(path)
    n = len(path)

    if n < 4:
        return best_path

    positions = []
    for i in range(max(1, n // 2), n - 1):
        positions.append(i)
    for i in range(1, max(1, n // 4)):
        positions.append(i)
    rng.shuffle(positions)

    for pos in positions[:20]:
        if time.time() > deadline - 3:
            break
        prefix = path[:pos + 1]
        remaining = deadline - time.time()
        if remaining < 2:
            break
        dur = min(3.0, remaining - 1)
        improved = _beam_search_from_path(prefix, 40, time.time() + dur, rng, alpha, beta)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    rng = random.Random(42)

    best_path = None
    best_len = 0

    # Phase 1: Greedy seed
    greedy = _greedy_extend(0)
    best_path = greedy
    best_len = len(greedy)

    # Phase 2: Beam search with different parameter settings
    # Try several combinations of beam_width and scoring parameters
    # Increased beam widths and adjusted scoring weights
    param_configs = [
        (100, 8.0, 1.0, 0.5),
        (200, 10.0, 1.0, 0.3),
        (300, 12.0, 1.5, 0.2),
        (400, 14.0, 1.0, 0.4),
        (500, 16.0, 2.0, 0.1),
    ]

    for bw, duration, alpha, beta in param_configs:
        if time.time() > deadline - 12:
            break
        remaining = deadline - time.time()
        if remaining < duration + 2:
            duration = max(2.0, remaining - 2)
        t0 = time.time()
        result = _beam_search(0, bw, t0 + duration, rng, alpha, beta)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result

    # Phase 3: Extend the best path found
    if time.time() < deadline - 15 and best_len < 200:
        remaining = deadline - time.time()
        if remaining > 10:
            dur = min(15.0, remaining - 5)
            result = _beam_search_from_path(best_path, 100, time.time() + dur, rng, 1.0, 0.3)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 4: Multi-restart with varied parameters
    if time.time() < deadline - 15:
        starts = list(range(SIZE))
        rng.shuffle(starts)
        restart_configs = [
            (100, 3.5, 1.0, 0.5),
            (150, 3.0, 1.2, 0.3),
            (200, 2.5, 0.8, 0.4),
        ]
        for start in starts[:30]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time()
            if remaining < 3:
                break
            for bw, dur, alpha, beta in restart_configs:
                if time.time() > deadline - 3:
                    break
                if remaining < dur:
                    dur = max(1.5, remaining - 1)
                result = _beam_search(start, bw, time.time() + dur, rng, alpha, beta)
                if len(result) > best_len:
                    best_len = len(result)
                    best_path = result
                if time.time() > deadline - 5:
                    break

    # Phase 5: Local improvement
    if time.time() < deadline - 8:
        improved = _local_improve(best_path, deadline, rng, 1.0, 0.3)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 6: Final large beam with tuned parameters
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 10:
            bw = 800
            dur = remaining - 5
            result = _beam_search(0, bw, time.time() + dur, rng, 1.5, 0.2)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 7: One more local improvement
    if time.time() < deadline - 5:
        improved = _local_improve(best_path, deadline, rng, 1.2, 0.2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path
