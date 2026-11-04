"""Greedy set-covering + simulated annealing for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _hamming_ball_centers(center, q, n, radius):
    """Return array of shape (n, m) where each column is a word within Hamming
    distance <= radius of center. Uses numpy for speed."""
    center = np.asarray(center, dtype=np.int32)
    # Generate all offsets of weight 0..radius
    offsets = [np.zeros(n, dtype=np.int32)]
    for r in range(1, radius + 1):
        for pos in itertools.combinations(range(n), r):
            for vals in itertools.product(range(1, q), repeat=r):
                off = np.zeros(n, dtype=np.int32)
                for i, v in zip(pos, vals):
                    off[i] = v
                offsets.append(off)
    offsets = np.array(offsets, dtype=np.int32).T  # shape (n, m)
    # center + offset mod q
    result = (center[:, None] + offsets) % q
    return result  # shape (n, m)


def _encode_word(w):
    """Encode a word (tuple/list of ints) to a single Python int for hashing."""
    x = 0
    for v in w:
        x = x * 3 + v
    return x


def _decode_word(x, n):
    w = []
    for _ in range(n):
        w.append(x % 3)
        x //= 3
    return w


def construct(q, n, radius):
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    total_space = q ** n
    # Precompute all words and their encoded forms
    all_words = list(itertools.product(range(q), repeat=n))
    all_encoded = [_encode_word(w) for w in all_words]
    encoded_to_idx = {e: i for i, e in enumerate(all_encoded)}

    # Precompute Hamming balls (as sets of encoded indices) for each word
    # This is memory heavy for large n, so only do it if feasible
    # For n=6, q=3: 729 words, each ball has 1 + 6*2 + 15*4 + ... = manageable
    # ball size for radius 1: 1 + n*(q-1) = 1 + 6*2 = 13
    ball_size = 1
    for r in range(1, radius + 1):
        ball_size += (n * (q - 1) ** r)  # approximate, not exact for higher r
    # More exact: sum_{r=0}^{radius} C(n,r)*(q-1)^r
    ball_size_exact = 0
    for r in range(radius + 1):
        ball_size_exact += (1 if r == 0 else 1)  # placeholder
    # Compute exact ball size
    from math import comb
    ball_size_exact = sum(comb(n, r) * (q - 1) ** r for r in range(radius + 1))

    # Precompute balls as frozensets of encoded indices
    # Only feasible if total_space * ball_size is not too large
    if total_space * ball_size_exact <= 20_000_000:
        balls = []
        for w in all_words:
            ball_centers = _hamming_ball_centers(w, q, n, radius)
            ball_set = frozenset(_encode_word(tuple(col)) for col in ball_centers.T)
            balls.append(ball_set)
    else:
        balls = None

    # Greedy set cover: repeatedly pick the word whose ball covers the most uncovered
    uncovered = set(range(total_space))
    selected = []
    selected_set = set()

    if balls is not None:
        # Greedy phase
        while uncovered and time.time() < DEADLINE - 30:
            best_idx = -1
            best_count = -1
            for i in range(total_space):
                if i in selected_set:
                    continue
                count = len(balls[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(all_words[best_idx])
            selected_set.add(best_idx)
            uncovered -= balls[best_idx]
    else:
        # Fallback: use direct sum construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            selected = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            selected = [list(w) for w in all_words]

    # Simulated annealing to remove redundant codewords
    # A codeword is redundant if removing it still leaves full coverage
    def _is_redundant(idx, selected_set, balls_list):
        """Check if codeword at index idx can be removed while maintaining coverage."""
        if balls_list is None:
            return False
        ball = balls_list[idx]
        # Check if any word in ball is only covered by this codeword
        for word_idx in ball:
            # Find how many selected codewords cover word_idx
            count = 0
            for s in selected_set:
                if s == idx:
                    continue
                if word_idx in balls_list[s]:
                    count += 1
                    if count >= 1:
                        break
            if count == 0:
                return False
        return True

    # SA phase: try to remove redundant codewords
    if balls is not None:
        sa_temperature = 1.0
        sa_cooling = 0.95
        random.seed(42)
        while time.time() < DEADLINE - 5 and sa_temperature > 0.01:
            improved = False
            for _ in range(min(200, len(selected))):
                if time.time() > DEADLINE - 5:
                    break
                # Pick a random selected codeword to try removing
                pos = random.randrange(len(selected))
                w = selected[pos]
                enc = _encode_word(tuple(w))
                idx = encoded_to_idx.get(enc, -1)
                if idx < 0:
                    continue
                # Check redundancy
                if _is_redundant(idx, selected_set, balls):
                    # Remove it
                    selected.pop(pos)
                    selected_set.discard(idx)
                    improved = True
            if not improved:
                sa_temperature *= sa_cooling
            else:
                sa_temperature *= sa_cooling

    # Final verification (cheap check on a sample if full check is expensive)
    if time.time() < DEADLINE - 2:
        # Verify coverage
        covered = set()
        for w in selected:
            ball_centers = _hamming_ball_centers(w, q, n, radius)
            for col in ball_centers.T:
                covered.add(_encode_word(tuple(col)))
        if len(covered) < total_space:
            # Fallback to direct sum if greedy failed
            if n >= 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                selected = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                selected = [list(w) for w in all_words]

    # Deduplicate
    seen = set()
    unique_selected = []
    for w in selected:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_selected.append(list(w))

    return unique_selected
# EVOLVE-BLOCK-END
