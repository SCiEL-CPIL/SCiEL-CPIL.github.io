"""Construct largest binary code of length 18, distance >= 6."""
import time
import random

N = 18
MIN_DIST = 6
TOTAL_WORDS = 1 << N  # 262144


def _reed_muller_seed():
    """First-order Reed-Muller code on 16 coords, extended with 2 zero bits. 32 words, dist 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    rng = random.Random(42)

    # Start with Reed-Muller seed (32 words, all pairwise distance 8)
    code = _reed_muller_seed()
    code_set = set(code)

    # Greedy random addition: try random candidates, add if valid
    # The code will grow until we hit the deadline or stop finding valid candidates
    while time.time() < deadline:
        # Process a batch of candidates
        for _ in range(10000):
            if time.time() >= deadline:
                break
            cand = rng.randrange(TOTAL_WORDS)
            if cand in code_set:
                continue
            # Check minimum distance constraint
            valid = True
            for w in code:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                code.append(cand)
                code_set.add(cand)

    return code
