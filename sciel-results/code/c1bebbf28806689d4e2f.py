import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 75.0

    # Start with Reed-Muller code (32 words, distance 8)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)

    rng = random.Random(42)
    best_size = len(words)
    best_words = list(words)

    def can_add(w, code):
        for c in code:
            if (w ^ c).bit_count() < MIN_DIST:
                return False
        return True

    def try_add_random(code, rng, attempts=300):
        for _ in range(attempts):
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                return True
        return False

    def try_flip(code, rng):
        idx = rng.randrange(len(code))
        bit = rng.randrange(N)
        new_w = code[idx] ^ (1 << bit)
        for j, c in enumerate(code):
            if j != idx and (new_w ^ c).bit_count() < MIN_DIST:
                return False
        code[idx] = new_w
        return True

    def try_double_flip(code, rng):
        idx = rng.randrange(len(code))
        b1 = rng.randrange(N)
        b2 = rng.randrange(N)
        if b1 == b2:
            b2 = (b2 + 1) % N
        new_w = code[idx] ^ (1 << b1) ^ (1 << b2)
        for j, c in enumerate(code):
            if j != idx and (new_w ^ c).bit_count() < MIN_DIST:
                return False
        code[idx] = new_w
        return True

    def try_swap(code, rng):
        idx = rng.randrange(len(code))
        old_w = code[idx]
        code.pop(idx)
        for _ in range(200):
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                return True
        code.insert(idx, old_w)
        return False

    def kick(code, rng, k=3):
        removed = []
        for _ in range(min(k, len(code))):
            idx = rng.randrange(len(code))
            removed.append(code.pop(idx))
        for w in removed:
            if can_add(w, code):
                code.append(w)
        try_add_random(code, rng, attempts=200)

    # Phase 1: Greedy add
    while time.time() - start_time < time_limit * 0.25:
        if not try_add_random(words, rng, attempts=500):
            break

    if len(words) > best_size:
        best_size = len(words)
        best_words = list(words)

    # Phase 2: Local search
    while time.time() - start_time < time_limit:
        move = rng.random()
        if move < 0.25:
            try_add_random(words, rng, attempts=100)
        elif move < 0.5:
            try_flip(words, rng)
        elif move < 0.7:
            try_double_flip(words, rng)
        elif move < 0.85:
            try_swap(words, rng)
        else:
            kick(words, rng, k=rng.randint(2, 5))

        if len(words) > best_size:
            best_size = len(words)
            best_words = list(words)

    if len(best_words) >= len(words):
        return best_words
    return words
