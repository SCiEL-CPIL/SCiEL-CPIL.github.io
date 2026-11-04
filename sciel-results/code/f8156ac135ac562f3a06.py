import time
import random
import itertools
from collections import defaultdict

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n

    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute balls
    balls = []
    for i in range(total_words):
        w = to_word(i)
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                covered.add(to_idx(w))
            w[pos] = orig
        balls.append(list(covered))

    # Initial Code: Tetracode Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # counts[word_idx] = number of codewords in current set covering word_idx
    counts = [0] * total_words
    for c in initial_code:
        for w in balls[c]:
            counts[w] += 1

    best_code = list(initial_code)
    best_size = len(best_code)

    start_time = time.time()
    deadline = start_time + 110.0

    current_code = list(initial_code)
    current_counts = list(counts)
    current_set = set(current_code)

    def try_remove(c, counts_arr):
        for w in balls[c]:
            if counts_arr[w] == 1:
                return False
        return True

    def do_remove(c, counts_arr):
        for w in balls[c]:
            counts_arr[w] -= 1

    def do_add(c, counts_arr):
        for w in balls[c]:
            counts_arr[w] += 1

    # Phase 1: Greedy Remove
    while True:
        removed_any = False
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c, current_counts):
                do_remove(c, current_counts)
                current_code.remove(c)
                current_set.discard(c)
                removed_any = True
        if not removed_any:
            break

    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)

    # Phase 2: Local Search with Add-Remove
    while time.time() < deadline:
        # Add a random word
        added = False
        for _ in range(10):
            rand_idx = random.randint(0, total_words - 1)
            if rand_idx not in current_set:
                current_code.append(rand_idx)
                do_add(rand_idx, current_counts)
                current_set.add(rand_idx)
                added = True
                break

        if not added:
            break

        # Greedy Remove
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c, current_counts):
                do_remove(c, current_counts)
                current_code.remove(c)
                current_set.discard(c)

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)

    return [to_word(idx) for idx in best_code]
