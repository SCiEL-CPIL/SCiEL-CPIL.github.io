import time
import random
from itertools import combinations


def construct():
    N = 18
    D = 6
    bc = int.bit_count

    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # --- Reed-Muller [16,5,8] code, extended by 2 zero bits -> 32 words, dist 8 ---
    def reed_muller_words():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    def is_valid_candidate(w, code):
        for c in code:
            if bc(w ^ c) < D:
                return False
        return True

    def greedy_from_seed(seed, order, dl):
        code = list(seed)
        code_set = set(code)
        for w in order:
            if time.time() > dl:
                break
            if w in code_set:
                continue
            if is_valid_candidate(w, code):
                code.append(w)
                code_set.add(w)
        return code

    def generate_by_weight(w):
        """Generate all N-bit words with exactly w bits set."""
        result = []
        for positions in combinations(range(N), w):
            word = 0
            for p in positions:
                word |= 1 << p
            result.append(word)
        return result

    best_code = []

    # --- Strategy 1: Reed-Muller seed + weight-ordered greedy ---
    # Start with the 32 RM words (distance 8), then greedily add more
    # by trying candidates in order of weight (4..14), which tends to
    # find valid codewords early and build a denser code.
    rm_seed = reed_muller_words()
    rm_seed_set = set(rm_seed)

    # Build weight-ordered candidate list, skipping RM words
    weight_order = []
    for w in range(4, 15):  # weights 4 through 14
        if time.time() > deadline:
            break
        for candidate in generate_by_weight(w):
            if candidate not in rm_seed_set:
                weight_order.append(candidate)

    code1 = greedy_from_seed(rm_seed, weight_order, deadline)
    if len(code1) > len(best_code):
        best_code = code1

    # --- Strategy 2: Empty seed + weight-ordered greedy (different packing) ---
    if time.time() < deadline - 5.0:
        # Regenerate order with slight randomization per weight for diversity
        weight_order2 = []
        for w in range(4, 15):
            if time.time() > deadline:
                break
            cands = generate_by_weight(w)
            random.shuffle(cands)
            weight_order2.extend(cands)

        code2 = greedy_from_seed([], weight_order2, deadline)
        if len(code2) > len(best_code):
            best_code = code2

    # --- Strategy 3: RM seed + random shuffle of remaining candidates ---
    if time.time() < deadline - 5.0:
        remaining = [w for w in range(1 << N) if w not in rm_seed_set]
        random.shuffle(remaining)
        code3 = greedy_from_seed(rm_seed, remaining, deadline)
        if len(code3) > len(best_code):
            best_code = code3

    # --- Strategy 4: Local search / swap improvement on best_code ---
    # Try to replace codewords with others to pack more densely.
    if time.time() < deadline - 3.0 and len(best_code) > 0:
        code = list(best_code)
        code_set = set(code)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try adding one more word by scanning a random sample
            for _ in range(20000):
                if time.time() > deadline - 1.0:
                    break
                w = random.randint(0, (1 << N) - 1)
                if w in code_set:
                    continue
                if is_valid_candidate(w, code):
                    code.append(w)
                    code_set.add(w)
                    improved = True

        if len(code) > len(best_code):
            best_code = code

    return best_code
