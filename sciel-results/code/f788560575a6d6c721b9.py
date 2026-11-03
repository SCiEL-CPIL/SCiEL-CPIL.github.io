"""Greedy set-cover with simulated annealing for K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _int_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def _word_to_idx(word, q, n):
    b = 0
    for x in word:
        b = b * q + x
    return b


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s

    total = q ** n
    pows = [q ** (n - 1 - i) for i in range(n)]

    # Precompute balls for all words
    all_balls = {}
    for idx in range(total):
        w = _int_to_word(idx, q, n)
        base = idx
        ball = [base]
        for i in range(n):
            pw = pows[i]
            w_i = w[i]
            for v in range(q):
                if v != w_i:
                    ball.append(base + (v - w_i) * pw)
        all_balls[idx] = ball

    # --- Strong initial seed: direct sum of ternary tetracode (for q=3, n>=4) ---
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed_words = [h + list(tail)
                      for h in tetracode
                      for tail in itertools.product(range(3), repeat=n - 4)]
        seed = [_word_to_idx(w, q, n) for w in seed_words]
    else:
        seed = []

    # Coverage count array
    coverage = [0] * total
    code_indices = list(seed)
    for idx in code_indices:
        for b in all_balls[idx]:
            coverage[b] += 1

    # Check if fully covered
    def is_fully_covered():
        return all(c > 0 for c in coverage)

    # Greedy fill if not covered
    while not is_fully_covered():
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            if idx in code_indices:
                continue
            g = 0
            for b in all_balls[idx]:
                if coverage[b] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_idx = idx
        if best_gain <= 0:
            break
        code_indices.append(best_idx)
        for b in all_balls[best_idx]:
            coverage[b] += 1

    # --- Simulated Annealing for reduction ---
    # Objective: minimize number of codewords while maintaining full coverage
    # Moves: remove a codeword (if coverage stays > 0), or swap (remove one, add another)
    
    def try_remove(c_idx):
        """Try removing c_idx. Returns True if removal is valid (coverage stays > 0)."""
        for b in all_balls[c_idx]:
            if coverage[b] == 1:
                return False
        # All balls have coverage >= 2, safe to remove
        for b in all_balls[c_idx]:
            coverage[b] -= 1
        code_indices.remove(c_idx)
        return True

    def try_add(c_idx):
        """Add c_idx to code."""
        code_indices.append(c_idx)
        for b in all_balls[c_idx]:
            coverage[b] += 1

    # SA parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_iters = 50000
    iters = 0
    best_size = len(code_indices)
    best_code = list(code_indices)
    best_coverage = list(coverage)

    while T > T_min and iters < max_iters:
        if time.time() > deadline - 10:
            break
        
        # Random move: try to remove a random codeword
        if len(code_indices) > 1:
            c_idx = random.choice(code_indices)
            
            # Check if removal is valid
            can_remove = True
            for b in all_balls[c_idx]:
                if coverage[b] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Perform removal
                for b in all_balls[c_idx]:
                    coverage[b] -= 1
                code_indices.remove(c_idx)
                
                # Track best
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                    best_coverage = list(coverage)
            else:
                # Try swap: remove c_idx, add a random other
                other_idx = random.randint(0, total - 1)
                if other_idx not in code_indices:
                    # Check if swap maintains coverage
                    # After removal, some balls might have coverage 0
                    # After addition, they need to be covered
                    uncovered_after_remove = []
                    for b in all_balls[c_idx]:
                        if coverage[b] == 1:
                            uncovered_after_remove.append(b)
                    
                    # Check if other_idx covers all uncovered
                    if all(b in all_balls[other_idx] for b in uncovered_after_remove):
                        # Perform swap
                        for b in all_balls[c_idx]:
                            coverage[b] -= 1
                        code_indices.remove(c_idx)
                        for b in all_balls[other_idx]:
                            coverage[b] += 1
                        code_indices.append(other_idx)
                    
                    # Track best (size same, but might enable future removals)
                    if len(code_indices) < best_size:
                        best_size = len(code_indices)
                        best_code = list(code_indices)
                        best_coverage = list(coverage)
        
        # Cooling
        T *= cooling
        iters += 1

    # Restore best code if current is worse
    if len(code_indices) > best_size:
        code_indices = list(best_code)
        coverage = list(best_coverage)

    # Final greedy fill if needed
    while not is_fully_covered():
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            if idx in code_indices:
                continue
            g = 0
            for b in all_balls[idx]:
                if coverage[b] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_idx = idx
        if best_gain <= 0:
            break
        code_indices.append(best_idx)
        for b in all_balls[best_idx]:
            coverage[b] += 1

    # Convert back to words
    result = [_int_to_word(idx, q, n) for idx in code_indices]
    return result
# EVOLVE-BLOCK-END
