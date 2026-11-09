import time
import random
import math
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55  # 5s safety margin

    # Thue-Morse initialization
    def thue_morse(n):
        seq = []
        for i in range(n):
            c = bin(i).count('1')
            if c % 2 == 0:
                seq.append(1)
            else:
                seq.append(-1)
        return seq

    seq = thue_morse(n)

    def get_cost(s):
        max_disc = 0
        nn = n
        for d in range(1, nn + 1):
            for r in range(d):
                curr = 0
                min_v = 0
                max_v = 0
                idx = r
                while idx < nn:
                    curr += s[idx]
                    if curr < min_v:
                        min_v = curr
                    if curr > max_v:
                        max_v = curr
                    idx += d
                disc = max_v - min_v
                if disc > max_disc:
                    max_disc = disc
        return max_disc

    best_seq = seq[:]
    best_cost = get_cost(seq)
    current_seq = seq[:]
    current_cost = best_cost

    rng = random.Random(42)

    # Simulated Annealing
    T = 15.0
    cooling = 0.999
    min_T = 0.01
    patience = 3000
    no_improve = 0

    # Precompute AP indices for faster evaluation
    # Store (start, step) -> list of indices
    # But that's too much memory. Instead, we'll use the direct method.
    # However, we can optimize by only checking APs that are likely to matter.
    # Actually, let's just use the direct method but with early termination.

    def get_cost_fast(s):
        max_disc = 0
        nn = n
        # Skip steps that are too large (few elements)
        for d in range(1, nn + 1):
            # If d > nn/2, each AP has at most 2 elements, max disc <= 2
            # Actually for d >= nn/2, APs have length 1 or 2
            # Length 1: disc = 1
            # Length 2: disc = |s[i]| + |s[i+d]| = 2 if same sign, 0 if diff
            # So max for d >= nn/2 is 2
            if d >= nn // 2:
                if max_disc < 2:
                    max_disc = 2
                break
            for r in range(d):
                curr = 0
                min_v = 0
                max_v = 0
                idx = r
                while idx < nn:
                    curr += s[idx]
                    if curr < min_v:
                        min_v = curr
                    if curr > max_v:
                        max_v = curr
                    # Early termination: if even reaching all remaining can't beat max_disc
                    # But this is complex. Just continue.
                    idx += d
                disc = max_v - min_v
                if disc > max_disc:
                    max_disc = disc
                    # Can't early terminate easily
        return max_disc

    # Let's use a smarter approach: vectorized numpy for the cost computation
    def get_cost_numpy(s):
        max_disc = 0
        nn = n
        s_arr = np.array(s)
        for d in range(1, nn + 1):
            for r in range(d):
                # Get indices r, r+d, r+2d, ...
                indices = range(r, nn, d)
                if len(indices) <= 1:
                    continue
                # Cumulative sum
                vals = s_arr[list(indices)]
                cumsum = np.cumsum(vals)
                disc = cumsum.max() - cumsum.min()
                if disc > max_disc:
                    max_disc = disc
        return max_disc

    # The numpy version might be slower due to list creation. Let's stick with Python but optimize.
    # Actually, let's try a different strategy: use a tabu-like approach with random flips
    # but focus on high-impact positions.

    while time.time() < deadline:
        # Random flip
        idx = rng.randrange(n)
        old_val = current_seq[idx]
        current_seq[idx] = -old_val

        new_cost = get_cost(current_seq)
        delta = new_cost - current_cost

        if delta < 0:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = current_seq[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability exp(-delta/T)
            if T > 0 and rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                no_improve += 1
            else:
                current_seq[idx] = old_val
                no_improve += 1

        T *= cooling
        if T < min_T:
            # Restart with new temperature from best solution
            T = 15.0
            current_seq = best_seq[:]
            current_cost = best_cost
            no_improve = 0

        if no_improve > patience:
            # Perturb the best solution
            for _ in range(5):
                idx = rng.randrange(n)
                best_seq[idx] *= -1
            current_seq = best_seq[:]
            current_cost = get_cost(current_seq)
            T = 15.0
            no_improve = 0

    return best_seq
