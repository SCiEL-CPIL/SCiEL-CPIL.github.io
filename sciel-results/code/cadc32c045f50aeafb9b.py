"""Sumset/difference-set exponent lower bound maximization via FFT + greedy + SA."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def _elapsed():
        return time.time() - start_time

    def _score_set(U):
        """Compute score using FFT for sumset and difference set."""
        U = sorted(set(U))
        if 0 not in U:
            return -1.0
        n = len(U)
        if n < 2 or max(U) == 0:
            return -1.0
        M = max(U)
        # Sumset: U+U ranges from 0 to 2M
        # Difference set: U-U ranges from -M to M
        # Use FFT: indicator of U convolved with itself
        size = 2 * M + 1
        # Next power of 2 for FFT
        fft_size = 1
        while fft_size < 2 * size:
            fft_size <<= 1
        arr = np.zeros(fft_size, dtype=np.float64)
        for v in U:
            arr[v] = 1.0
        # Convolution: arr * arr gives sumset indicator (count of representations)
        conv = np.fft.irfft(np.fft.rfft(arr) ** 2)
        # Sumset: indices 0..2M where conv > 0.5
        plus_count = np.sum(conv[:size] > 0.5)
        # Difference set: shift conv to handle negative indices
        # U[i]-U[j] = (U[i] + (M - U[j])) - M, so use arr shifted
        arr2 = np.zeros(fft_size, dtype=np.float64)
        for v in U:
            arr2[M + v] = 1.0  # shift by M
        conv2 = np.fft.irfft(np.fft.rfft(arr2) ** 2)
        # Differences range from 0 to 2M in shifted space, actual from -M to M
        minus_count = np.sum(conv2[M:M + size] > 0.5)
        if plus_count == 0:
            return -1.0
        ratio = minus_count / plus_count
        if ratio <= 1:
            return -1.0
        denom = math.log2(2 * M + 1)
        if denom <= 0:
            return -1.0
        return 1 + math.log2(ratio) / denom

    def _score_set_small(U):
        """Direct computation for small sets (faster for n < 30)."""
        U = sorted(set(U))
        if 0 not in U:
            return -1.0
        n = len(U)
        if n < 2 or max(U) == 0:
            return -1.0
        plus_set = set()
        minus_set = set()
        for i in range(n):
            for j in range(n):
                plus_set.add(U[i] + U[j])
                minus_set.add(U[i] - U[j])
        ratio = len(minus_set) / len(plus_set)
        if ratio <= 1:
            return -1.0
        denom = math.log2(2 * max(U) + 1)
        if denom <= 0:
            return -1.0
        return 1 + math.log2(ratio) / denom

    def _eval(U):
        """Choose evaluation method based on set size."""
        if len(U) <= 25 and max(U) <= 200:
            return _score_set_small(U)
        elif max(U) <= 50000:
            return _score_set(U)
        else:
            return _score_set_small(U)

    # Phase 1: Greedy construction
    # Start with {0, 1} and greedily add elements that maximize score
    def _greedy_construct(max_n=60, max_val=500):
        U = [0, 1]
        current_score = _eval(U)
        for step in range(max_n - 2):
            if _elapsed() > deadline - 30:
                break
            best_val = None
            best_score = current_score
            # Try candidate positions
            max_u = max(U)
            # Focus on positions that create new differences
            candidates = set()
            # Positions near existing elements
            for u in U:
                for d in range(1, min(20, max_u + 1)):
                    candidates.add(u + d)
                    if u - d > 0:
                        candidates.add(u - d)
            # Also try some spread-out positions
            for d in range(2, min(50, max_u + 1)):
                candidates.add(max_u + d)
            # Limit candidates for speed
            if len(candidates) > 200:
                candidates = random.sample(list(candidates), 200)
            for val in sorted(candidates):
                if val <= 0 or val in U or val > max_val:
                    continue
                U_new = sorted(U + [val])
                s = _eval(U_new)
                if s > best_score:
                    best_score = s
                    best_val = val
            if best_val is None:
                break
            U.append(best_val)
            U.sort()
            current_score = best_score
        return U, current_score

    # Phase 2: Simulated annealing
    def _simulated_annealing(U, max_iter=5000):
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        current_score = _eval(U)
        if current_score < 0:
            return U, current_score
        best_U = U[:]
        best_score = current_score
        n = len(U)
        T = 0.05
        T_min = 0.001
        cooling = 0.9995

        for it in range(max_iter):
            if _elapsed() > deadline - 10:
                break
            U_new = U[:]
            op = random.random()
            if op < 0.25 and len(U) > 3:
                # Remove random element (keep 0)
                idx = random.choice([i for i in range(len(U)) if U[i] != 0])
                U_new = [U[i] for i in range(len(U)) if i != idx]
            elif op < 0.50:
                # Add random element
                max_u = max(U) if U else 1
                new_val = random.randint(1, max_u * 2 + 50)
                U_new.append(new_val)
            elif op < 0.75:
                # Swap
                if len(U) > 3:
                    idx = random.choice([i for i in range(len(U)) if U[i] != 0])
                    U_new = [U[i] for i in range(len(U)) if i != idx]
                    max_u = max(U_new) if U_new else 1
                    new_val = random.randint(1, max_u * 2 + 50)
                    U_new.append(new_val)
            else:
                # Shift
                idx = random.choice([i for i in range(len(U)) if U[i] != 0])
                delta = random.randint(-max(1, U[idx] // 5), max(1, U[idx] // 5))
                new_val = U[idx] + delta
                if new_val > 0:
                    U_new[idx] = new_val
                else:
                    continue

            U_new = sorted(set(U_new))
            if 0 not in U_new or len(U_new) < 2 or max(U_new) > 100000:
                continue
            new_score = _eval(U_new)
            if new_score > best_score:
                best_score = new_score
                best_U = U_new[:]
                U = U_new[:]
                current_score = new_score
            elif new_score > current_score:
                U = U_new[:]
                current_score = new_score
            else:
                delta_s = new_score - current_score
                if T > T_min and random.random() < math.exp(delta_s / T):
                    U = U_new[:]
                    current_score = new_score
            T *= cooling

        return best_U, best_score

    # Phase 3: Targeted local search on specific positions
    def _local_search_positions(U, max_iter=3000):
        """Try shifting individual elements to nearby positions."""
        U = sorted(set(U))
        current_score = _eval(U)
        if current_score < 0:
            return U, current_score
        best_U = U[:]
        best_score = current_score

        for it in range(max_iter):
            if _elapsed() > deadline - 10:
                break
            idx = random.choice([i for i in range(len(U)) if U[i] != 0])
            # Try a few nearby positions
            found_improvement = False
            for delta in random.sample(range(-50, 51), min(10, 101)):
                new_val = U[idx] + delta
                if new_val <= 0 or new_val in U:
                    continue
                U_new = U[:]
                U_new[idx] = new_val
                U_new = sorted(set(U_new))
                if 0 not in U_new or len(U_new) < 2:
                    continue
                s = _eval(U_new)
                if s > best_score:
                    best_score = s
                    best_U = U_new[:]
                    U = U_new[:]
                    current_score = s
                    found_improvement = True
                    break
            if not found_improvement:
                # Small random perturbation to escape
                if random.random() < 0.3:
                    idx2 = random.choice([i for i in range(len(U)) if U[i] != 0])
                    delta2 = random.randint(-30, 30)
                    new_val2 = U[idx2] + delta2
                    if new_val2 > 0 and new_val2 not in U:
                        U[idx2] = new_val2
                        U = sorted(set(U))
                        current_score = _eval(U)

        return best_U, best_score

    # Main search
    best_score = -1.0
    best_U = [0, 1, 3]

    # Try simple Golomb rulers (consecutive integers)
    for k in range(3, 30):
        if _elapsed() > deadline - 60:
            break
        U = list(range(k))
        s = _eval(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Greedy construction with different parameters
    for trial in range(5):
        if _elapsed() > deadline - 40:
            break
        random.seed(trial * 42 + 7)
        max_n = random.randint(20, 60)
        max_val = random.randint(100, 500)
        U, s = _greedy_construct(max_n=max_n, max_val=max_val)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Simulated annealing from best
    if _elapsed() < deadline - 30:
        U, s = _simulated_annealing(best_U, max_iter=4000)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Position-based local search
    if _elapsed() < deadline - 15:
        U, s = _local_search_positions(best_U, max_iter=3000)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # More SA rounds
    for rnd in range(3):
        if _elapsed() > deadline - 10:
            break
        U, s = _simulated_annealing(best_U, max_iter=2000)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Final polish
    if _elapsed() < deadline - 5:
        U, s = _local_search_positions(best_U, max_iter=2000)
        if s > best_score:
            best_score = s
            best_U = U[:]

    return best_U
# EVOLVE-BLOCK-END
