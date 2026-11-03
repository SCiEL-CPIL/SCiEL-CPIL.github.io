"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    # Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
    deadline = time.time() + 118.0

    # For the specific case q=3, n=6, radius=1, we aim for the known optimum of 73.
    # Strategy: Fast greedy set cover + aggressive local search with multiple restarts.

    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline)

    # Generic fallback: direct sum construction or greedy
    if n <= 4:
        # Small case: brute force or direct
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        else:
            # For n < 4, use a simple greedy
            pass

    # Generic greedy set cover
    return _greedy_set_cover(q, n, radius, deadline)


def _solve_ternary_6_1(deadline):
    q, n, radius = 3, 6, 1
    total = q ** n  # 729

    # Precompute balls for all 729 words efficiently
    # For radius 1, each ball has 1 + 6*2 = 13 words
    balls = np.zeros((total, 13), dtype=np.int32)
    for i in range(total):
        # Get digits of word i
        digits = [(i // (3 ** (n - 1 - pos))) % 3 for pos in range(n)]
        ball_indices = [i]
        for pos in range(n):
            for val in range(3):
                if val != digits[pos]:
                    new_digits = digits[:]
                    new_digits[pos] = val
                    new_idx = sum(d * (3 ** (n - 1 - p)) for p, d in enumerate(new_digits))
                    ball_indices.append(new_idx)
        balls[i, :] = ball_indices

    # Ball membership: for each word, which codewords cover it
    # We'll compute this on the fly or precompute
    # For speed, let's precompute the reverse mapping
    # covered_by[word] = list of codeword indices that cover this word
    covered_by = [[] for _ in range(total)]
    for i in range(total):
        for b_idx in balls[i]:
            covered_by[b_idx].append(i)

    # Greedy initialization using the tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            cw = h + list(tail)
            idx = sum(v * (3 ** (n - 1 - p)) for p, v in enumerate(cw))
            initial_indices.append(idx)

    # Start with greedy completion
    def greedy_complete(initial_set, time_limit):
        covered = np.zeros(total, dtype=bool)
        cw_set = set(initial_set)
        for idx in cw_set:
            for b_idx in balls[idx]:
                covered[b_idx] = True
        
        uncovered_count = total - covered.sum()
        while uncovered_count > 0 and time.time() < time_limit:
            best_idx = -1
            best_gain = -1
            for idx in range(total):
                if idx in cw_set:
                    continue
                gain = 0
                for b_idx in balls[idx]:
                    if not covered[b_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx < 0 or best_gain <= 0:
                break
            cw_set.add(best_idx)
            for b_idx in balls[best_idx]:
                covered[b_idx] = True
            uncovered_count -= best_gain
        return cw_set

    # Try multiple different initial sets and keep the best
    best_result = None
    best_size = float('inf')
    
    # Initial set 1: Tetracode construction
    t1_start = time.time()
    cw_set1 = greedy_complete(initial_indices, deadline - 20.0)
    if len(cw_set1) < best_size:
        best_size = len(cw_set1)
        best_result = cw_set1.copy()

    # Local search to reduce
    if time.time() < deadline - 15.0:
        improved = _local_search_reduce(best_result, balls, covered_by, total, deadline - 10.0)
        if len(improved) < best_size:
            best_size = len(improved)
            best_result = improved.copy()

    # Try random restarts if time allows
    if time.time() < deadline - 25.0:
        random.seed(42)
        for restart in range(5):
            if time.time() >= deadline - 15.0:
                break
            # Create a perturbed initial set
            perturbed = set(initial_indices)
            # Remove some random codewords
            to_remove = random.sample(list(perturbed), min(10, len(perturbed)))
            for idx in to_remove:
                perturbed.discard(idx)
            # Add some random codewords
            for _ in range(5):
                perturbed.add(random.randint(0, total - 1))
            
            cw_set_r = greedy_complete(list(perturbed), deadline - 20.0)
            if len(cw_set_r) < best_size:
                best_size = len(cw_set_r)
                best_result = cw_set_r.copy()
            
            # Local search on this result
            if time.time() < deadline - 10.0:
                improved_r = _local_search_reduce(best_result, balls, covered_by, total, deadline - 10.0)
                if len(improved_r) < best_size:
                    best_size = len(improved_r)
                    best_result = improved_r.copy()

    # Final polish
    if time.time() < deadline - 5.0 and best_result is not None:
        final = _local_search_reduce(best_result, balls, covered_by, total, deadline - 3.0)
        if len(final) < best_size:
            best_result = final

    # Convert back to lists
    result = []
    for idx in best_result:
        digits = [(idx // (3 ** (n - 1 - pos))) % 3 for pos in range(n)]
        result.append(digits)

    return result


def _local_search_reduce(cw_set, balls, covered_by, total, deadline):
    """
    Aggressive local search to reduce the number of codewords.
    Uses coverage counts and tries to remove redundant codewords.
    """
    cw_set = set(cw_set)
    
    # Build coverage count array
    coverage_count = np.zeros(total, dtype=np.int32)
    for idx in cw_set:
        for b_idx in balls[idx]:
            coverage_count[b_idx] += 1

    def can_remove(idx):
        """Check if codeword idx can be removed without uncovering any word."""
        for b_idx in balls[idx]:
            if coverage_count[b_idx] == 1:
                return False
        return True

    def remove_codeword(idx):
        for b_idx in balls[idx]:
            coverage_count[b_idx] -= 1

    # Phase 1: Remove all obviously removable codewords (iterate until stable)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(cw_set):
            if can_remove(idx):
                cw_set.remove(idx)
                remove_codeword(idx)
                changed = True

    # Phase 2: Try swaps - remove one codeword, add another that covers the gaps
    if time.time() < deadline - 5.0:
        for _ in range(100):  # Limited number of swap attempts
            if time.time() >= deadline - 5.0:
                break
            
            # Find a codeword that, if removed, leaves few uncovered words
            best_remove = None
            best_uncovered_count = float('inf')
            
            for idx in cw_set:
                # Count how many words would become uncovered
                uncovered = 0
                for b_idx in balls[idx]:
                    if coverage_count[b_idx] == 1:
                        uncovered += 1
                if uncovered < best_uncovered_count:
                    best_uncovered_count = uncovered
                    best_remove = idx
            
            if best_remove is None:
                break
            
            if best_uncovered_count == 0:
                # Can remove without adding
                cw_set.remove(best_remove)
                remove_codeword(best_remove)
                continue
            
            # Find a codeword that covers the most of the uncovered words
            # First, temporarily mark what would be uncovered
            temp_uncovered = set()
            for b_idx in balls[best_remove]:
                if coverage_count[b_idx] == 1:
                    temp_uncovered.add(b_idx)
            
            best_add = None
            best_coverage_gain = -1
            for idx in range(total):
                if idx in cw_set:
                    continue
                # Count how many of the temp_uncovered words this codeword covers
                gain = 0
                for b_idx in balls[idx]:
                    if b_idx in temp_uncovered:
                        gain += 1
                if gain > best_coverage_gain:
                    best_coverage_gain = gain
                    best_add = idx
            
            if best_add is not None and best_coverage_gain == len(temp_uncovered):
                # This swap works: remove best_remove, add best_add
                cw_set.remove(best_remove)
                remove_codeword(best_remove)
                cw_set.add(best_add)
                for b_idx in balls[best_add]:
                    coverage_count[b_idx] += 1
            else:
                # No good swap found, break
                break

    # Phase 3: Final cleanup - remove any newly removable codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(cw_set):
            if can_remove(idx):
                cw_set.remove(idx)
                remove_codeword(idx)
                changed = True

    return cw_set


def _greedy_set_cover(q, n, radius, deadline):
    total = q ** n

    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        return tuple(digits)

    # Precompute balls
    balls = []
    for i in range(total):
        w = idx_to_tuple(i)
        ball = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(tuple(new_w))
        idxs = [tuple_to_idx(b) for b in ball]
        balls.append(idxs)

    covered = np.zeros(total, dtype=bool)
    codeword_set = set()

    while not covered.all() and time.time() < deadline - 5.0:
        best_idx = -1
        best_gain = -1
        for idx in range(total):
            if idx in codeword_set:
                continue
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0:
            break
        codeword_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True

    result = []
    for idx in codeword_set:
        result.append(list(idx_to_tuple(idx)))
    return result
# EVOLVE-BLOCK-END
