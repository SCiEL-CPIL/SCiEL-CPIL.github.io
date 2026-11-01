"""Rewritten program using LFSR-based candidate generation and refined local search."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    best_score = 0
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        sumset = set()
        diffset = set()
        for i in range(n):
            a = U[i]
            for j in range(n):
                b = U[j]
                sumset.add(a + b)
                diffset.add(a - b)
        n_sum = len(sumset)
        n_diff = len(diffset)
        if n_sum == 0:
            return -1
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def is_golomb(U):
        U = sorted(U)
        n = len(U)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Phase 1: Known Golomb rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52, 60],
    ]
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)

    # Phase 2: LFSR-based candidate generation
    # Linear Feedback Shift Register with primitive polynomials
    def lfsr_positions(primitive_poly, n_bits, seed_val, n_positions):
        """
        Generate positions of 1s in an LFSR sequence.
        primitive_poly: integer representing the polynomial (e.g., 0b1011 for x^3+x+1)
        n_bits: register size
        seed_val: initial state
        n_positions: how many 1s to collect (positions where output is 1)
        """
        if seed_val == 0:
            seed_val = 1
        state = seed_val
        poly = primitive_poly
        positions = []
        # Generate n_bits * 2 outputs to get a full period
        total_outputs = n_bits * 2
        for step in range(total_outputs):
            # Output is the LSB
            output = state & 1
            if output == 1:
                positions.append(step)
            # Shift and feedback
            feedback = (state >> (n_bits - 1)) ^ ((state & poly) >> (n_bits - 2) if n_bits > 1 else 0)
            state = (state << 1) & ((1 << n_bits) - 1)
            if feedback:
                state |= 1
        # Return the first n_positions 1s, or all if fewer
        return positions[:n_positions]

    # Try various LFSR configurations
    lfsr_configs = [
        # (poly, n_bits, seed, n_positions)
        (0b1011, 3, 1, 5), (0b1011, 3, 1, 6), (0b1011, 3, 1, 7),
        (0b1011, 3, 2, 5), (0b1011, 3, 2, 6), (0b1011, 3, 2, 7),
        (0b1011, 3, 3, 5), (0b1011, 3, 3, 6), (0b1011, 3, 3, 7),
        (0b1011, 3, 4, 5), (0b1011, 3, 4, 6), (0b1011, 3, 4, 7),
        (0b11011, 4, 1, 6), (0b11011, 4, 1, 7), (0b11011, 4, 1, 8), (0b11011, 4, 1, 9),
        (0b11011, 4, 2, 6), (0b11011, 4, 2, 7), (0b11011, 4, 2, 8), (0b11011, 4, 2, 9),
        (0b11011, 4, 3, 6), (0b11011, 4, 3, 7), (0b11011, 4, 3, 8), (0b11011, 4, 3, 9),
        (0b1110111, 5, 1, 8), (0b1110111, 5, 1, 9), (0b1110111, 5, 1, 10), (0b1110111, 5, 1, 11),
        (0b1110111, 5, 2, 8), (0b1110111, 5, 2, 9), (0b1110111, 5, 2, 10), (0b1110111, 5, 2, 11),
        (0b1110111, 5, 3, 8), (0b1110111, 5, 3, 9), (0b1110111, 5, 3, 10), (0b1110111, 5, 3, 11),
        (0b11110001101, 7, 1, 10), (0b11110001101, 7, 1, 11), (0b11110001101, 7, 1, 12),
        (0b11110001101, 7, 1, 13), (0b11110001101, 7, 2, 10), (0b11110001101, 7, 2, 11),
        (0b11110001101, 7, 2, 12), (0b11110001101, 7, 2, 13),
        (0b111110001101, 8, 1, 12), (0b111110001101, 8, 1, 13), (0b111110001101, 8, 1, 14),
        (0b111110001101, 8, 2, 12), (0b111110001101, 8, 2, 13), (0b111110001101, 8, 2, 14),
        (0b111111100001101, 10, 1, 14), (0b111111100001101, 10, 1, 15),
        (0b111111100001101, 10, 1, 16), (0b111111100001101, 10, 2, 14),
    ]

    for poly, n_bits, seed, n_pos in lfsr_configs:
        if time.time() > deadline - 10:
            break
        try:
            pos = lfsr_positions(poly, n_bits, seed, n_pos)
            if len(pos) >= 3:
                # Ensure 0 is included
                if 0 not in pos:
                    pos = [0] + [p for p in pos if p != 0]
                U = sorted(set(pos))
                if len(U) >= 3 and 0 in U:
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
        except Exception:
            continue

    # Phase 3: Systematic search over small sets with wider ranges
    for max_u in range(1, 60):
        if time.time() > deadline - 15:
            break
        for size in range(3, min(14, max_u + 1)):
            if time.time() > deadline - 15:
                break
            candidates = []
            # Greedy Golomb construction
            for start2 in [1, 2, 3, 4, 5]:
                if start2 >= max_u:
                    continue
                gr_candidate = [0, start2]
                used_diffs = {start2}
                for _ in range(size - 2):
                    best_next = -1
                    for candidate in range(gr_candidate[-1] + 1, max_u + 1):
                        valid = True
                        for existing in gr_candidate:
                            d = candidate - existing
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_next = candidate
                            break
                    if best_next == -1:
                        break
                    gr_candidate.append(best_next)
                    for existing in gr_candidate[:-1]:
                        used_diffs.add(gr_candidate[-1] - existing)
                if len(gr_candidate) == size:
                    candidates.append(gr_candidate)
            # Powers of 2
            if max_u >= 2 ** (size - 1):
                geo = [2 ** i for i in range(size)]
                if geo[-1] <= max_u:
                    candidates.append(geo)
            # Fibonacci
            fib = [0, 1]
            while len(fib) < size:
                fib.append(fib[-1] + fib[-2])
            if fib[-1] <= max_u:
                candidates.append(fib[:size])
            # Evaluate
            seen = set()
            for cand in candidates:
                cand = sorted(set(cand))
                if len(cand) != size or 0 not in cand:
                    continue
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)

    # Phase 4: Enhanced local search with gap-based representation and wider moves
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 500

        def to_gaps(U):
            """Convert sorted set to gap representation."""
            U = sorted(U)
            if not U:
                return []
            gaps = [U[0]]
            for i in range(1, len(U)):
                gaps.append(U[i] - U[i-1])
            return gaps

        def from_gaps(gaps):
            """Convert gap representation to sorted set."""
            if not gaps:
                return []
            U = [gaps[0]]
            for i in range(1, len(gaps)):
                U.append(U[-1] + gaps[i])
            return U

        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            improved = False

            # Try adding an element with wider range
            if len(current) < 18:
                max_cur = max(current)
                candidates_to_add = set()
                for el in current:
                    for delta in range(-8, 9):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 25):
                    candidates_to_add.add(new_el)
                for new_el in sorted(candidates_to_add):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break

            # Try replacing an element with wider range
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    for new_el in range(max(1, old_el - 12), min(max_cur + 15, old_el + 13)):
                        if new_el == old_el or new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != len(current):
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                    if improved:
                        break

            # Try gap-based moves: adjust individual gaps
            if not improved and len(current) >= 3:
                gaps = to_gaps(current)
                n_gaps = len(gaps)
                for g_idx in range(n_gaps):
                    if g_idx == 0:
                        continue  # Keep first element as 0
                    old_gap = gaps[g_idx]
                    for new_gap in range(max(1, old_gap - 5), min(max(current) + 10, old_gap + 6)):
                        if new_gap == old_gap:
                            continue
                        new_gaps = list(gaps)
                        new_gaps[g_idx] = new_gap
                        cand = from_gaps(new_gaps)
                        if len(cand) != len(current) or 0 not in cand:
                            continue
                        # Check for duplicates
                        if len(set(cand)) != len(cand):
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                    if improved:
                        break

            # Try 2-swap moves with wider range
            if len(current) >= 4 and not improved:
                max_cur = max(current)
                n = len(current)
                for i in range(n):
                    if current[i] == 0:
                        continue
                    for j in range(i + 1, n):
                        if current[j] == 0:
                            continue
                        old_i = current[i]
                        old_j = current[j]
                        for new_i in range(max(1, old_i - 8), min(max_cur + 10, old_i + 9)):
                            if new_i == old_i or new_i == old_j or new_i in current:
                                continue
                            for new_j in range(max(1, old_j - 8), min(max_cur + 10, old_j + 9)):
                                if new_j == old_j or new_j == old_i or new_j == new_i or new_j in current:
                                    continue
                                cand = list(current)
                                cand[i] = new_i
                                cand[j] = new_j
                                cand = sorted(set(cand))
                                if 0 not in cand or len(cand) != n:
                                    continue
                                s = compute_score(cand)
                                if s > current_score + 1e-15:
                                    current = list(cand)
                                    current_score = s
                                    if s > best_score:
                                        best_score = s
                                        best_set = list(cand)
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

            # Try removing an element
            if len(current) > 3 and not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

    # Phase 5: Enhanced random restarts with diverse initialization strategies
    if time.time() < deadline - 5:
        random.seed(12345)
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            size = random.randint(4, 14)
            max_u = random.randint(15, 100)
            
            # Strategy 1: LFSR-like greedy construction
            if restart % 3 == 0:
                gr = [0, 1]
                used_diffs = {1}
                for _ in range(size - 2):
                    best_next = -1
                    for candidate in range(2, max_u + 1):
                        valid = True
                        for existing in gr:
                            d = candidate - existing
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_next = candidate
                            break
                    if best_next == -1:
                        break
                    gr.append(best_next)
                    for existing in gr[:-1]:
                        used_diffs.add(gr[-1] - existing)
                if len(gr) >= 3:
                    s = compute_score(gr)
                    if s > best_score:
                        best_score = s
                        best_set = list(gr)
            
            # Strategy 2: Random gaps with refinement
            elif restart % 3 == 1:
                gaps = [0] + [random.randint(1, max(1, max_u // size)) for _ in range(size - 1)]
                U = from_gaps(gaps)
                if len(U) == size and 0 in U and max(U) <= max_u:
                    # Quick local refinement
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        improved_local = False
                        for i in range(1, len(U)):
                            for delta in range(-3, 4):
                                new_val = U[i] + delta
                                if new_val <= 0 or new_val in U:
                                    continue
                                cand = list(U)
                                cand[i] = new_val
                                cand = sorted(cand)
                                if len(set(cand)) != len(cand):
                                    continue
                                s = compute_score(cand)
                                if s > best_score + 1e-15:
                                    U = list(cand)
                                    improved_local = True
                                    break
                            if improved_local:
                                break
                        if not improved_local:
                            break
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
            
            # Strategy 3: Fibonacci-like with random perturbations
            else:
                fib = [0, 1]
                while len(fib) < size:
                    fib.append(fib[-1] + fib[-2])
                if fib[-1] <= max_u:
                    U = list(fib[:size])
                    # Random perturbations
                    for _ in range(10):
                        if time.time() > deadline - 5:
                            break
                        idx = random.randint(1, len(U) - 1)
                        delta = random.randint(-2, 3)
                        new_val = U[idx] + delta
                        if new_val <= 0 or new_val in U:
                            continue
                        cand = list(U)
                        cand[idx] = new_val
                        cand = sorted(cand)
                        if len(set(cand)) != len(cand):
                            continue
                        s = compute_score(cand)
                        if s > best_score + 1e-15:
                            U = list(cand)
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)

    return list(best_set)
# EVOLVE-BLOCK-END
