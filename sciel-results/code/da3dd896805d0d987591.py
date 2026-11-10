import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def evaluate(coeffs):
        if not is_valid(coeffs):
            return float('inf')
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    best_score = float('inf')
    best_poly = [1] * 12  # baseline: 12 ones, square has 23 nonzeros

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Precompute: for a given support (list of 12 indices) and sign vector,
    # compute the square nonzero count efficiently.
    # Key insight: P(x) = sum_{i in S} s_i * c_i * x^i
    # P^2 has terms at positions i+j for i,j in S.
    # The coefficient at position k in P^2 is sum_{i+j=k, i,j in S} s_i*c_i * s_j*c_j
    # We want to minimize the number of k where this sum != 0.

    # Represent state as (support, signs, mags) where:
    #   support: sorted list of 12 distinct positions in [0, 40]
    #   signs: list of 12 values in {-1, 1}
    #   mags: list of 12 values in {1, 2, 3, ...} (positive integers)
    # Coefficient at position support[i] = signs[i] * mags[i]

    def compute_square_count(support, signs, mags):
        """Compute number of nonzero terms in the square."""
        # Build the full coefficient array
        max_deg = max(support) if support else 0
        coeffs = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for i in range(len(support)):
            coeffs[support[i]] = signs[i] * mags[i]
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    # Simulated annealing over the discrete state space
    # State: (support_tuple, signs_tuple, mags_tuple)
    # Moves:
    #   1. Swap two support positions
    #   2. Move one support position to a new random position
    #   3. Flip one sign
    #   4. Flip two signs
    #   5. Change one magnitude (within 1..64)
    #   6. Swap two magnitudes
    #   7. Change two magnitudes

    def random_initial_state():
        support = sorted(random.sample(range(41), 12))
        signs = tuple(random.choice([-1, 1]) for _ in range(12))
        mags = tuple(random.choice([1, 1, 1, 2, 3]) for _ in range(12))
        return tuple(support), signs, mags

    def state_to_coeffs(support, signs, mags):
        max_deg = max(support)
        coeffs = [0] * (max_deg + 1)
        for i in range(12):
            coeffs[support[i]] = signs[i] * mags[i]
        # Trim trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def valid_state(support, signs, mags):
        if len(support) != 12:
            return False
        if len(set(support)) != 12:
            return False
        if any(s < 0 or s > 40 for s in support):
            return False
        if any(abs(s) != 1 for s in signs):
            return False
        if any(m < 1 or m > 64 for m in mags):
            return False
        return True

    # Start with some structured initial states
    initial_states = []

    # State 1: consecutive support, all +1, all mag 1
    sup = tuple(range(12))
    sig = tuple([1]*12)
    mag = tuple([1]*12)
    initial_states.append((sup, sig, mag))

    # State 2: powers of 2 support (max spread)
    sup = tuple(sorted(random.sample(range(41), 12)))
    sig = tuple([1]*12)
    mag = tuple([1]*12)
    initial_states.append((sup, sig, mag))

    # State 3: symmetric support around center
    center = 20
    offsets = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9]
    sup = tuple(sorted([center + o for o in offsets]))
    sig = tuple([1]*12)
    mag = tuple([1]*12)
    initial_states.append((sup, sig, mag))

    # State 4: random
    for _ in range(10):
        initial_states.append(random_initial_state())

    # Evaluate initial states
    for state in initial_states:
        support, signs, mags = state
        if not valid_state(support, signs, mags):
            continue
        cnt = compute_square_count(support, signs, mags)
        coeffs = state_to_coeffs(support, signs, mags)
        if is_valid(coeffs):
            try_poly(coeffs)

    # Simulated annealing
    best_state = None
    best_state_score = best_score

    # Find the best initial state
    for state in initial_states:
        support, signs, mags = state
        if not valid_state(support, signs, mags):
            continue
        cnt = compute_square_count(support, signs, mags)
        if cnt < best_state_score:
            best_state_score = cnt
            best_state = state

    if best_state is None:
        best_state = initial_states[0]
        best_state_score = compute_square_count(*best_state)

    current_support, current_signs, current_mags = best_state
    current_score = best_state_score

    # SA parameters
    T0 = 10.0
    T_end = 0.01
    total_steps = 50000

    for step in range(total_steps):
        if time.time() > deadline - 5:
            break

        # Temperature schedule (geometric cooling)
        progress = step / total_steps
        T = T0 * (T_end / T0) ** progress

        # Generate a random move
        move_type = random.randint(1, 7)
        new_support = list(current_support)
        new_signs = list(current_signs)
        new_mags = list(current_mags)

        if move_type == 1:  # Swap two support positions
            i, j = random.sample(range(12), 2)
            new_support[i], new_support[j] = new_support[j], new_support[i]
            new_support = tuple(sorted(new_support))
            # Re-sort signs and mags to match
            # Actually, we need to track which index goes where
            # Let's redo: pick two indices, swap their positions
            i, j = random.sample(range(12), 2)
            new_support = list(current_support)
            new_support[i], new_support[j] = new_support[j], new_support[i]
            new_support = tuple(sorted(new_support))
            # Remap signs and mags: the term that was at position i is now at position j and vice versa
            # This is complex; simpler: just swap the support values
            new_support = list(current_support)
            new_support[i], new_support[j] = new_support[j], new_support[i]
            new_support = tuple(sorted(new_support))
            # Signs and mags stay with their "slot" (index in the sorted support)
            # Actually this approach is messy. Let's use a different representation.

        # Simpler approach: represent state as a list of 12 (position, sign, mag) tuples
        # Then moves are straightforward.

        # Let me restructure: use a list of 12 (pos, sign, mag) triples
        pass

    # Let me restart with a cleaner representation
    # State: list of 12 (position, sign, mag) tuples, sorted by position

    def make_state(pos_sign_mag_list):
        """pos_sign_mag_list: list of 12 (pos, sign, mag) tuples"""
        sorted_list = sorted(pos_sign_mag_list, key=lambda x: x[0])
        return tuple(sorted_list)

    def state_score(state):
        positions = [s[0] for s in state]
        signs = [s[1] for s in state]
        mags = [s[2] for s in state]
        max_deg = max(positions)
        coeffs = np.zeros(max_deg + 1, dtype=np.int64)
        for p, s, m in state:
            coeffs[p] = s * m
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def state_to_coeffs(state):
        positions = [s[0] for s in state]
        max_deg = max(positions)
        coeffs = [0] * (max_deg + 1)
        for p, s, m in state:
            coeffs[p] = s * m
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def valid_state(state):
        if len(state) != 12:
            return False
        positions = [s[0] for s in state]
        if len(set(positions)) != 12:
            return False
        if any(p < 0 or p > 40 for p in positions):
            return False
        for p, s, m in state:
            if s not in (-1, 1):
                return False
            if m < 1 or m > 64:
                return False
        return True

    # Rebuild initial states in new format
    initial_states2 = []

    # State 1: consecutive
    initial_states2.append(make_state([(i, 1, 1) for i in range(12)]))

    # State 2: random spread
    positions = sorted(random.sample(range(41), 12))
    initial_states2.append(make_state([(p, 1, 1) for p in positions]))

    # State 3: symmetric
    center = 20
    offsets = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9]
    positions = sorted([center + o for o in offsets])
    initial_states2.append(make_state([(p, 1, 1) for p in positions]))

    # State 4: all same sign, spread
    positions = sorted(random.sample(range(41), 12))
    initial_states2.append(make_state([(p, 1, 1) for p in positions]))

    # State 5: alternating signs on consecutive
    initial_states2.append(make_state([(i, 1 if i % 2 == 0 else -1, 1) for i in range(12)]))

    # State 6: Fibonacci-like spacing
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    positions = [0]
    for i in range(1, 12):
        positions.append(min(positions[-1] + fibs[i], 40))
    # Deduplicate
    positions = sorted(set(positions))
    while len(positions) < 12:
        positions.append(random.randint(0, 40))
    positions = sorted(set(positions))[:12]
    if len(positions) == 12:
        initial_states2.append(make_state([(p, 1, 1) for p in positions]))

    # More random states
    for _ in range(20):
        positions = sorted(random.sample(range(41), 12))
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.choice([1, 1, 1, 2, 3]) for _ in range(12)]
        initial_states2.append(make_state([(positions[i], signs[i], mags[i]) for i in range(12)]))

    # Evaluate and find best
    best_state2 = None
    best_state2_score = float('inf')
    for state in initial_states2:
        if not valid_state(state):
            continue
        cnt = state_score(state)
        if cnt < best_state2_score:
            best_state2_score = cnt
            best_state2 = state
        coeffs = state_to_coeffs(state)
        if is_valid(coeffs):
            try_poly(coeffs)

    if best_state2 is None:
        best_state2 = initial_states2[0]
        best_state2_score = state_score(best_state2)

    current_state = best_state2
    current_score = best_state2_score

    # Simulated annealing
    T0 = 10.0
    T_end = 0.01
    total_steps = 80000
    best_sa_score = current_score
    best_sa_state = current_state

    for step in range(total_steps):
        if time.time() > deadline - 3:
            break

        progress = step / total_steps
        T = T0 * (T_end / T0) ** progress

        # Generate random move
        state_list = list(current_state)  # list of (pos, sign, mag)
        move_type = random.randint(1, 8)

        if move_type == 1:  # Move one position
            i = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            if new_pos not in [s[0] for s in state_list if s[0] != state_list[i][0]]:
                state_list[i] = (new_pos, state_list[i][1], state_list[i][2])

        elif move_type == 2:  # Swap two positions
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pj, si, mi)
            state_list[j] = (pi, sj, mj)

        elif move_type == 3:  # Flip one sign
            i = random.randint(0, 11)
            p, s, m = state_list[i]
            state_list[i] = (p, -s, m)

        elif move_type == 4:  # Flip two signs
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, -si, mi)
            state_list[j] = (pj, -sj, mj)

        elif move_type == 5:  # Change one magnitude
            i = random.randint(0, 11)
            p, s, m = state_list[i]
            new_m = random.randint(1, 10)
            state_list[i] = (p, s, new_m)

        elif move_type == 6:  # Change two magnitudes
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, si, random.randint(1, 10))
            state_list[j] = (pj, sj, random.randint(1, 10))

        elif move_type == 7:  # Swap two magnitudes
            i, j = random.sample(range(12), 2)
            pi, si, mi = state_list[i]
            pj, sj, mj = state_list[j]
            state_list[i] = (pi, si, mj)
            state_list[j] = (pj, sj, mi)

        elif move_type == 8:  # Move position + flip sign
            i = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            p, s, m = state_list[i]
            if new_pos not in [x[0] for x in state_list if x[0] != p]:
                state_list[i] = (new_pos, -s, m)

        new_state = make_state(state_list)
        if not valid_state(new_state):
            continue

        new_score = state_score(new_state)
        delta = new_score - current_score

        if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_state = new_state
            current_score = new_score
            if new_score < best_sa_score:
                best_sa_score = new_score
                best_sa_state = new_state
                coeffs = state_to_coeffs(new_state)
                if is_valid(coeffs):
                    try_poly(coeffs)

    # Final local search: greedy hill climbing from best state
    if time.time() < deadline - 2:
        current_state = best_sa_state
        current_score = best_sa_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                # Try moving position i
                for new_pos in range(41):
                    if new_pos == current_state[i][0]:
                        continue
                    if new_pos in [s[0] for s in current_state if s[0] != current_state[i][0]]:
                        continue
                    state_list = list(current_state)
                    state_list[i] = (new_pos, state_list[i][1], state_list[i][2])
                    new_state = make_state(state_list)
                    if not valid_state(new_state):
                        continue
                    new_score = state_score(new_state)
                    if new_score < current_score:
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        coeffs = state_to_coeffs(new_state)
                        if is_valid(coeffs):
                            try_poly(coeffs)
                        break
                if time.time() > deadline - 2:
                    break
                # Try flipping sign of i
                state_list = list(current_state)
                p, s, m = state_list[i]
                state_list[i] = (p, -s, m)
                new_state = make_state(state_list)
                new_score = state_score(new_state)
                if new_score < current_score:
                    current_state = new_state
                    current_score = new_score
                    improved = True
                    coeffs = state_to_coeffs(new_state)
                    if is_valid(coeffs):
                        try_poly(coeffs)
                # Try changing magnitude of i
                for new_m in range(1, 11):
                    if new_m == m:
                        continue
                    state_list = list(current_state)
                    state_list[i] = (p, s, new_m)
                    new_state = make_state(state_list)
                    new_score = state_score(new_state)
                    if new_score < current_score:
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        coeffs = state_to_coeffs(new_state)
                        if is_valid(coeffs):
                            try_poly(coeffs)
                        break

    # Ensure we have a valid result
    if not is_valid(best_poly):
        best_poly = [1] * 12

    return best_poly
