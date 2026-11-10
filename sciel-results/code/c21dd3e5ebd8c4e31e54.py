import time
import random
import numpy as np

N = 256
DEADLINE = 80.0  # Safety margin before 90s limit
RNG_SEED = 12345


def _build_4ap_clauses():
    """Return list of 4-element tuples (a,b,c,d) forming a 4-AP in [1,N]."""
    clauses = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                clauses.append((a, b, c, e))
    return clauses


def _build_3ap_clauses():
    """Return list of 3-element tuples (a,b,c) forming a 3-AP in [1,N]."""
    clauses = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = b + d
            if c <= N:
                clauses.append((a, b, c))
    return clauses


def _count_3ap(assign, three_aps):
    """Count satisfied 3-AP clauses given a boolean assignment (1-indexed array)."""
    count = 0
    for a, b, c in three_aps:
        if assign[a] and assign[b] and assign[c]:
            count += 1
    return count


def _count_4ap(assign, four_aps):
    """Count violated 4-AP clauses (should be 0 for valid solutions)."""
    count = 0
    for a, b, c, d in four_aps:
        if assign[a] and assign[b] and assign[c] and assign[d]:
            count += 1
    return count


def _walksat(assign, four_aps, three_aps, four_weights, three_weights,
             start_time, max_steps, temp, p_flip):
    """
    WalkSAT with dynamic clause weighting.
    - Hard clauses (4-APs): penalty if violated
    - Soft clauses (3-APs): reward if satisfied
    Score = (# satisfied 3-APs) - BIG * (# violated 4-APs)
    """
    BIG = 10000
    best_score = _count_3ap(assign, three_aps) - BIG * _count_4ap(assign, four_aps)
    best_assign = assign.copy()
    current_score = best_score

    # Pre-compute: for each variable, which 4-APs and 3-APs it appears in
    var_4aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_aps):
        for v in (a, b, c, d):
            var_4aps[v].append(idx)

    var_3aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps):
        for v in (a, b, c):
            var_3aps[v].append(idx)

    step = 0
    while step < max_steps:
        if time.time() - start_time > DEADLINE - 3:
            break
        step += 1

        # Pick a clause to focus on
        # With some probability pick a violated 4-AP (hard), else pick a random 3-AP
        violated_4 = [i for i in range(len(four_aps))
                      if all(assign[four_aps[i][j]] for j in range(4))]
        if violated_4 and (random.random() < 0.7 or not three_aps):
            # Pick a violated 4-AP, weighted by its weight
            if random.random() < 0.5:
                idx = random.choice(violated_4)
            else:
                wts = [four_weights[i] for i in violated_4]
                total = sum(wts)
                r = random.random() * total
                acc = 0
                idx = violated_4[0]
                for i, w in zip(violated_4, wts):
                    acc += w
                    if r <= acc:
                        idx = i
                        break
            clause_vars = four_aps[idx]
            is_hard = True
        else:
            idx = random.randrange(len(three_aps))
            clause_vars = three_aps[idx]
            is_hard = False

        # Pick a variable in the clause to flip
        # Weight by how much flipping would improve the score
        candidates = []
        for v in clause_vars:
            # Estimate gain from flipping v
            gain = 0
            if assign[v]:
                # Flipping v from 1->0
                # Check 4-APs containing v: if v was making them violated, removing helps
                for ci in var_4aps[v]:
                    a4, b4, c4, d4 = four_aps[ci]
                    if all(assign[x] for x in (a4, b4, c4, d4) if x != v):
                        gain += BIG * four_weights[ci]
                # Check 3-APs containing v: if v was making them satisfied, removing hurts
                for ci in var_3aps[v]:
                    a3, b3, c3 = three_aps[ci]
                    if all(assign[x] for x in (a3, b3, c3) if x != v):
                        gain -= three_weights[ci]
            else:
                # Flipping v from 0->1
                # Check 4-APs: if adding v creates a violation
                for ci in var_4aps[v]:
                    a4, b4, c4, d4 = four_aps[ci]
                    if all(assign[x] for x in (a4, b4, c4, d4) if x != v):
                        gain -= BIG * four_weights[ci]
                # Check 3-APs: if adding v creates a satisfaction
                for ci in var_3aps[v]:
                    a3, b3, c3 = three_aps[ci]
                    if all(assign[x] for x in (a3, b3, c3) if x != v):
                        gain += three_weights[ci]
            candidates.append((gain, v))

        # Pick variable to flip: with prob p_flip pick best, else random
        if random.random() < p_flip:
            # Weighted random among candidates
            gains = [max(0, g) + 0.1 for g, _ in candidates]
            total = sum(gains)
            r = random.random() * total
            acc = 0
            flip_var = candidates[0][1]
            for (g, v), w in zip(candidates, gains):
                acc += w
                if r <= acc:
                    flip_var = v
                    break
        else:
            # Pick the best
            flip_var = max(candidates, key=lambda x: x[0])[1]

        # Perform the flip
        assign[flip_var] = 1 - assign[flip_var]

        # Update score
        new_score = _count_3ap(assign, three_aps) - BIG * _count_4ap(assign, four_aps)

        if new_score > current_score:
            current_score = new_score
            # Update weights: decrease weight of the clause we focused on
            if is_hard:
                four_weights[idx] = max(0.1, four_weights[idx] * 0.9)
            else:
                three_weights[idx] = max(0.1, three_weights[idx] * 0.95)
            if current_score > best_score:
                best_score = current_score
                best_assign = assign.copy()
        else:
            # Accept with probability based on temperature (simulated annealing flavor)
            delta = new_score - current_score
            if random.random() < np.exp(delta / max(temp, 0.01)):
                current_score = new_score
            else:
                assign[flip_var] = 1 - assign[flip_var]  # Revert
                # Increase weight of the clause
                if is_hard:
                    four_weights[idx] = min(10.0, four_weights[idx] * 1.1)
                else:
                    three_weights[idx] = min(5.0, three_weights[idx] * 1.05)

        # Decay temperature
        temp *= 0.9999

    return best_assign, best_score


def _greedy_seed(four_aps, three_aps, start_time, order):
    """Quick greedy to get a feasible starting assignment."""
    assign = np.zeros(N + 1, dtype=np.int8)
    for x in order:
        if time.time() - start_time > DEADLINE - 20:
            break
        # Check if adding x creates a 4-AP
        ok = True
        for (a, b, c, d) in four_aps:
            if x in (a, b, c, d):
                others = [v for v in (a, b, c, d) if v != x]
                if all(assign[v] for v in others):
                    ok = False
                    break
        if ok:
            assign[x] = 1
    return assign


def construct():
    start_time = time.time()
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    four_aps = _build_4ap_clauses()
    three_aps = _build_3ap_clauses()

    # Pre-compute variable-to-clause mappings for speed
    var_4aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_aps):
        for v in (a, b, c, d):
            var_4aps[v].append(idx)

    var_3aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps):
        for v in (a, b, c):
            var_3aps[v].append(idx)

    best_overall = np.zeros(N + 1, dtype=np.int8)
    best_score = 0

    # Multiple WalkSAT restarts with different initial assignments
    n_restarts = 12
    steps_per_restart = 8000

    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Generate initial assignment
        if restart == 0:
            # Natural order greedy
            init = _greedy_seed(four_aps, three_aps, start_time, range(1, N + 1))
        elif restart == 1:
            # Reverse order greedy
            init = _greedy_seed(four_aps, three_aps, start_time, range(N, 0, -1))
        elif restart == 2:
            # Only evens
            init = _greedy_seed(four_aps, three_aps, start_time, range(2, N + 1, 2))
        elif restart == 3:
            # Only odds
            init = _greedy_seed(four_aps, three_aps, start_time, range(1, N + 1, 2))
        else:
            # Random shuffle greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            init = _greedy_seed(four_aps, three_aps, start_time, order)

        # Ensure no 4-APs (fix by removing elements)
        changed = True
        while changed:
            changed = False
            for (a, b, c, d) in four_aps:
                if all(init[v] for v in (a, b, c, d)):
                    # Remove the one with least 3-AP contribution
                    min_contrib = N
                    min_var = a
                    for v in (a, b, c, d):
                        contrib = 0
                        for ci in var_3aps[v]:
                            a3, b3, c3 = three_aps[ci]
                            if all(init[x] for x in (a3, b3, c3) if x != v):
                                contrib += 1
                        if contrib < min_contrib:
                            min_contrib = contrib
                            min_var = v
                    init[min_var] = 0
                    changed = True
                    break

        # Run WalkSAT
        four_weights = np.ones(len(four_aps))
        three_weights = np.ones(len(three_aps))
        temp = 2.0
        p_flip = 0.5

        result, score = _walksat(init, four_aps, three_aps, four_weights,
                                  three_weights, start_time, steps_per_restart,
                                  temp, p_flip)

        if score > best_score:
            best_score = score
            best_overall = result.copy()

        # Occasionally do a second pass with different params
        if time.time() - start_time < DEADLINE - 10 and restart % 3 == 0:
            temp2 = 1.0
            p_flip2 = 0.3
            result2, score2 = _walksat(result, four_aps, three_aps,
                                        np.ones(len(four_aps)),
                                        np.ones(len(three_aps)),
                                        start_time, steps_per_restart // 2,
                                        temp2, p_flip2)
            if score2 > best_score:
                best_score = score2
                best_overall = result2.copy()

    # Final: ensure validity and extract set
    final_set = [i for i in range(1, N + 1) if best_overall[i]]

    # Safety check: verify no 4-APs
    for (a, b, c, d) in four_aps:
        if all(x in final_set for x in (a, b, c, d)):
            # Remove d (or any one)
            final_set.remove(d)

    # If somehow empty, add 1
    if not final_set:
        final_set = [1]

    return sorted(final_set)
