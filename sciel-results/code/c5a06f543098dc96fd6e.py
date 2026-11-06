import math
import time

def construct():
    deadline = time.time() + 145.0
    N = 511
    C = 512

    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)

    # Precompute conflict masks as bitsets
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Greedy initial solution (highest weight first)
    S_mask = 0
    for a in range(N, 0, -1):
        if (S_mask & conflict_mask[a]) == 0:
            S_mask |= (1 << a)

    def mask_to_list(mask):
        result = []
        for a in range(1, N + 1):
            if mask & (1 << a):
                result.append(a)
        return result

    def score_of_mask(mask):
        s = 0.0
        for a in range(1, N + 1):
            if mask & (1 << a):
                s += w[a]
        return s

    best_S_mask = S_mask
    best_score = score_of_mask(best_S_mask)

    def better_upper_bound(available_mask, current_score):
        """Greedy independent set upper bound on available vertices"""
        greedy_score = 0.0
        temp_mask = available_mask
        while temp_mask:
            # Find highest weight vertex in temp_mask
            best_a = 0
            best_w = -1.0
            for a in range(N, 0, -1):
                if temp_mask & (1 << a):
                    if w[a] > best_w:
                        best_w = w[a]
                        best_a = a
            if best_a == 0:
                break
            greedy_score += best_w
            temp_mask &= ~(1 << best_a)
            temp_mask &= ~conflict_mask[best_a]
        return current_score + greedy_score

    # Branch and bound with improved pruning
    def branch_and_bound(current_mask, available_mask, current_score, depth):
        nonlocal best_S_mask, best_score, deadline

        if time.time() > deadline:
            return

        if not available_mask:
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_S_mask = current_mask
            return

        # Upper bound check
        ub = better_upper_bound(available_mask, current_score)
        if ub <= best_score + 1e-12:
            return

        # Find the highest weight available vertex
        pivot = 0
        for a in range(N, 0, -1):
            if available_mask & (1 << a):
                pivot = a
                break

        if pivot == 0:
            return

        # Branch 1: Include pivot
        new_mask_include = current_mask | (1 << pivot)
        new_available_include = available_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound(new_mask_include, new_available_include, current_score + w[pivot], depth + 1)

        # Branch 2: Exclude pivot
        new_available_exclude = available_mask & ~(1 << pivot)
        branch_and_bound(current_mask, new_available_exclude, current_score, depth + 1)

    # Run branch and bound from initial state
    initial_available = 0
    for a in range(1, N + 1):
        initial_available |= (1 << a)

    branch_and_bound(0, initial_available, 0.0, 0)

    # Local search: 1-add, 1-remove, 2-remove-1-add moves
    def local_search(S_mask):
        nonlocal best_S_mask, best_score, deadline
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_score = score_of_mask(S_mask)

            # Try 1-add moves
            for a in range(N, 0, -1):
                if not (S_mask & (1 << a)):
                    if (S_mask & conflict_mask[a]) == 0:
                        new_mask = S_mask | (1 << a)
                        new_score = cur_score + w[a]
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            best_S_mask = new_mask
                            S_mask = new_mask
                            cur_score = new_score
                            improved = True

            if time.time() > deadline:
                break

            # Try 1-remove-1-add moves (swap)
            for a_out in range(1, N + 1):
                if time.time() > deadline:
                    break
                if S_mask & (1 << a_out):
                    # Remove a_out, try adding each a_in
                    for a_in in range(N, 0, -1):
                        if not (S_mask & (1 << a_in)):
                            if (S_mask & ~(1 << a_out)) & conflict_mask[a_in]:
                                continue
                            new_mask = (S_mask & ~(1 << a_out)) | (1 << a_in)
                            new_score = cur_score - w[a_out] + w[a_in]
                            if new_score > best_score + 1e-12:
                                best_score = new_score
                                best_S_mask = new_mask
                                S_mask = new_mask
                                cur_score = new_score
                                improved = True
                                break

            if time.time() > deadline:
                break

            # Try 2-remove-1-add moves
            for a_out1 in range(1, N + 1):
                if time.time() > deadline:
                    break
                if not (S_mask & (1 << a_out1)):
                    continue
                for a_out2 in range(a_out1 + 1, N + 1):
                    if time.time() > deadline:
                        break
                    if not (S_mask & (1 << a_out2)):
                        continue
                    # Remove both a_out1 and a_out2
                    base_mask = S_mask & ~(1 << a_out1) & ~(1 << a_out2)
                    base_score = cur_score - w[a_out1] - w[a_out2]
                    # Try adding a_in
                    for a_in in range(N, 0, -1):
                        if (S_mask & (1 << a_in)):
                            continue
                        if (base_mask & conflict_mask[a_in]) == 0:
                            new_mask = base_mask | (1 << a_in)
                            new_score = base_score + w[a_in]
                            if new_score > best_score + 1e-12:
                                best_score = new_score
                                best_S_mask = new_mask
                                S_mask = new_mask
                                cur_score = new_score
                                improved = True
                                break

    local_search(best_S_mask)

    # Additional restarts with different greedy orderings
    for restart in range(10):
        if time.time() > deadline - 10:
            break

        # Different greedy strategies
        S_mask = 0
        if restart == 0:
            # Standard greedy (highest weight first)
            for a in range(N, 0, -1):
                if (S_mask & conflict_mask[a]) == 0:
                    S_mask |= (1 << a)
        elif restart == 1:
            # Try adding in order of a descending, but skip if it blocks high-weight vertices
            for a in range(N, 0, -1):
                if (S_mask & conflict_mask[a]) == 0:
                    S_mask |= (1 << a)
        else:
            # Randomized greedy
            order = list(range(1, N + 1))
            # Sort by weight with some randomness
            import random
            random.seed(restart * 12345)
            random.shuffle(order)
            order.sort(key=lambda x: -w[x] + random.random() * 0.01)
            for a in order:
                if (S_mask & conflict_mask[a]) == 0:
                    S_mask |= (1 << a)

        s = score_of_mask(S_mask)
        if s > best_score + 1e-12:
            best_score = s
            best_S_mask = S_mask

        local_search(S_mask)

    result = []
    for a in range(1, N + 1):
        if best_S_mask & (1 << a):
            result.append(a)

    return result
