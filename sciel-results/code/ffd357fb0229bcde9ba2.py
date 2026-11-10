"""Combined program: MILP initialization + conflict-graph local search polish."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 256
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # 1. Precompute all 3-term APs and conflict structure
    ap_list = []
    for a in range(1, n + 1):
        max_b = (n + a) // 2
        for b in range(a + 1, max_b + 1):
            c = 2 * b - a
            if c <= n and c > b:
                ap_list.append((a, b, c))

    m = len(ap_list)

    # conflict_pairs[x] = list of (y, z) such that {x,y,z} is a 3-AP
    conflict_pairs = [[] for _ in range(n + 1)]
    for a, b, c in ap_list:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # affects[v] = list of (x, other) such that (v, other) is in conflict_pairs[x]
    # i.e., if v and other are both in S, then x is blocked
    affects = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y, z in conflict_pairs[x]:
            affects[y].append((x, z))
            affects[z].append((x, y))

    # State management functions
    def make_state(in_S):
        """Compute block_count from in_S."""
        block_count = [0] * (n + 1)
        for x in range(1, n + 1):
            for y, z in conflict_pairs[x]:
                if in_S[y] and in_S[z]:
                    block_count[x] += 1
        return block_count

    def add_to_state(x, in_S, block_count):
        in_S[x] = True
        for other_x, other_y in affects[x]:
            if in_S[other_y]:
                block_count[other_x] += 1

    def remove_from_state(x, in_S, block_count):
        for other_x, other_y in affects[x]:
            if in_S[other_y]:
                block_count[other_x] -= 1
        in_S[x] = False

    def can_add(x, in_S, block_count):
        return not in_S[x] and block_count[x] == 0

    def compute_sum(in_S):
        total = 0.0
        for x in range(1, n + 1):
            if in_S[x]:
                total += recip[x]
        return total

    def greedy_fill(forbidden_mask, in_S, block_count, descending=False):
        order = range(n, 0, -1) if descending else range(1, n + 1)
        for x in order:
            if forbidden_mask & (1 << x):
                continue
            if can_add(x, in_S, block_count):
                add_to_state(x, in_S, block_count)

    # 2. Try MILP first for a strong initial solution
    best_result = None
    best_sum = 0.0

    # Build MILP
    c_obj = np.array([1.0 / i for i in range(1, n + 1)])
    A_ub = np.zeros((m, n))
    for idx, (a, b, c) in enumerate(ap_list):
        A_ub[idx, a - 1] = 1
        A_ub[idx, b - 1] = 1
        A_ub[idx, c - 1] = 1
    b_ub = np.full(m, 2)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)

    elapsed = time.time() - start_time
    milp_time_limit = max(5.0, DEADLINE - elapsed - 10.0)  # Leave time for local search

    try:
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': milp_time_limit}
        )
        if res.status in (0, 1) and res.x is not None:
            x_sol = np.array(res.x)
            x_sol[x_sol < 0.5] = 0
            x_sol[x_sol >= 0.5] = 1
            in_S = [False] * (n + 1)
            for i in range(n):
                if x_sol[i] == 1:
                    in_S[i + 1] = True
            if any(in_S[1:]):
                block_count = make_state(in_S)
                # Verify no AP violations (safety check)
                valid = True
                for a, b, c in ap_list:
                    if in_S[a] and in_S[b] and in_S[c]:
                        valid = False
                        break
                if valid:
                    s = compute_sum(in_S)
                    if s > best_sum:
                        best_sum = s
                        best_result = [i for i in range(1, n + 1) if in_S[i]]
    except Exception:
        pass

    # 3. Fallback greedy if MILP failed
    if not best_result:
        in_S = [False] * (n + 1)
        block_count = [0] * (n + 1)
        greedy_fill(0, in_S, block_count)
        best_sum = compute_sum(in_S)
        best_result = [i for i in range(1, n + 1) if in_S[i]]
    else:
        # Reconstruct state from best_result for local search
        in_S = [False] * (n + 1)
        for x in best_result:
            in_S[x] = True
        block_count = make_state(in_S)

    # 4. Local search polish
    if time.time() - start_time < DEADLINE - 3.0:
        improved = True
        max_ls_time = DEADLINE - 3.0
        while improved and (time.time() - start_time) < max_ls_time:
            improved = False

            # Phase 1: Add any addable element
            for x in range(1, n + 1):
                if can_add(x, in_S, block_count):
                    add_to_state(x, in_S, block_count)
                    best_sum += recip[x]
                    improved = True
                    break
            if improved:
                continue

            # Phase 2: 1-swap local search
            best_gain = 0
            best_x = -1
            best_y = -1

            for y in range(1, n + 1):
                if in_S[y]:
                    continue
                # Find best x to remove: largest x (smallest recip) that allows adding y
                for x in range(n, 0, -1):
                    if not in_S[x]:
                        continue
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        break
                    # Check if removing x allows adding y
                    remove_from_state(x, in_S, block_count)
                    valid = block_count[y] == 0
                    add_to_state(x, in_S, block_count)
                    if valid:
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
                        break  # Best x for this y is the largest removable one

            if best_y != -1:
                remove_from_state(best_x, in_S, block_count)
                add_to_state(best_y, in_S, block_count)
                best_sum += best_gain
                improved = True
                continue

            # Phase 3: 1-remove + re-greedy
            best_remove_gain = -1
            best_remove_x = -1
            best_in_S = None
            best_block = None

            for x in range(1, n + 1):
                if time.time() - start_time > max_ls_time:
                    break
                if not in_S[x]:
                    continue

                # Save state
                in_S_copy = in_S[:]
                block_copy = block_count[:]

                remove_from_state(x, in_S, block_count)
                greedy_fill(0, in_S, block_count)
                new_sum = compute_sum(in_S)
                gain = new_sum - best_sum

                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_x = x
                    best_in_S = in_S[:]
                    best_block = block_count[:]

                # Restore
                in_S[:] = in_S_copy
                block_count[:] = block_copy

            if best_remove_x != -1:
                in_S[:] = best_in_S
                block_count[:] = best_block
                best_sum += best_remove_gain
                improved = True
                continue

            # Phase 4: Limited 2-swap
            if time.time() - start_time < max_ls_time - 2.0:
                S_list = [x for x in range(1, n + 1) if in_S[x]]
                not_S_list = [x for x in range(1, n + 1) if not in_S[x]]

                if len(S_list) >= 2 and len(not_S_list) >= 2:
                    # Try a few random 2-swaps
                    num_attempts = 50
                    for _ in range(num_attempts):
                        if time.time() - start_time > max_ls_time:
                            break
                        x1, x2 = random.sample(S_list, 2)
                        y1, y2 = random.sample(not_S_list, 2)

                        gain = (recip[y1] + recip[y2]) - (recip[x1] + recip[x2])
                        if gain <= 1e-15:
                            continue

                        # Save state
                        in_S_copy = in_S[:]
                        block_copy = block_count[:]

                        remove_from_state(x1, in_S, block_count)
                        remove_from_state(x2, in_S, block_count)

                        valid = True
                        if can_add(y1, in_S, block_count):
                            add_to_state(y1, in_S, block_count)
                            if can_add(y2, in_S, block_count):
                                add_to_state(y2, in_S, block_count)
                            else:
                                valid = False
                        else:
                            valid = False

                        if valid:
                            new_sum = compute_sum(in_S)
                            if new_sum > best_sum + 1e-15:
                                best_sum = new_sum
                                improved = True
                                break
                        else:
                            # Restore
                            in_S[:] = in_S_copy
                            block_count[:] = block_copy

            # Phase 5: Descending greedy restart (try different order)
            if not improved and time.time() - start_time < max_ls_time - 1.0:
                in_S_copy = in_S[:]
                block_copy = block_count[:]

                in_S[:] = [False] * (n + 1)
                block_count[:] = [0] * (n + 1)
                greedy_fill(0, in_S, block_count, descending=True)
                new_sum = compute_sum(in_S)

                if new_sum > best_sum + 1e-15:
                    best_sum = new_sum
                    improved = True
                else:
                    # Restore
                    in_S[:] = in_S_copy
                    block_count[:] = block_copy

    # 5. Final validity check and return
    in_S_final = [False] * (n + 1)
    for x in best_result:
        in_S_final[x] = True

    # Re-verify
    valid = True
    for a, b, c in ap_list:
        if in_S_final[a] and in_S_final[b] and in_S_final[c]:
            valid = False
            break

    if not valid:
        # Fallback to a known-valid greedy set
        in_S_fallback = [False] * (n + 1)
        block_count_fallback = [0] * (n + 1)
        greedy_fill(0, in_S_fallback, block_count_fallback)
        best_result = [i for i in range(1, n + 1) if in_S_fallback[i]]
    else:
        # Rebuild from current in_S state if it's better
        current_set = [i for i in range(1, n + 1) if in_S[i]]
        current_sum = compute_sum(in_S)
        if current_sum > best_sum + 1e-15:
            best_result = current_set

    if not best_result:
        # Absolute fallback
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
