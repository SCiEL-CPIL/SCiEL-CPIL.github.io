import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    N = 1000
    max_rep = 2
    MAX_SUM = 2 * N  # 2000

    # Pre-allocated arrays for fast count tracking
    # counts[s] = number of representations of sum s
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    in_set = np.zeros(N + 1, dtype=bool)

    def is_valid_addition(x, counts, in_set):
        """Check if adding x keeps all sums <= max_rep."""
        # Check 2*x
        if counts[2 * x] >= max_rep:
            return False
        # Check x + y for all y in set
        # We need to iterate over set members
        # Use a list for fast iteration
        return True  # placeholder, actual check done in caller with S list

    def update_counts_add(x, S_list, counts):
        """Add x to counts: increment counts for x+y (y in S) and 2x."""
        for y in S_list:
            counts[x + y] += 1
        counts[2 * x] += 1

    def update_counts_remove(x, S_list, counts):
        """Remove x from counts: decrement counts for x+y (y in S) and 2x."""
        for y in S_list:
            counts[x + y] -= 1
        counts[2 * x] -= 1

    def greedy_build(order, counts, in_set):
        """Greedy: add elements from order if valid."""
        S_list = []
        S_set = set()
        for x in order:
            if in_set[x]:
                continue
            # Check validity
            valid = True
            if counts[2 * x] >= max_rep:
                valid = False
            else:
                for y in S_list:
                    if counts[x + y] >= max_rep:
                        valid = False
                        break
            if valid:
                S_list.append(x)
                S_set.add(x)
                in_set[x] = True
                # Update counts
                for y in S_list[:-1]:  # all previous elements
                    counts[x + y] += 1
                counts[2 * x] += 1
        return S_list, S_set, counts, in_set

    def local_search(S_list, S_set, counts, in_set, time_budget):
        """Local search: try to add elements, then remove-one-add-two."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # Phase 1: Try to add any candidate
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            if not candidates:
                break
            random.shuffle(candidates)
            for x in candidates[:30]:
                valid = True
                if counts[2 * x] >= max_rep:
                    valid = False
                else:
                    for y in S_list:
                        if counts[x + y] >= max_rep:
                            valid = False
                            break
                if valid:
                    S_list.append(x)
                    S_set.add(x)
                    in_set[x] = True
                    for y in S_list[:-1]:
                        counts[x + y] += 1
                    counts[2 * x] += 1
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Remove-one-add-two
            # Try removing u from S, then adding two elements v1, v2
            # This can grow the set by 1 net
            if len(S_list) < 2:
                break

            # Try a few random removals
            for _ in range(5):
                if time.time() >= time_budget:
                    break
                u_idx = random.randint(0, len(S_list) - 1)
                u = S_list[u_idx]

                # Temporarily remove u
                old_counts = counts.copy()
                in_set[u] = False
                S_list.pop(u_idx)
                S_set.discard(u)
                for y in S_list:
                    counts[u + y] -= 1
                counts[2 * u] -= 1

                # Now try to add two elements
                added = []
                candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
                random.shuffle(candidates)

                # Try to add first element
                for v1 in candidates[:40]:
                    valid = True
                    if counts[2 * v1] >= max_rep:
                        valid = False
                    else:
                        for y in S_list:
                            if counts[v1 + y] >= max_rep:
                                valid = False
                                break
                    if valid:
                        S_list.append(v1)
                        S_set.add(v1)
                        in_set[v1] = True
                        for y in S_list[:-1]:
                            counts[v1 + y] += 1
                        counts[2 * v1] += 1
                        added.append(v1)
                        break

                # Try to add second element
                if len(added) == 1:
                    candidates2 = [x for x in range(1, N + 1) if not in_set[x] and x != u and x != added[0]]
                    random.shuffle(candidates2)
                    for v2 in candidates2[:40]:
                        valid = True
                        if counts[2 * v2] >= max_rep:
                            valid = False
                        else:
                            for y in S_list:
                                if counts[v2 + y] >= max_rep:
                                    valid = False
                                    break
                        if valid:
                            S_list.append(v2)
                            S_set.add(v2)
                            in_set[v2] = True
                            for y in S_list[:-1]:
                                counts[v2 + y] += 1
                            counts[2 * v2] += 1
                            added.append(v2)
                            break

                # If we added 2 elements, net gain is +1 (removed 1, added 2)
                if len(added) == 2:
                    improved = True
                    break
                else:
                    # Rollback: restore u and undo additions
                    # First undo additions
                    for v in added:
                        in_set[v] = False
                        S_set.discard(v)
                        # Find and remove from S_list
                        if v in S_list:
                            S_list.remove(v)
                        for y in S_list:
                            counts[v + y] -= 1
                        counts[2 * v] -= 1

                    # Restore u
                    S_list.append(u)
                    S_set.add(u)
                    in_set[u] = True
                    for y in S_list[:-1]:
                        counts[u + y] += 1
                    counts[2 * u] += 1

                    # Restore original counts (in case partial updates caused issues)
                    # Actually the above should be correct if done carefully
                    # But to be safe, let's just use old_counts
                    counts[:] = old_counts
                    in_set[u] = True
                    S_list.append(u) if u not in S_list else None
                    S_set.add(u)

                    # Re-add u properly
                    if u not in S_list:
                        S_list.append(u)
                    for y in S_list:
                        if y != u:
                            counts[u + y] += 1
                    counts[2 * u] += 1

                    break  # Skip remaining removals this iteration

        return S_list, S_set, counts, in_set

    best_size = 0
    best_set = []

    num_restarts = 80  # Increased due to faster arrays

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Reset state
        counts[:] = 0
        in_set[:] = False

        if r == 0:
            order = list(range(1, N + 1))
        elif r == 1:
            order = list(range(N, 0, -1))
        elif r == 2:
            # Even then odd
            order = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        elif r == 3:
            # Odd then even
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        S_list, S_set, counts, in_set = greedy_build(order, counts, in_set)

        # Local search
        S_list, S_set, counts, in_set = local_search(
            S_list, S_set, counts, in_set, deadline
        )

        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)

    return best_set
