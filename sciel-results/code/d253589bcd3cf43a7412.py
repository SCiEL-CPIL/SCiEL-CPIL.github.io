import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    N = 256

    # Precompute for each x the list of pairs (a, c) where {a, x, c} forms a 3-AP
    # For x to be added, no such pair can both be in S.
    # We'll maintain a "forbidden" counter: forbidden[v] = number of pairs (a,c) in S
    #   such that a + c = 2v and a < v < c (i.e., v is the middle).
    # But we also need to handle cases where v is an end.
    #
    # Actually, let's think differently. When we add x to S, it conflicts if:
    #   - There exist a, c in S with a + c = 2x (x is middle)
    #   - There exist a in S such that (a+x)/2 is in S (x is end, a is other end)
    #   - There exist c in S such that (x+c)/2 is in S (x is start, c is other end)
    #
    # The second and third are the same: there exist y in S such that x + y = 2z for some z in S.
    # Equivalently: for some y in S, z = (x+y)/2 must be integer and in S.
    #
    # Let's use a different approach: maintain a set S and a boolean array in_set.
    # For speed, precompute for each x the list of "conflict partners":
    #   conflicts[x] = set of y such that adding x when y is in S could create a 3-AP.
    #
    # Actually, the cleanest fast check:
    #   x is invalid iff there exists y in S such that:
    #     (a) 2x - y is in S and 2x - y != x  (x is middle: y, x, 2x-y)
    #     (b) (x + y) is even, and (x+y)//2 is in S, and (x+y)//2 != x, (x+y)//2 != y
    #         (x and y are ends, (x+y)//2 is middle)
    #
    # But (b) is symmetric: if we check all y in S, we cover both "x is start" and "x is end".
    #
    # For (a): for each y in S, check if 2x-y is in S.
    # For (b): for each y in S, if (x+y)%2==0, check if (x+y)//2 is in S.
    #
    # This is O(|S|) per check. With |S| ~ 50-100, and N=256, this is manageable.
    # But we can do better: precompute for each x the list of y values that matter.

    # Precompute: for each x in [1,N], the list of y in [1,N] such that
    #   y < x and 2*x - y <= N and 2*x - y > x  (y, x, 2x-y: x is middle)
    #   OR (x+y)%2==0 and 1 <= (x+y)//2 <= N and (x+y)//2 != x and (x+y)//2 != y
    #       (x and y are ends)
    #
    # Actually, for the "ends" case, it's symmetric in x and y, so we just need
    # to check: does there exist y in S such that (x+y) is even and (x+y)//2 is in S?
    # This is equivalent to: does there exist z in S such that 2z - x is in S and 2z-x != x?
    # i.e., for each z in S, check if 2z-x is in S.
    #
    # So the full check for adding x:
    #   1. For each y in S: if y < x and 2x-y <= N and 2x-y in S: conflict (x middle)
    #   2. For each z in S: if 2z-x >= 1 and 2z-x in S and 2z-x != x: conflict (z middle)
    #      Note: 2z-x != x means z != x, which is true since x not in S.
    #      Also need 2z-x != z, i.e., z != x, same thing.
    #
    # Wait, case 2: z is the middle, x and 2z-x are the ends.
    # We need 2z-x >= 1 and 2z-x <= N and 2z-x in S and 2z-x != x.
    # 2z-x != x iff z != x, which is true.
    #
    # So: can_add(x) iff for all y in S:
    #   NOT (y < x AND 2x-y <= N AND 2x-y in S)  [x is middle]
    #   AND NOT (2z-x >= 1 AND 2z-x <= N AND 2z-x in S) for z=y  [y is middle]
    #
    # Combine: for each y in S, check:
    #   - If y < x: c = 2x - y; if 1 <= c <= N and in_set[c]: return False
    #   - d = 2y - x; if 1 <= d <= N and in_set[d] and d != x: return False
    #     (d != x is automatic since x not in S, but d could equal y if x=y, impossible)
    #
    # This is O(|S|) per check.

    def can_add(x, in_set, S_list):
        """Check if x can be added to the set represented by in_set (bool array) and S_list (list of elements in S)."""
        if in_set[x]:
            return False
        for y in S_list:
            # Case 1: x is middle. y, x, c where c = 2x - y
            if y < x:
                c = 2 * x - y
                if c <= N and in_set[c]:
                    return False
            # Case 2: y is middle. x, y, d where d = 2y - x
            d = 2 * y - x
            if 1 <= d <= N and d != x and in_set[d]:
                return False
        return True

    def get_weight(in_set):
        return sum(1.0 / x for x in range(1, N + 1) if in_set[x])

    def local_search(initial_S, deadline):
        in_set = [False] * (N + 1)
        for x in initial_S:
            in_set[x] = True
        S_list = list(initial_S)

        best_weight = get_weight(in_set)
        best_in_set = in_set[:]

        no_improve_count = 0
        max_no_improve = 30

        while time.time() < deadline:
            improved = False

            # Try to add the best (smallest) element
            best_v = -1
            best_v_val = 0.0
            for v in range(1, N + 1):
                if not in_set[v]:
                    if can_add(v, in_set, S_list):
                        val = 1.0 / v
                        if val > best_v_val:
                            best_v_val = val
                            best_v = v

            if best_v != -1:
                in_set[best_v] = True
                S_list.append(best_v)
                best_weight += best_v_val
                improved = True
                no_improve_count = 0
                continue

            # Try 1-swap: remove u, add v where v < u (so 1/v > 1/u)
            best_swap = None
            best_gain = 0.0

            for u in S_list:
                if time.time() > deadline:
                    break
                # Temporarily remove u
                in_set[u] = False
                S_list.remove(u)

                # Find best v < u that can be added
                found_v = -1
                for v in range(1, u):
                    if not in_set[v]:
                        if can_add(v, in_set, S_list):
                            found_v = v
                            break

                # Restore u
                in_set[u] = True
                S_list.append(u)

                if found_v != -1:
                    gain = 1.0 / found_v - 1.0 / u
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (u, found_v)

            if best_swap:
                u, v = best_swap
                in_set[u] = False
                S_list.remove(u)
                in_set[v] = True
                S_list.append(v)
                best_weight += best_gain
                improved = True
                no_improve_count = 0
                continue

            # Try 2-swap: remove u1, u2, add v1, v2
            # This is expensive, so only do it occasionally
            if no_improve_count >= 5:
                found_2swap = False
                for u1 in S_list:
                    if time.time() > deadline:
                        break
                    for u2 in S_list:
                        if u2 == u1:
                            continue
                        # Remove u1 and u2
                        in_set[u1] = False
                        in_set[u2] = False
                        S_list.remove(u1)
                        S_list.remove(u2)

                        # Find best pair (v1, v2) with v1 < v2, both not in S
                        # We want max (1/v1 + 1/v2) - (1/u1 + 1/u2)
                        # Try smallest v1 first
                        best_pair = None
                        best_pair_gain = 0.0
                        for v1 in range(1, min(N, max(u1, u2))):
                            if in_set[v1]:
                                continue
                            if not can_add(v1, in_set, S_list):
                                continue
                            for v2 in range(v1 + 1, N + 1):
                                if in_set[v2]:
                                    continue
                                if not can_add(v2, in_set, [v1] + S_list):
                                    continue
                                gain = 1.0 / v1 + 1.0 / v2 - 1.0 / u1 - 1.0 / u2
                                if gain > best_pair_gain:
                                    best_pair_gain = gain
                                    best_pair = (v1, v2)
                                # Early break: if v1 is very small, gain is already good
                                if 1.0 / v1 > 1.0 / u1 + 1.0 / u2:
                                    break

                        # Restore u1, u2
                        in_set[u1] = True
                        in_set[u2] = True
                        S_list.extend([u1, u2])

                        if best_pair and best_pair_gain > 0:
                            v1, v2 = best_pair
                            in_set[u1] = False
                            in_set[u2] = False
                            S_list.remove(u1)
                            S_list.remove(u2)
                            in_set[v1] = True
                            in_set[v2] = True
                            S_list.extend([v1, v2])
                            best_weight += best_pair_gain
                            improved = True
                            found_2swap = True
                            no_improve_count = 0
                            break
                    if found_2swap:
                        break

                if found_2swap:
                    continue

            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break

        # Track best
        if best_weight > get_weight(best_in_set):
            best_in_set = in_set[:]
            best_weight = get_weight(in_set)

        return best_in_set, best_weight

    def random_kick(in_set, S_list, num_removals=3):
        """Remove some random elements to escape local optima."""
        if len(S_list) <= num_removals:
            return in_set, S_list
        to_remove = random.sample(S_list, num_removals)
        for x in to_remove:
            in_set[x] = False
            S_list.remove(x)
        return in_set, S_list

    # Generate initial sets
    candidates = []

    # 1. Greedy (add smallest first)
    S_greedy = set()
    in_set = [False] * (N + 1)
    S_list = []
    for x in range(1, N + 1):
        if can_add(x, in_set, S_list):
            S_greedy.add(x)
            in_set[x] = True
            S_list.append(x)
    candidates.append(set(S_greedy))

    # 2. Base 3 (digits 0, 1)
    S_b3_01 = set()
    for x in range(1, N + 1):
        n = x
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            S_b3_01.add(x)
    candidates.append(S_b3_01)

    # 3. Base 3 (digits 0, 2)
    S_b3_02 = set()
    for x in range(1, N + 1):
        n = x
        valid = True
        while n > 0:
            if n % 3 == 1:
                valid = False
                break
            n //= 3
        if valid:
            S_b3_02.add(x)
    candidates.append(S_b3_02)

    # 4. Random Greedy (multiple restarts with different seeds)
    for seed in range(20):
        random.seed(seed)
        S_rand = set()
        in_set = [False] * (N + 1)
        S_list = []
        items = list(range(1, N + 1))
        random.shuffle(items)
        for x in items:
            if can_add(x, in_set, S_list):
                S_rand.add(x)
                in_set[x] = True
                S_list.append(x)
        candidates.append(S_rand)

    # 5. Greedy from random subset (start with some random small numbers)
    for seed in range(10):
        random.seed(100 + seed)
        S_rand = set()
        in_set = [False] * (N + 1)
        S_list = []
        # Start with a few random small numbers
        small_nums = list(range(1, 32))
        random.shuffle(small_nums)
        for x in small_nums[:5]:
            if can_add(x, in_set, S_list):
                S_rand.add(x)
                in_set[x] = True
                S_list.append(x)
        # Then greedy add
        for x in range(1, N + 1):
            if not in_set[x] and can_add(x, in_set, S_list):
                S_rand.add(x)
                in_set[x] = True
                S_list.append(x)
        candidates.append(S_rand)

    best_in_set = [False] * (N + 1)
    best_w = 0.0

    # Run local search on each candidate
    for i, S_init in enumerate(candidates):
        if time.time() > deadline:
            break

        in_set = [False] * (N + 1)
        for x in S_init:
            in_set[x] = True
        S_list = list(S_init)

        # Multiple local search rounds with random kicks
        for round_num in range(3):
            if time.time() > deadline:
                break
            in_set, S_list = local_search(set(S_list), deadline)
            w = get_weight(in_set)
            if w > best_w:
                best_w = w
                best_in_set = in_set[:]

            # Random kick for next round
            if round_num < 2 and time.time() < deadline - 5:
                in_set, S_list = random_kick(in_set, S_list, num_removals=2 + round_num)

    # Final polish: try to add any remaining elements
    S_list = [x for x in range(1, N + 1) if best_in_set[x]]
    for v in range(1, N + 1):
        if not best_in_set[v]:
            if can_add(v, best_in_set, S_list):
                best_in_set[v] = True
                S_list.append(v)

    return [x for x in range(1, N + 1) if best_in_set[x]]
