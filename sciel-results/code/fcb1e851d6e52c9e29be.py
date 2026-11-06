import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 256

    # Precompute canonical 3-APs and 4-APs (0-indexed internally)
    canonical_aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            if c < n:
                canonical_aps3.append((a, b, c))

    canonical_aps4 = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                canonical_aps4.append((a, b, c, e))

    # Precompute partners for fast delta computation
    partners3 = [[] for _ in range(n)]
    for a, b, c in canonical_aps3:
        partners3[a].append((b, c))
        partners3[b].append((a, c))
        partners3[c].append((a, b))

    partners4 = [[] for _ in range(n)]
    for a, b, c, d in canonical_aps4:
        partners4[a].append((b, c, d))
        partners4[b].append((a, c, d))
        partners4[c].append((a, b, d))
        partners4[d].append((a, b, c))

    # Degree scores for greedy ordering
    deg3 = [len(partners3[v]) for v in range(n)]
    deg4 = [len(partners4[v]) for v in range(n)]

    def count_3aps_mask(A):
        count = 0
        for a, b, c in canonical_aps3:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1:
                count += 1
        return count

    def count_4aps_mask(A):
        count = 0
        for a, b, c, d in canonical_aps4:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1 and (A >> d) & 1:
                count += 1
        return count

    def gain_add(A, v):
        """Number of new 3-APs created by adding v to A."""
        gain = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                gain += 1
        return gain

    def loss_remove(A, v):
        """Number of 3-APs destroyed by removing v from A."""
        loss = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                loss += 1
        return loss

    def can_add(A, v):
        """Check if adding v to A creates no 4-AP."""
        for u, w, x in partners4[v]:
            if ((A >> u) & 1) and ((A >> w) & 1) and ((A >> x) & 1):
                return False
        return True

    def greedy_construct(order):
        """Greedy: add elements in given order if no 4-AP is created."""
        A = 0
        for v in order:
            if can_add(A, v):
                A |= (1 << v)
        return A

    def hill_climb(A, time_limit):
        """Iterative improvement: try all add/remove/swap moves, take best improving move."""
        current_score = count_3aps_mask(A)
        best_score = current_score
        best_A = A

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_move = None

            # Try all adds
            for v in range(n):
                if (A >> v) & 1:
                    continue
                if not can_add(A, v):
                    continue
                g = gain_add(A, v)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', v, -1)

            # Try all removes
            for v in range(n):
                if not ((A >> v) & 1):
                    continue
                l = loss_remove(A, v)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', v, -1)

            # Try all swaps
            for u in range(n):
                if not ((A >> u) & 1):
                    continue
                for v in range(n):
                    if (A >> v) & 1:
                        continue
                    if u == v:
                        continue

                    # Check if swap is valid (no 4-AP after removing u, adding v)
                    valid = True
                    for w, x, y in partners4[v]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and ((A >> y) & 1):
                            if w != u and x != u and y != u:
                                valid = False
                                break
                    if not valid:
                        continue

                    # Compute delta
                    loss_u = 0
                    for w, x in partners3[u]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and w != v and x != v:
                            loss_u += 1

                    gain_v = 0
                    for w, x in partners3[v]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and w != u and x != u:
                            gain_v += 1

                    delta = gain_v - loss_u
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', v, u)

            if best_move is not None and best_delta > 0:
                mtype, v, u = best_move
                if mtype == 'add':
                    A |= (1 << v)
                elif mtype == 'remove':
                    A &= ~(1 << v)
                else:  # swap
                    A &= ~(1 << u)
                    A |= (1 << v)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
                improved = True

        return best_A, best_score

    # Main search: multiple greedy constructions with different orderings, then hill-climb each
    best_A_global = 0
    best_score_global = 0

    # Generate different orderings
    orderings = []

    # Ordering 1: by decreasing 3-AP degree
    order1 = sorted(range(n), key=lambda v: (-deg3[v], v))
    orderings.append(order1)

    # Ordering 2: by decreasing (3-AP degree - 4-AP degree)
    order2 = sorted(range(n), key=lambda v: (-(deg3[v] - deg4[v]), v))
    orderings.append(order2)

    # Ordering 3: natural order
    orderings.append(list(range(n)))

    # Ordering 4: reverse natural
    orderings.append(list(range(n - 1, -1, -1)))

    # Ordering 5-8: random shuffles with fixed seeds
    for s in range(4):
        rng = random.Random(100 + s)
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)

    # Run greedy construction for each ordering, then hill-climb
    for i, order in enumerate(orderings):
        if time.time() > deadline:
            break

        A = greedy_construct(order)

        # Hill-climb with time budget
        remaining = deadline - time.time()
        time_per_hill = remaining / (len(orderings) - i)
        A, score = hill_climb(A, time.time() + time_per_hill)

        if score > best_score_global:
            best_score_global = score
            best_A_global = A

    # Additional perturbation restarts if time remains
    if time.time() < deadline - 5.0:
        rng = random.Random(999)
        for restart in range(5):
            if time.time() > deadline:
                break

            # Start from best solution and perturb
            A = best_A_global
            # Randomly remove a few elements
            for _ in range(3):
                if A == 0:
                    break
                bits = [v for v in range(n) if (A >> v) & 1]
                if not bits:
                    break
                v = rng.choice(bits)
                A &= ~(1 << v)

            # Re-add some elements greedily
            for v in range(n):
                if not ((A >> v) & 1) and can_add(A, v):
                    A |= (1 << v)

            # Hill-climb
            remaining = deadline - time.time()
            time_per_hill = remaining / (6 - restart)
            A, score = hill_climb(A, time.time() + time_per_hill)

            if score > best_score_global:
                best_score_global = score
                best_A_global = A

    # Final verification
    if count_4aps_mask(best_A_global) != 0:
        # Fallback: simple greedy
        best_A_global = greedy_construct(list(range(n)))

    # Convert to 1-indexed list
    result = [v + 1 for v in range(n) if (best_A_global >> v) & 1]

    return sorted(result)
