import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 256

    # ---------- Precomputation (0-indexed internally) ----------
    # All 3-APs as (a, b, c) with 0-indexed
    aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            if c < n:
                aps3.append((a, b, c))
    m3 = len(aps3)

    # All 4-APs as (a, b, c, d) with 0-indexed
    aps4 = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                aps4.append((a, b, c, e))
    m4 = len(aps4)

    # For each element x, list of partner pairs for 3-APs
    # partners3[x] = list of (u, w) such that {x, u, w} forms a 3-AP
    partners3 = [[] for _ in range(n)]
    for a, b, c in aps3:
        partners3[a].append((b, c))
        partners3[b].append((a, c))
        partners3[c].append((a, b))

    # For each element x, list of partner triples for 4-APs
    # partners4[x] = list of (u, v, w) such that {x, u, v, w} forms a 4-AP
    partners4 = [[] for _ in range(n)]
    for a, b, c, d in aps4:
        partners4[a].append((b, c, d))
        partners4[b].append((a, c, d))
        partners4[c].append((a, b, d))
        partners4[d].append((a, b, c))

    # Degree scores
    deg3 = [len(partners3[v]) for v in range(n)]
    deg4 = [len(partners4[v]) for v in range(n)]

    # ---------- Core operations on bitmask S (uint64, bit x set if x in S) ----------
    def count_3aps(S):
        cnt = 0
        for a, b, c in aps3:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def has_4ap(S):
        for a, b, c, d in aps4:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                return True
        return False

    def count_4aps(S):
        cnt = 0
        for a, b, c, d in aps4:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt

    def can_add(S, x):
        """Check if adding x to S creates a 4-AP."""
        for u, v, w in partners4[x]:
            if (S >> u) & 1 and (S >> v) & 1 and (S >> w) & 1:
                return False
        return True

    def gain_add(S, x):
        """Number of new 3-APs formed by adding x to S."""
        g = 0
        for u, w in partners3[x]:
            if (S >> u) & 1 and (S >> w) & 1:
                g += 1
        return g

    def loss_remove(S, x):
        """Number of 3-APs lost by removing x from S."""
        g = 0
        for u, w in partners3[x]:
            if (S >> u) & 1 and (S >> w) & 1:
                g += 1
        return g

    # ---------- Greedy construction ----------
    def greedy_construct(order, use_score=True):
        S = 0
        for x in order:
            if not can_add(S, x):
                continue
            if use_score:
                g = gain_add(S, x)
                if g > 0 or bin(S).count('1') < 15:
                    S |= (1 << x)
            else:
                S |= (1 << x)
        return S

    # ---------- Hill climbing with add/remove/swap moves ----------
    def hill_climb(S, current_score, time_limit):
        """Iterative improvement: try all add/remove/swap moves, take best improving move."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_move = None

            # Try all adds
            for x in range(n):
                if (S >> x) & 1:
                    continue
                if not can_add(S, x):
                    continue
                g = gain_add(S, x)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', x, -1)

            # Try all removes
            for x in range(n):
                if not ((S >> x) & 1):
                    continue
                l = loss_remove(S, x)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', x, -1)

            # Try all swaps
            for u in range(n):
                if not ((S >> u) & 1):
                    continue
                for v in range(n):
                    if (S >> v) & 1:
                        continue
                    if u == v:
                        continue

                    # Check if swap is valid (no 4-AP after removing u, adding v)
                    valid = True
                    for w, x, y in partners4[v]:
                        if ((S >> w) & 1) and ((S >> x) & 1) and ((S >> y) & 1):
                            if w != u and x != u and y != u:
                                valid = False
                                break
                    if not valid:
                        continue

                    # Compute delta
                    loss_u = 0
                    for w, x in partners3[u]:
                        if ((S >> w) & 1) and ((S >> x) & 1) and w != v and x != v:
                            loss_u += 1

                    gain_v = 0
                    for w, x in partners3[v]:
                        if ((S >> w) & 1) and ((S >> x) & 1) and w != u and x != u:
                            gain_v += 1

                    delta = gain_v - loss_u
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', v, u)

            if best_move is not None and best_delta > 0:
                mtype, v, u = best_move
                if mtype == 'add':
                    S |= (1 << v)
                elif mtype == 'remove':
                    S &= ~(1 << v)
                else:  # swap
                    S &= ~(1 << u)
                    S |= (1 << v)
                current_score += best_delta
                improved = True

        return S, current_score

    # ---------- Simulated Annealing ----------
    def simulated_annealing(S, current_score, time_limit, rng, T0=5.0, cooling=0.999):
        """SA with add/remove/swap moves."""
        T = T0
        best_S = S
        best_score = current_score
        no_improve = 0

        while time.time() < time_limit:
            # Choose a random move
            move_type = rng.random()
            new_S = S
            new_score = current_score
            delta = 0

            if move_type < 0.4:  # Add
                x = rng.randint(0, n - 1)
                if (S >> x) & 1:
                    continue
                if not can_add(S, x):
                    continue
                new_S = S | (1 << x)
                delta = gain_add(S, x)
                new_score = current_score + delta
            elif move_type < 0.7:  # Remove
                bits = [v for v in range(n) if (S >> v) & 1]
                if not bits:
                    continue
                x = rng.choice(bits)
                new_S = S & ~(1 << x)
                delta = -loss_remove(S, x)
                new_score = current_score + delta
            else:  # Swap
                bits_in = [v for v in range(n) if (S >> v) & 1]
                bits_out = [v for v in range(n) if not (S >> v) & 1]
                if not bits_in or not bits_out:
                    continue
                u = rng.choice(bits_in)
                v = rng.choice(bits_out)

                # Check validity
                valid = True
                for w, x, y in partners4[v]:
                    if ((S >> w) & 1) and ((S >> x) & 1) and ((S >> y) & 1):
                        if w != u and x != u and y != u:
                            valid = False
                            break
                if not valid:
                    continue

                new_S = (S & ~(1 << u)) | (1 << v)
                loss_u = 0
                for w, x in partners3[u]:
                    if ((S >> w) & 1) and ((S >> x) & 1) and w != v and x != v:
                        loss_u += 1
                gain_v = 0
                for w, x in partners3[v]:
                    if ((S >> w) & 1) and ((S >> x) & 1) and w != u and x != u:
                        gain_v += 1
                delta = gain_v - loss_u
                new_score = current_score + delta

            # Accept with Metropolis criterion
            if delta > 0 or (T > 0.01 and rng.random() < pow(2.718281828, delta / max(T, 0.01))):
                S = new_S
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_S = S
            else:
                no_improve += 1

            T *= cooling

        return best_S, best_score

    # ---------- Prune from full ----------
    def prune_from_full():
        S = (1 << n) - 1  # all elements
        # Greedily remove to break 4-APs
        for ap in aps4:
            if all((S >> elem) & 1 for elem in ap):
                # Remove the element with lowest loss_remove
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(S, elem)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << min_elem)
        return S

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

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

    # Orderings 5-12: random shuffles with fixed seeds
    for s in range(8):
        rng = random.Random(100 + s)
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)

    # Run greedy construction for each ordering, then hill-climb
    for i, order in enumerate(orderings):
        if time.time() > deadline - 10:
            break

        S = greedy_construct(order, use_score=True)
        sc = count_3aps(S)

        # Hill-climb with time budget
        remaining = deadline - time.time() - 10
        time_per_hill = max(1.0, remaining / (len(orderings) - i))
        S, sc = hill_climb(S, sc, time.time() + time_per_hill)

        if sc > best_score:
            best_score = sc
            best_S = S

    # Strategy: Prune from full
    if time.time() < deadline - 15:
        S_p = prune_from_full()
        sc_p = count_3aps(S_p)
        remaining = deadline - time.time() - 10
        time_per_hill = max(1.0, remaining / 3)
        S_p, sc_p = hill_climb(S_p, sc_p, time.time() + time_per_hill)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Simulated Annealing phase
    if time.time() < deadline - 10:
        rng_sa = random.Random(12345)
        sa_time = min(20.0, deadline - time.time() - 8)
        S_sa, sc_sa = simulated_annealing(best_S, best_score, time.time() + sa_time, rng_sa, T0=3.0, cooling=0.9995)
        if sc_sa > best_score:
            best_score = sc_sa
            best_S = S_sa

    # Additional perturbation restarts if time remains
    if time.time() < deadline - 8:
        rng = random.Random(999)
        for restart in range(5):
            if time.time() > deadline - 5:
                break

            # Start from best solution and perturb
            S = best_S
            # Randomly remove a few elements
            for _ in range(3):
                if S == 0:
                    break
                bits = [v for v in range(n) if (S >> v) & 1]
                if not bits:
                    break
                v = rng.choice(bits)
                S &= ~(1 << v)

            # Re-add some elements greedily
            for v in range(n):
                if not ((S >> v) & 1) and can_add(S, v):
                    S |= (1 << v)

            # Hill-climb
            remaining = deadline - time.time() - 4
            time_per_hill = max(0.5, remaining / (6 - restart))
            S, sc = hill_climb(S, count_3aps(S), time.time() + time_per_hill)

            if sc > best_score:
                best_score = sc
                best_S = S

    # Final hill climb on best
    if time.time() < deadline - 3 and best_S:
        best_S, best_score = hill_climb(best_S, best_score, deadline - 1)

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(n) if (best_S >> x) & 1]
    if not result:
        result = [1]
    return result
