import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0
    N = 180

    # Build all triples (a<b<c) with 1/a = 1/b + 1/c <=> (b-a)(c-a)=a^2
    triples = []
    seen = set()
    for a in range(1, N + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    t = tuple(sorted((a, b, c)))
                    if len(set(t)) == 3 and t not in seen:
                        seen.add(t)
                        triples.append(t)
                if d != e:
                    b2 = a + e
                    c2 = a + d
                    if b2 <= N and c2 <= N and b2 != c2:
                        t2 = tuple(sorted((a, b2, c2)))
                        if len(set(t2)) == 3 and t2 not in seen:
                            seen.add(t2)
                            triples.append(t2)
            d += 1

    # Build adjacency: for each vertex, list of (other1, other2, tri_idx)
    adj = [[] for _ in range(N + 1)]
    for ti, (a, b, c) in enumerate(triples):
        adj[a].append((b, c, ti))
        adj[b].append((a, c, ti))
        adj[c].append((a, b, ti))

    def count_conflicts(p):
        cnt = 0
        for (a, b, c) in triples:
            if p[a] and p[b] and p[c]:
                cnt += 1
        return cnt

    def conflict_driven_local_search(p, max_iter=20000):
        """
        Structural change: instead of random flips, explicitly target active conflicts.
        For each active triple, try removing one of its three elements and see if we can
        re-add a different absent element to maintain size.
        """
        p = p[:]
        cur_conf = count_conflicts(p)
        cur_size = sum(p[1:])
        best_p = p[:]
        best_size = cur_size
        best_conf = cur_conf

        it = 0
        while it < max_iter:
            if time.time() > DEADLINE - 5:
                break
            it += 1

            if cur_conf == 0:
                # No conflicts: try to add more elements greedily
                added = False
                for v in range(1, N + 1):
                    if p[v]:
                        continue
                    safe = True
                    for (u, w, ti) in adj[v]:
                        if p[u] and p[w]:
                            safe = False
                            break
                    if safe:
                        p[v] = True
                        cur_size += 1
                        added = True
                if not added:
                    break
                if cur_size > best_size:
                    best_size = cur_size
                    best_p = p[:]
                continue

            # Find all active conflicts
            active_triples = []
            for (a, b, c) in triples:
                if p[a] and p[b] and p[c]:
                    active_triples.append((a, b, c))

            if not active_triples:
                break

            # Pick a random active triple
            a, b, c = random.choice(active_triples)

            # Try removing each of the three elements
            candidates = [(a, b, c), (b, a, c), (c, a, b)]
            random.shuffle(candidates)

            improved = False
            for v_rm, u1, u2 in candidates:
                if not p[v_rm]:
                    continue
                # Temporarily remove v_rm
                p[v_rm] = False
                new_conf = count_conflicts(p)
                new_size = cur_size - 1

                if new_conf < cur_conf:
                    # Good: we reduced conflicts. Now try to add back an absent element
                    best_add = None
                    best_add_conf = new_conf
                    for v_add in range(1, N + 1):
                        if p[v_add]:
                            continue
                        # Check if adding v_add creates conflicts
                        add_conf = 0
                        for (u, w, ti) in adj[v_add]:
                            if p[u] and p[w]:
                                add_conf += 1
                        if add_conf == 0 and new_size + 1 > new_size:
                            best_add = v_add
                            break
                        elif add_conf < best_add_conf and new_size + 1 >= new_size:
                            best_add = v_add
                            best_add_conf = add_conf

                    if best_add is not None and best_add_conf == 0:
                        p[best_add] = True
                        cur_conf = best_add_conf
                        cur_size = new_size + 1
                        improved = True
                    elif best_add is not None and best_add_conf < cur_conf:
                        p[best_add] = True
                        cur_conf = best_add_conf
                        cur_size = new_size + 1
                        improved = True
                    else:
                        # Can't add back, keep the removal
                        cur_conf = new_conf
                        cur_size = new_size
                        improved = True
                    break
                else:
                    # Revert
                    p[v_rm] = True

            if improved:
                if cur_conf < best_conf or (cur_conf == best_conf and cur_size > best_size):
                    best_conf = cur_conf
                    best_size = cur_size
                    best_p = p[:]
            else:
                # Stuck: try a random flip to escape
                v = random.randint(1, N)
                old = p[v]
                p[v] = not old
                new_conf = count_conflicts(p)
                new_size = cur_size + (1 if not old else -1)
                if new_conf <= cur_conf:
                    cur_conf = new_conf
                    cur_size = new_size
                    if cur_conf < best_conf or (cur_conf == best_conf and cur_size > best_size):
                        best_conf = cur_conf
                        best_size = cur_size
                        best_p = p[:]
                else:
                    p[v] = old

        return best_p, best_size

    def simulated_annealing(p, time_limit=30.0):
        """SA with conflict-based objective"""
        p = p[:]
        cur_conf = count_conflicts(p)
        cur_size = sum(p[1:])
        best_p = p[:]
        best_size = cur_size
        best_conf = cur_conf

        T = 2.0
        TMIN = 0.005
        sa_end = time.time() + time_limit
        step = 0
        while time.time() < sa_end and T > TMIN:
            v = random.randint(1, N)
            old = p[v]
            # Compute delta in conflicts
            dconf = 0
            for (u, w, ti) in adj[v]:
                if p[u] and p[w]:
                    if not old:
                        dconf += 1
                    else:
                        dconf -= 1
            dsize = 1 if not old else -1
            # Objective: maximize size, minimize conflicts
            old_score = cur_size - 10000 * cur_conf
            new_conf = cur_conf + dconf
            new_size = cur_size + dsize
            new_score = new_size - 10000 * new_conf
            ds = new_score - old_score
            if ds >= 0 or random.random() < math.exp(ds / T):
                p[v] = not old
                cur_conf = new_conf
                cur_size = new_size
                if cur_conf < best_conf or (cur_conf == best_conf and cur_size > best_size):
                    best_conf = cur_conf
                    best_size = cur_size
                    best_p = p[:]
            T *= 0.9997
            step += 1

        return best_p, best_size

    def greedy_add(p, order):
        """Add elements in given order if safe"""
        p = p[:]
        for v in order:
            if p[v]:
                continue
            safe = True
            for (u, w, ti) in adj[v]:
                if p[u] and p[w]:
                    safe = False
                    break
            if safe:
                p[v] = True
        return p

    # Multiple restarts with different strategies
    random.seed(42)
    best_set = None
    best_size = 0

    # Strategy 1: Start from high numbers
    for lo in [80, 90, 100, 110, 120]:
        p = [False] * (N + 1)
        for v in range(lo, N + 1):
            p[v] = True
        # Greedy add remaining
        p = greedy_add(p, list(range(1, lo)))
        conf = count_conflicts(p)
        sz = sum(p[1:])
        if conf == 0 and sz > best_size:
            best_size = sz
            best_set = p[:]
        # Conflict-driven local search polish
        if time.time() < DEADLINE - 10:
            p2, sz2 = conflict_driven_local_search(p)
            if sz2 > best_size and count_conflicts(p2) == 0:
                best_size = sz2
                best_set = p2[:]

    # Strategy 2: Random restarts
    for restart in range(8):
        if time.time() > DEADLINE - 15:
            break
        p = [False] * (N + 1)
        # Random initial: include ~50% of numbers
        for v in range(1, N + 1):
            if random.random() < 0.5:
                p[v] = True
        # Greedy add
        p = greedy_add(p, list(range(1, N + 1)))
        conf = count_conflicts(p)
        sz = sum(p[1:])
        if conf == 0 and sz > best_size:
            best_size = sz
            best_set = p[:]
        # Conflict-driven local search
        if time.time() < DEADLINE - 10:
            p2, sz2 = conflict_driven_local_search(p)
            if sz2 > best_size and count_conflicts(p2) == 0:
                best_size = sz2
                best_set = p2[:]

    # Final polish on best
    if best_set is None:
        best_set = [False] * (N + 1)
        for v in range(91, N + 1):
            best_set[v] = True
        best_size = sum(best_set[1:])

    if time.time() < DEADLINE - 5:
        p2, sz2 = conflict_driven_local_search(best_set)
        if sz2 > best_size and count_conflicts(p2) == 0:
            best_size = sz2
            best_set = p2[:]

    # Greedy add any remaining safe elements
    if time.time() < DEADLINE - 2:
        best_set = greedy_add(best_set, list(range(1, N + 1)))

    res = [v for v in range(1, N + 1) if best_set[v]]
    if not res:
        res = list(range(91, N + 1))
    return res
