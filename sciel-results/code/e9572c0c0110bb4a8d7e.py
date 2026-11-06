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

    def is_valid(p):
        """Check if present set has no active triples"""
        for (a, b, c) in triples:
            if p[a] and p[b] and p[c]:
                return False
        return True

    def count_conflicts(p):
        cnt = 0
        for (a, b, c) in triples:
            if p[a] and p[b] and p[c]:
                cnt += 1
        return cnt

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

    def local_search_2flip(p, max_iter=5000):
        """Try 2-flip moves: remove one present, add one absent"""
        p = p[:]
        cur_conf = count_conflicts(p)
        cur_size = sum(p[1:])
        best_p = p[:]
        best_size = cur_size
        best_conf = cur_conf

        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            # Try random 2-flips
            present_list = [v for v in range(1, N+1) if p[v]]
            absent_list = [v for v in range(1, N+1) if not p[v]]
            if not present_list or not absent_list:
                break
            v_rm = random.choice(present_list)
            v_add = random.choice(absent_list)
            if v_rm == v_add:
                continue
            # Check if this 2-flip reduces conflicts or maintains while same size
            old_conf = cur_conf
            # Compute delta
            delta = 0
            for (u, w, ti) in adj[v_rm]:
                if p[u] and p[w]:
                    delta -= 1  # removing v_rm breaks this conflict
            for (u, w, ti) in adj[v_add]:
                if p[u] and p[w] and u != v_rm and w != v_rm:
                    delta += 1  # adding v_add creates this conflict
                elif p[u] and p[w]:
                    # one of u,w is v_rm which we're removing
                    # need to check the other
                    pass
            # More careful: just compute new conflicts
            p[v_rm] = False
            p[v_add] = True
            new_conf = count_conflicts(p)
            new_size = sum(p[1:])
            if new_conf < old_conf or (new_conf == old_conf and new_size > best_size):
                cur_conf = new_conf
                cur_size = new_size
                if new_conf == 0 and new_size > best_size:
                    best_size = new_size
                    best_p = p[:]
            else:
                p[v_rm] = True
                p[v_add] = False

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
            # Score = size - 10000 * conflicts
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
        # SA polish
        if time.time() < DEADLINE - 10:
            p2, sz2 = simulated_annealing(p, time_limit=8.0)
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
        # SA
        if time.time() < DEADLINE - 10:
            p2, sz2 = simulated_annealing(p, time_limit=10.0)
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
        p2, sz2 = simulated_annealing(best_set, time_limit=15.0)
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
