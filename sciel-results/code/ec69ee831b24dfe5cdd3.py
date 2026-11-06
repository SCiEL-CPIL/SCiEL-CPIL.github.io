import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s/150s limit

    def time_left():
        return deadline - time.time()

    # ---------- Fast C5 check ----------
    def has_c5(adj, n):
        """C5 exists iff some edge (u,v) has a simple path of length 3
        between N(u)\{v} and N(v)\{u} in G - {u,v}.
        Equivalent to: exists b not in {u,v} with
        a in N(u)\{v}, c in N(v)\{u}, a-b, b-c edges, a!=c,
        and (a,v) not an edge, (c,u) not an edge (to keep the cycle simple)."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                Nu = adj[u] - {v}
                Nv = adj[v] - {u}
                if not Nu or not Nv:
                    continue
                # BFS from Nu, tracking distance, up to 3
                # dist 1: Nu ; dist 2: neighbors of Nu (excluding u,v) ; dist 3: neighbors of those
                # We need a vertex c in Nv at distance exactly 3, with intermediate b
                # (b at dist 2, a at dist 1). Also ensure a!=c and the cycle is simple.
                # Simple approach:
                #   For each b in V - {u,v}:
                #       A = adj[b] & Nu   (candidate a)
                #       C = adj[b] & Nv   (candidate c)
                #       if A and C:
                #           for a in A:
                #               for c in C:
                #                   if a == c: continue
                #                   # cycle u-a-b-c-v-u ; ensure simple:
                #                   # u,a,b,c,v distinct. a in Nu so a!=u,v. c in Nv so c!=u,v.
                #                   # a!=b (a in adj[b]), c!=b. a!=c checked.
                #                   # Also need no shortcut making it not a 5-cycle?
                #                   # A 5-cycle just needs the 5 edges to exist and 5 distinct vertices.
                #                   # Edges: u-a (yes), a-b (yes), b-c (yes), c-v (yes), v-u (yes).
                #                   return True
                # This is sufficient: if such a,b,c exist with a!=c, we have a C5.
                for b in range(n):
                    if b == u or b == v:
                        continue
                    A = adj[b] & Nu
                    if not A:
                        continue
                    C = adj[b] & Nv
                    if not C:
                        continue
                    # need a in A, c in C with a != c
                    if len(A) + len(C) > 0:
                        # find a != c
                        if A != C or len(A) > 1:
                            # there exist distinct a,c
                            # quick check
                            found = False
                            for a in A:
                                for c in C:
                                    if a != c:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                return True
        return False

    # ---------- Fast independent set of size 9 check ----------
    def has_independent_set_9(adj, n):
        # We want to know if the complement graph has a clique of size 9,
        # i.e. original has independent set of size 9.
        # Branch and bound with greedy coloring upper bound on candidates.
        candidates = set(range(n))

        def greedy_color_bound(cand):
            # Upper bound on clique size in complement induced by cand.
            # Equivalent to: max independent set in original induced by cand.
            # Greedy: repeatedly pick a vertex, remove it and its neighbors (in original)
            # from cand. Count picks. This is a lower bound on alpha, not upper.
            # For an UPPER bound, use greedy coloring of the complement (i.e., partition cand
            # into cliques of original = independent sets). Simpler: use the standard
            # "color the complement greedily" which gives an upper bound on clique size.
            # Complement edge between x,y iff x NOT in adj[y].
            # Greedy coloring of complement: assign colors so that adjacent-in-complement
            # (non-adjacent-in-original) get different colors. Number of colors = upper bound
            # on clique size in complement = upper bound on independent set size in original.
            verts = sorted(cand)
            color_of = {}
            color_count = 0
            # order by degree in complement (ascending) for tighter bound
            def comp_deg(v):
                return len(cand) - 1 - len(adj[v] & cand)
            verts = sorted(verts, key=comp_deg)
            for v in verts:
                used = set()
                for w in cand:
                    if w == v:
                        continue
                    if w not in adj[v]:  # adjacent in complement
                        if w in color_of:
                            used.add(color_of[w])
                c = 0
                while c in used:
                    c += 1
                color_of[v] = c
                color_count = max(color_count, c + 1)
            return color_count

        best = 0

        def search(cand):
            nonlocal best
            if not cand:
                return False
            if len(cand) <= 8:
                return False  # can't reach 9 from here (we only care >=9)
            ub = greedy_color_bound(cand)
            if ub < 9:
                return False
            # pick a vertex to branch on (high degree in complement = low degree in original)
            v = max(cand, key=lambda x: (len(cand) - len(adj[x] & cand)))
            # Include v: candidates become cand \ (adj[v] ∪ {v})  [in original, can't take neighbors]
            new_cand = cand - adj[v] - {v}
            if search(new_cand):
                return True
            # Exclude v
            if search(cand - {v}):
                return True
            return False

        return search(cand)

    def is_valid(n, adj):
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    # ---------- Known valid: 8 copies of K4 (n=32, alpha=8, C5-free) ----------
    def make_8k4():
        n = 32
        adj = [set() for _ in range(n)]
        for b in range(8):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * b + i
                    v = 4 * b + j
                    adj[u].add(v)
                    adj[v].add(u)
        return n, adj

    n_best, adj_best = make_8k4()
    if is_valid(n_best, adj_best):
        pass  # confirmed
    else:
        # Should never happen; keep as fallback anyway.
        pass

    random.seed(12345)

    # ---------- Search for larger n ----------
    # Try to find valid graphs for n = 33, 34, 35.
    # Use random graphs with tuned edge probability, plus local search (edge flips)
    # to escape local optima.

    def try_random_graphs(target_n, attempts, p_lo, p_hi):
        nonlocal n_best, adj_best
        for attempt in range(attempts):
            if time_left() < 5:
                return False
            p = random.uniform(p_lo, p_hi)
            adj = [set() for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            if not is_valid(target_n, adj):
                continue
            n_best = target_n
            adj_best = adj
            return True
        return False

    def local_search(adj, target_n, max_iters):
        """Try to fix violations by flipping edges. Returns (adj, valid)."""
        nonlocal n_best, adj_best
        cur = [set(s) for s in adj]
        best_score = None
        for it in range(max_iters):
            if time_left() < 5:
                break
            # compute violations
            c5 = has_c5(cur, target_n)
            ind = has_independent_set_9(cur, target_n)
            if not c5 and not ind:
                n_best = target_n
                adj_best = cur
                return True
            # pick a random edge to flip
            i = random.randrange(target_n)
            j = random.randrange(target_n)
            if i == j:
                continue
            if j in cur[i]:
                cur[i].discard(j)
                cur[j].discard(i)
            else:
                cur[i].add(j)
                cur[j].add(i)
        return False

    # Phase 1: random graphs for n=33
    if n_best < 33:
        found = try_random_graphs(33, 300, 0.55, 0.85)
        if not found and time_left() > 20:
            # local search from random starts
            for attempt in range(20):
                if time_left() < 10:
                    break
                p = random.uniform(0.6, 0.8)
                adj = [set() for _ in range(33)]
                for i in range(33):
                    for j in range(i + 1, 33):
                        if random.random() < p:
                            adj[i].add(j)
                            adj[j].add(i)
                if local_search(adj, 33, 2000):
                    break

    # Phase 2: random graphs for n=34
    if n_best < 34:
        found = try_random_graphs(34, 200, 0.6, 0.9)
        if not found and time_left() > 20:
            for attempt in range(15):
                if time_left() < 10:
                    break
                p = random.uniform(0.65, 0.85)
                adj = [set() for _ in range(34)]
                for i in range(34):
                    for j in range(i + 1, 34):
                        if random.random() < p:
                            adj[i].add(j)
                            adj[j].add(i)
                if local_search(adj, 34, 1500):
                    break

    # Phase 3: random graphs for n=35
    if n_best < 35:
        found = try_random_graphs(35, 150, 0.65, 0.92)
        if not found and time_left() > 15:
            for attempt in range(10):
                if time_left() < 8:
                    break
                p = random.uniform(0.7, 0.88)
                adj = [set() for _ in range(35)]
                for i in range(35):
                    for j in range(i + 1, 35):
                        if random.random() < p:
                            adj[i].add(j)
                            adj[j].add(i)
                if local_search(adj, 35, 1200):
                    break

    # Phase 4: structured extension from 8xK4 for n=33
    if n_best < 33 and time_left() > 15:
        base_n, base_adj = make_8k4()
        for attempt in range(300):
            if time_left() < 5:
                break
            adj = [set(s) for s in base_adj]
            adj.append(set())
            n = 33
            # Connect vertex 32 to a random subset
            for i in range(32):
                if random.random() < 0.5:
                    adj[32].add(i)
                    adj[i].add(32)
            if is_valid(n, adj):
                n_best = 33
                adj_best = adj
                break

    # Final: ensure we have a valid graph
    if not is_valid(n_best, adj_best):
        n_best, adj_best = make_8k4()

    return {'n': n_best, 'edges': adj_to_edges(adj_best, n_best)}
