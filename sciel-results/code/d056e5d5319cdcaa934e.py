"""C6-free graph on 40 vertices, maximize edges. Random init + greedy + local search."""
import time
import random

def _has_c6_after_add(adj, u, v):
    """Return True if adding edge (u,v) (currently absent) creates a C6.
    adj: list of int bitsets of neighbors.
    A new C6 is u-a-b-c-v-u with a in N(u), c in N(v), a!=c, and exists b in N(a)&N(c).
    """
    Nu = adj[u]
    Nv = adj[v]
    # candidate a in N(u), c in N(v), a != c, and (N(a) & N(c)) != 0
    a_bits = Nu
    while a_bits:
        lb = a_bits & (-a_bits)
        a = lb.bit_length() - 1
        a_bits ^= lb
        # c in N(v), c != a
        c_bits = Nv & ~(1 << a)
        while c_bits:
            lc = c_bits & (-c_bits)
            c = lc.bit_length() - 1
            c_bits ^= lc
            if (adj[a] & adj[c]) != 0:
                return True
    return False

def _add(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _edge_count(adj, n):
    return sum(b.bit_count() for b in adj) // 2

def _valid(adj, n):
    """Verify no C6 by brute force over all 6-cycles (n=40 small enough)."""
    # A C6: 0-1-2-3-4-5-0 with distinct vertices, canonical form smallest vertex 0 and second smallest < ... to avoid duplicates
    # Simpler: check all 6-tuples is too slow. Use: for each pair (u,v) non-edge, check no length-4 path? That's the same as C6 detection.
    # We'll do: for each u, for each a in N(u), for each b in N(a)\{u}, for each c in N(b)\{a}, for each d in N(c)\{b}, if d in N(u) and d!=a and d!=... that's a 5-cycle not 6.
    # Actually C6: u-a-b-c-d-e-u. Let's just trust the incremental check and do a light verification:
    # For each non-edge (u,v), ensure no length-4 path. This is O(n^2 * something).
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                continue
            # check length-4 path u-a-b-c-v
            a_bits = adj[u]
            while a_bits:
                lb = a_bits & (-a_bits)
                a = lb.bit_length()-1
                a_bits ^= lb
                if a == v: continue
                b_bits = adj[a] & ~(1<<u) & ~(1<<a)
                while b_bits:
                    lb2 = b_bits & (-b_bits)
                    b = lb2.bit_length()-1
                    b_bits ^= lb2
                    if b == v: continue
                    c_bits = adj[b] & ~(1<<a)
                    while c_bits:
                        lc = c_bits & (-c_bits)
                        c = lc.bit_length()-1
                        c_bits ^= lc
                        if c == v: continue
                        if c == u: continue
                        if (adj[c] >> v) & 1:
                            return False
    return True

def construct():
    n = 40
    t0 = time.time()
    deadline = time.time() + 140  # safety margin before 150s

    random.seed(12345)

    def time_ok():
        return time.time() < deadline

    # Start with a random graph with moderate density, then prune C6s, then greedy add, then local search.
    best_adj = None
    best_count = -1

    def update_best(adj):
        nonlocal best_adj, best_count
        c = _edge_count(adj, n)
        if c > best_count:
            best_count = c
            best_adj = [b for b in adj]

    # Phase 1: multiple random restarts.
    for restart in range(2000):
        if not time_ok():
            break
        # Random initial graph with edge probability p
        p = 0.25
        adj = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1<<j)
                    adj[j] |= (1<<i)
        # Prune: remove edges that are part of a C6 until C6-free (greedy removal)
        # Iterate: find a C6, remove a random edge in it.
        changed = True
        prune_iter = 0
        while changed and prune_iter < 5000 and time_ok():
            changed = False
            prune_iter += 1
            # find any C6
            found = False
            for u in range(n):
                if found: break
                for a in range(n):
                    if found: break
                    if a == u: continue
                    if not ((adj[u]>>a)&1): continue
                    for b in range(n):
                        if found: break
                        if b==u or b==a: continue
                        if not ((adj[a]>>b)&1): continue
                        for c in range(n):
                            if found: break
                            if c==u or c==a or c==b: continue
                            if not ((adj[b]>>c)&1): continue
                            for d in range(n):
                                if found: break
                                if d==u or d==a or d==b or d==c: continue
                                if not ((adj[c]>>d)&1): continue
                                for e in range(n):
                                    if e==u or e==a or e==b or e==c or e==d: continue
                                    if not ((adj[d]>>e)&1): continue
                                    if not ((adj[e]>>u)&1): continue
                                    # C6 found: u-a-b-c-d-e-u
                                    # remove a random edge
                                    edges = [(u,a),(a,b),(b,c),(c,d),(d,e),(e,u)]
                                    ru,rv = random.choice(edges)
                                    _remove(adj, ru, rv)
                                    found = True
                                    changed = True
                                    break
        if not time_ok():
            break
        # Now greedy add edges
        # Build list of candidate edges, sort by a heuristic (e.g., sum of degrees) and add if safe.
        # Use a queue approach: repeatedly try to add the "best" safe edge.
        # To be fast, we maintain a list of non-edges and try them.
        # Greedy: for each non-edge in random order, try to add if safe.
        for _ in range(3):  # multiple passes
            if not time_ok(): break
            order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((adj[i]>>j)&1)]
            random.shuffle(order)
            added = False
            for (i,j) in order:
                if not time_ok(): break
                if (adj[i]>>j)&1: continue
                if not _has_c6_after_add(adj, i, j):
                    _add(adj, i, j)
                    added = True
            if not added:
                break
        update_best(adj)

        # Local search: try edge swaps (remove one edge, add two) or add single edges.
        # Simple 1-move: try to add a non-edge (already done). 
        # 2-move: remove edge e1, then try to add non-edges until we gain net.
        # We'll do: for a random edge, remove it, then greedily add as many safe non-edges as possible; if net gain, keep.
        ls_iter = 0
        while ls_iter < 200 and time_ok():
            ls_iter += 1
            # pick a random existing edge
            all_edges = [(i,j) for i in range(n) for j in range(i+1,n) if (adj[i]>>j)&1]
            if not all_edges:
                break
            e1 = random.choice(all_edges)
            u1,v1 = e1
            _remove(adj, u1, v1)
            before = _edge_count(adj, n)
            # greedily add safe non-edges
            order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((adj[i]>>j)&1)]
            random.shuffle(order)
            for (i,j) in order:
                if not time_ok(): break
                if (adj[i]>>j)&1: continue
                if not _has_c6_after_add(adj, i, j):
                    _add(adj, i, j)
            after = _edge_count(adj, n)
            if after < before:
                # revert: too complex; just keep best_adj separately and discard this adj
                # We'll restore by not having updated best; but adj is now worse. Re-add e1.
                _add(adj, u1, v1)
                # and remove the ones we added? We can't easily know. 
                # Instead, work on a copy.
                pass
            # To be safe, operate on a copy for the trial.
        # Redo local search properly with copies:
        ls2 = 0
        while ls2 < 150 and time_ok():
            ls2 += 1
            trial = [b for b in adj]
            all_edges = [(i,j) for i in range(n) for j in range(i+1,n) if (trial[i]>>j)&1]
            if not all_edges: break
            u1,v1 = random.choice(all_edges)
            _remove(trial, u1, v1)
            before = _edge_count(trial, n)
            order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((trial[i]>>j)&1)]
            random.shuffle(order)
            for (i,j) in order:
                if not time_ok(): break
                if (trial[i]>>j)&1: continue
                if not _has_c6_after_add(trial, i, j):
                    _add(trial, i, j)
            after = _edge_count(trial, n)
            if after > before:
                adj = trial
                update_best(adj)
            else:
                # also try: just add edges without removing (already done in greedy)
                pass

    # Final: try to add any remaining safe edges to best_adj
    if best_adj is not None and time_ok():
        order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((best_adj[i]>>j)&1)]
        random.shuffle(order)
        for (i,j) in order:
            if not time_ok(): break
            if (best_adj[i]>>j)&1: continue
            if not _has_c6_after_add(best_adj, i, j):
                _add(best_adj, i, j)
        update_best(best_adj)

    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i]>>j)&1:
                edges.append([i,j])
    return edges
