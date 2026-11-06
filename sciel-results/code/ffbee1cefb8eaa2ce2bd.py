import time
import random
import numpy as np

def construct():
    n = 40
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - start_time)

    def bitsets_to_edges(A):
        edges = []
        for i in range(n):
            for j in np.nonzero(A[i])[0]:
                if i < j:
                    edges.append([int(i), int(j)])
        return edges

    # ---------- Bitset-based C6 check ----------
    # A is an n x n 0/1 adjacency matrix (no self loops).
    # Adding edge (u,v) creates a C6 iff there is a length-4 path u-...-v
    # with distinct internal vertices. We test the 4-step reachability and
    # verify vertex-distinctness for the small number of candidates.
    def creates_c6(A, u, v):
        # Candidate intermediate sequences x-y-z-w with
        # u-x, x-y, y-z, z-w, w-v all edges, all of {x,y,z,w} distinct
        # and not equal to u or v.
        # Fast path: 4-step boolean reachability via matrix powers.
        # R2 = A^2 (paths length 2), R4 = R2^2 (paths length 4).
        # A length-4 path from u to v means R4[u,v] > 0.
        # But we must ensure a simple path (distinct internal vertices).
        # For speed we first test R4; if 0, no C6. If >0, verify distinctness.
        R2 = A @ A
        R4 = R2 @ R2
        if R4[u, v] == 0:
            return False
        # Verify existence of a distinct-vertex path.
        # Enumerate: x in N(u), w in N(v), need length-2 path x-...-w
        # through y,z distinct from {u,v,x,w} and from each other.
        Nu = set(np.nonzero(A[u])[0])
        Nv = set(np.nonzero(A[v])[0])
        # Precompute length-2 neighborhoods
        for x in Nu:
            if x == v:
                continue
            Nx = set(np.nonzero(A[x])[0])
            for w in Nv:
                if w == u or w == x:
                    continue
                Nw = set(np.nonzero(A[w])[0])
                # y in N(x) \ {u,v,x,w}, z in N(y) ∩ N(w), z distinct
                for y in Nx:
                    if y in (u, v, x, w):
                        continue
                    Ny = set(np.nonzero(A[y])[0])
                    # z in Ny ∩ Nw, z not in {u,v,x,y,w}
                    for z in Ny & Nw:
                        if z in (u, v, x, y, w):
                            continue
                        return True
        return False

    # ---------- Build a dense seed skeleton ----------
    # Start from 8 disjoint K5s (80 edges, C6-free), then add inter-block
    # edges greedily. This gives a denser starting point than pure random.
    def build_seed():
        A = np.zeros((n, n), dtype=np.int8)
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    A[5 * k + a, 5 * k + b] = 1
                    A[5 * k + b, 5 * k + a] = 1
        return A

    def greedy_add(A, order, time_budget):
        for (u, v) in order:
            if time.time() - start_time > time_budget:
                break
            if A[u, v]:
                continue
            if not creates_c6(A, u, v):
                A[u, v] = 1
                A[v, u] = 1
        return A

    def all_missing_edges(A):
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                if not A[i, j]:
                    out.append((i, j))
        return out

    def edge_count(A):
        return int(A.sum() // 2)

    best_A = None
    best_ec = -1

    # Phase 1: multiple randomized greedy restarts on top of K5 seed
    random.seed(42)
    for trial in range(60):
        if time_left() < 8:
            break
        A = build_seed()
        missing = all_missing_edges(A)
        random.shuffle(missing)
        # Reserve time for later phases
        greedy_time = min(time_left() - 40, 2.0)
        if greedy_time < 0.2:
            greedy_time = 0.2
        A = greedy_add(A, missing, start_time + greedy_time)
        ec = edge_count(A)
        if ec > best_ec:
            best_ec = ec
            best_A = A.copy()

    if best_A is None:
        best_A = build_seed()

    # Phase 2: add any remaining edge that fits (fast pass)
    A = best_A
    improved = True
    while improved and time_left() > 10:
        improved = False
        for (u, v) in all_missing_edges(A):
            if time_left() < 10:
                break
            if not creates_c6(A, u, v):
                A[u, v] = 1
                A[v, u] = 1
                improved = True
    if edge_count(A) > best_ec:
        best_ec = edge_count(A)
        best_A = A.copy()

    # Phase 3: 2-opt edge swap local search (remove one, add another)
    A = best_A.copy()
    for trial in range(400):
        if time_left() < 15:
            break
        cur_ec = edge_count(A)
        improved = False
        # Pick a random edge to remove
        edge_list = [(int(i), int(j)) for i in range(n) for j in np.nonzero(A[i])[0] if i < j]
        if not edge_list:
            break
        random.shuffle(edge_list)
        # Try removing a few edges, for each try to add the best missing edge
        for (ru, rv) in edge_list[:6]:
            if time_left() < 15:
                break
            A[ru, rv] = 0
            A[rv, ru] = 0
            # candidate additions: missing edges
            cands = all_missing_edges(A)
            random.shuffle(cands)
            added = False
            for (au, av) in cands:
                if time_left() < 15:
                    break
                if not creates_c6(A, au, av):
                    A[au, av] = 1
                    A[av, au] = 1
                    if edge_count(A) > cur_ec:
                        improved = True
                        added = True
                        break
                    else:
                        # no gain, revert this addition
                        A[au, av] = 0
                        A[av, au] = 0
            if not added:
                # restore removed edge
                A[ru, rv] = 1
                A[rv, ru] = 1
            if improved:
                break
        if not improved:
            # try a fresh random restart of swaps with perturbation
            if time_left() < 25:
                break
            # perturb: remove a few random edges then re-greedy
            for _ in range(3):
                if edge_list:
                    pu, pv = random.choice(edge_list)
                    A[pu, pv] = 0
                    A[pv, pu] = 0
            missing = all_missing_edges(A)
            random.shuffle(missing)
            greedy_time = min(time_left() - 25, 1.0)
            if greedy_time > 0.2:
                A = greedy_add(A, missing, start_time + greedy_time)
            if edge_count(A) > best_ec:
                best_ec = edge_count(A)
                best_A = A.copy()

    if edge_count(A) > best_ec:
        best_ec = edge_count(A)
        best_A = A.copy()

    # Final: ensure C6-free by removing any offending edge (safety)
    A = best_A
    # Verify and fix: for each edge, if removing is not needed it's fine.
    # We trust the incremental check; do a light final pass to drop any
    # edge that the bitset test now flags (in case of a bug).
    for (u, v) in [(int(i), int(j)) for i in range(n) for j in np.nonzero(A[i])[0] if i < j]:
        if time_left() < 5:
            break
        A[u, v] = 0
        A[v, u] = 0
        if creates_c6(A, u, v):
            # keep it removed
            continue
        A[u, v] = 1
        A[v, u] = 1

    return bitsets_to_edges(A)
