"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16

    # ---- Precompute Q3 masks (12-edge 3-regular submasks of K4,4) ----
    def is_q3_mask(mask):
        da = [0]*4
        db = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    da[i] += 1
                    db[j] += 1
        return all(d == 3 for d in da) and all(d == 3 for d in db)

    q3_masks = []
    for combo in itertools.combinations(range(16), 12):
        m = 0
        for idx in combo:
            m |= (1 << idx)
        if is_q3_mask(m):
            q3_masks.append(m)
    q3_masks = tuple(q3_masks)  # 6 masks

    # ---- Precompute all K4,4 splits (A of size 4, B of size 4) ----
    all_verts = list(range(n))
    k44_list = []  # each: (A tuple, B tuple, pairs [(a,b,bitpos),...])
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            pairs = []
            for i in range(4):
                for j in range(4):
                    pairs.append((A[i], B[j], i*4 + j))
            k44_list.append((A, B, pairs))

    # ---- Index: for each edge, which K4,4 splits contain it ----
    edge_to_k44 = {}
    for idx, (A, B, pairs) in enumerate(k44_list):
        for a, b, _ in pairs:
            e = (a, b) if a < b else (b, a)
            edge_to_k44.setdefault(e, []).append(idx)

    # ---- Build graph as bitsets ----
    adj = [0] * n
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    def edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    # ---- Seed: K7 + K7 + K2 (57 edges, Q3-free) ----
    for i in range(7):
        for j in range(i+1, 7):
            add_edge(i, j)
    for i in range(7, 14):
        for j in range(i+1, 14):
            add_edge(i, j)
    add_edge(14, 15)

    best_edges = edge_count()
    best_adj = adj[:]

    def time_left():
        return deadline - time.time()

    # ---- Check if adding (u,v) creates a Q3 ----
    def creates_q3(u, v):
        e = (u, v) if u < v else (v, u)
        for kidx in edge_to_k44.get(e, ()):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if (a == u and b == v) or (a == v and b == u):
                    mask |= (1 << bit)  # the new edge
                elif has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    return True
        return False

    # ---- Check if current graph has ANY Q3 (full scan) ----
    def find_q3():
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    return True
        return False

    # ---- Count number of Q3 subgraphs in current graph ----
    def count_q3():
        cnt = 0
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    cnt += 1
        return cnt

    # ---- Greedy: try to add every missing edge ----
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = [e for e in all_pairs if not has_edge(*e)]
    random.seed(42)
    random.shuffle(missing)

    for u, v in missing:
        if time_left() < 15:
            break
        if not creates_q3(u, v):
            add_edge(u, v)
            if edge_count() > best_edges:
                best_edges = edge_count()
                best_adj = adj[:]

    # ---- MIS-based Branch and Bound on conflict hypergraph ----
    # Nodes: the 120 potential edges
    # Hyperedges: minimal sets of edges that complete a Q3 embedding
    # We want max independent set = max edges with no Q3

    # Precompute conflict structure using bitsets
    # For each edge e, compute which other edges conflict with it
    # (i.e., adding both e and f would create a Q3 given current state)
    
    # Actually, let's use a different approach: 
    # Start from current best, try to add edges via branch and bound
    # where we track which Q3 embeddings are "active" (have all but one edge present)
    
    # For efficiency, use the existing creates_q3 but with memoization
    
    # ---- Simulated Annealing on edge set ----
    # Objective: maximize edges subject to Q3-free constraint
    # Penalty approach: score = edges - penalty * q3_count
    # We allow temporary violations to escape local optima

    if time_left() > 20:
        # Start SA from current best
        adj[:] = best_adj[:]
        sa_edges = edge_count()
        sa_q3 = count_q3()  # should be 0

        # Temperature schedule
        T = 5.0
        cooling = 0.9995
        min_T = 0.1

        # Precompute list of all edges for random selection
        all_pairs_list = all_pairs[:]

        # Track which edges are in the current set for faster access
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if has_edge(i, j):
                    edge_set.add((i, j))

        while time_left() > 10 and T > min_T:
            # Pick a random edge to toggle
            u, v = random.choice(all_pairs_list)
            e = (u, v) if u < v else (v, u)
            if e in edge_set:
                # Try removing
                remove_edge(u, v)
                edge_set.remove(e)
                new_edges = sa_edges - 1
                new_q3 = count_q3()
                new_score = new_edges - 100 * new_q3
                old_score = sa_edges - 100 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
                    edge_set.add(e)
            else:
                # Try adding
                add_edge(u, v)
                edge_set.add(e)
                new_edges = sa_edges + 1
                new_q3 = count_q3()
                new_score = new_edges - 100 * new_q3
                old_score = sa_edges - 100 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)
                    edge_set.discard(e)

            T *= cooling

        # After SA, ensure we have a valid Q3-free graph
        # If current state has Q3s, try to remove edges to eliminate them
        if count_q3() > 0:
            # Greedily remove edges that are in Q3s
            for _ in range(100):
                if count_q3() == 0:
                    break
                if time_left() < 5:
                    break
                # Find an edge that participates in a Q3
                found = False
                for u, v in all_pairs:
                    if not has_edge(u, v):
                        continue
                    # Check if removing this edge reduces Q3 count
                    remove_edge(u, v)
                    if count_q3() < count_q3():
                        found = True
                        break
                    add_edge(u, v)
                if not found:
                    break

    # ---- Local search: 1-1 edge swaps to gain density ----
    adj[:] = best_adj[:]
    for _ in range(2000):
        if time_left() < 10:
            break
        # Pick a random edge to remove
        rem_edges = [e for e in all_pairs if has_edge(*e)]
        if not rem_edges:
            break
        ru, rv = random.choice(rem_edges)
        remove_edge(ru, rv)

        # Now try to add two missing edges
        miss = [e for e in all_pairs if not has_edge(*e)]
        random.shuffle(miss)
        added = 0
        for u, v in miss:
            if added >= 2:
                break
            if not creates_q3(u, v):
                add_edge(u, v)
                added += 1
        # After adding up to 2, check we gained
        cur = edge_count()
        if cur > best_edges:
            best_edges = cur
            best_adj = adj[:]
            # continue with this state
        else:
            # revert: remove the added edges and restore removed
            for u, v in miss[:added]:
                remove_edge(u, v)
            add_edge(ru, rv)

    # ---- Restore best state ----
    adj[:] = best_adj[:]

    # Final validity check
    if find_q3():
        # Fallback to seed
        adj = [0]*n
        for i in range(7):
            for j in range(i+1, 7):
                add_edge(i, j)
        for i in range(7, 14):
            for j in range(i+1, 14):
                add_edge(i, j)
        add_edge(14, 15)

    return [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
# EVOLVE-BLOCK-END
