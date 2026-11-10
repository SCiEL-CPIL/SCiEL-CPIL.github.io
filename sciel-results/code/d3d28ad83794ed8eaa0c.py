"""
Q3-free dense graph on 16 vertices — complement-removal + simulated-annealing local search.

Strategy (structural change from previous greedy+swap lineage):
  1. Start from the complete graph K16 (120 edges).
  2. Repeatedly detect a Q3 subgraph and remove a random edge from it.
  3. After reaching a Q3-free state, run a simulated-annealing local search:
     - Moves: add a missing edge (if it doesn't create Q3), remove an existing edge,
       or swap (remove one, add another) even if it temporarily decreases count.
     - Accept improvements always; accept decreases with probability exp(-delta/T).
  4. Track best solution found; return edge list.

This complement-removal + SA approach explores a fundamentally different search
landscape than pure constructive greedy: it starts dense and carves out forbidden
structures, then refines via probabilistic moves that can escape local optima.
"""
import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 142  # safety margin before 150 s limit
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_total = len(all_edges)  # 120

    # ---------- Q3 detection: find a Q3 subgraph in a dense graph ----------
    # Q3 on 8 vertices, 12 edges. We enumerate candidate 8-vertex subsets
    # and check if they contain a Q3. For speed, we use the structural check.

    def has_q3_in_subgraph(adj, subset):
        """Check if the induced subgraph on 'subset' (list of 8 vertices) contains a Q3.
        adj: list of bitmasks."""
        # Map subset vertices to 0..7
        idx = {v: i for i, v in enumerate(subset)}
        # Check all 3-bit labelings: assign 0..7 to the 8 vertices
        # Q3 edges: pairs with Hamming distance 1
        # For a labeling f: vertices -> {0..7}, check all 12 Q3 edges exist
        from itertools import permutations
        # Too many permutations (40320). Instead, use a smarter check:
        # The Q3 is a specific 3-cube. We check if there's a bijection such that
        # all 12 Q3 edges are present.
        # For 8 vertices, we can check: the induced subgraph must have exactly
        # the right structure. A necessary condition: each vertex has degree 3
        # within the subgraph, and the subgraph has exactly 12 edges.
        # Let's check degrees first as a filter.
        sub_adj = [0] * 8
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j and (adj[u] & (1 << v)):
                    sub_adj[i] |= (1 << j)
        # Check all degrees are 3
        for i in range(8):
            if bin(sub_adj[i]).count('1') != 3:
                return False
        # Now check if it's actually a Q3 (not e.g. 3-regular but different)
        # Q3: pick vertex 0, it has 3 neighbors. Those 3 neighbors have no edges
        # between them (in Q3, neighbors of 000 are 001,010,100 — pairwise non-adjacent).
        # Actually in Q3: 001-010? No. 001-100? No. 010-100? No. Good.
        # So: neighbors of any vertex form an independent set.
        for i in range(8):
            nb = sub_adj[i]
            # Check no two neighbors of i are adjacent
            nb_list = [j for j in range(8) if (nb >> j) & 1]
            for a in range(len(nb_list)):
                for b in range(a + 1, len(nb_list)):
                    if (sub_adj[nb_list[a]] >> nb_list[b]) & 1:
                        return False
        # Additionally check: for each vertex, the neighbors of its neighbors
        # (excluding itself and direct neighbors) form the "opposite" structure.
        # Simpler: check that the complement of Q3 edges in K8 is also consistent.
        # Q3 has 12 edges, K8 has 28, so 16 non-edges.
        edge_count = sum(bin(sub_adj[i]).count('1') for i in range(8)) // 2
        if edge_count != 12:
            return False
        # Check bipartite-like structure: Q3 is bipartite (even/odd weight)
        # Actually Q3 is bipartite. Check if subgraph is bipartite.
        color = [-1] * 8
        from collections import deque
        for start in range(8):
            if color[start] == -1:
                color[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in range(8):
                        if (sub_adj[u] >> v) & 1:
                            if color[v] == -1:
                                color[v] = 1 - color[u]
                                q.append(v)
                            elif color[v] == color[u]:
                                return False
        # If bipartite with 12 edges and all degrees 3 and neighbors independent, it's Q3
        return True

    def find_q3(adj):
        """Find a Q3 subgraph. Returns list of 8 vertices or None.
        Uses a heuristic: sample random 8-subsets and check."""
        from itertools import combinations
        # Precompute all 8-subsets? C(16,8) = 12870. That's feasible.
        # But checking each is expensive. Let's use a smarter approach:
        # Pick a vertex, look at its 3-clique-free neighborhood structure.
        # Alternative: for speed, randomly sample subsets.
        rng = random.Random()
        for _ in range(200):
            subset = rng.sample(range(n), 8)
            if has_q3_in_subgraph(adj, subset):
                return subset
        # Fallback: systematic (expensive)
        # For now, return None if not found in random samples
        return None

    # ---------- Faster Q3 check for edge addition ----------
    def creates_q3(adj, u, v):
        """Check if adding edge (u,v) creates a Q3.
        adj[i] bitmask of neighbors, NOT including (u,v) yet."""
        if adj[u] & (1 << v):
            return False  # already present
        neigh_u = adj[u]
        neigh_v = adj[v]

        # In Q3 with edge u-v, u has 2 other Q3-neighbors p,q (independent),
        # v has 2 other Q3-neighbors r,s (independent).
        # Structure: u=000, v=001, p=010, q=100, r=011, s=101, t=110, w=111
        # Edges: u-v, u-p, u-q, v-r, v-s, p-r, q-s, p-t, q-t, r-w, s-w, t-w
        # (one of two pairings: p-r & q-s, or p-s & q-r)

        p_bits = neigh_u
        while p_bits:
            p_bit = p_bits & (-p_bits)
            p_bits -= p_bit
            p_idx = p_bit.bit_length() - 1
            q_bits = neigh_u & ~(1 << p_idx)
            while q_bits:
                q_bit = q_bits & (-q_bits)
                q_bits -= q_bit
                q_idx = q_bit.bit_length() - 1

                # Check p and q are not adjacent (they should be independent in Q3)
                if adj[p_idx] & (1 << q_idx):
                    continue

                # Pairing 1: p-r, q-s
                r_cand = neigh_v & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand = neigh_v & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                if r_cand and s_cand:
                    r_bits = r_cand
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        s_cand2 = s_cand & ~(1 << r_idx)
                        if not s_cand2:
                            continue
                        # Check r,s independent
                        if adj[r_idx] & s_cand2:
                            continue
                        s_bits = s_cand2
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return True

                # Pairing 2: p-s, q-r
                r_cand2 = neigh_v & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand2 = neigh_v & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                if r_cand2 and s_cand2:
                    r_bits = r_cand2
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        s_cand3 = s_cand2 & ~(1 << r_idx)
                        if not s_cand3:
                            continue
                        if adj[r_idx] & s_cand3:
                            continue
                        s_bits = s_cand3
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return True
        return False

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += bin(adj[u]).count('1')
        return c // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def make_full_adj():
        adj = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v:
                    adj[u] |= (1 << v)
        return adj

    # ---------- Phase 1: Complement removal to get Q3-free ----------
    def remove_to_q3_free(adj, rng):
        """Remove edges until no Q3 exists."""
        iters = 0
        while time.time() < deadline - 50:
            q3 = find_q3(adj)
            if q3 is None:
                break
            # Remove a random edge from the Q3
            q3_edges = []
            for i in range(8):
                for j in range(i + 1, 8):
                    u, v = q3[i], q3[j]
                    if adj[u] & (1 << v):
                        q3_edges.append((u, v))
            if not q3_edges:
                break
            u, v = rng.choice(q3_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            iters += 1
        return adj, iters

    # ---------- Phase 2: Simulated annealing local search ----------
    def simulate_annealing(adj, rng, time_budget):
        """SA local search on a Q3-free graph."""
        best_adj = adj[:]
        best_count = count_edges(adj)
        cur_adj = adj[:]
        cur_count = count_edges(cur_adj)

        # Initial temperature
        T = 2.0
        T_min = 0.01
        cooling = 0.995

        missing = [(u, v) for u, v in all_edges if not (cur_adj[u] & (1 << v))]
        existing = [(u, v) for u, v in all_edges if cur_adj[u] & (1 << v)]

        step = 0
        while time.time() < time_budget:
            step += 1
            if step % 500 == 0:
                T = max(T * cooling, T_min)

            # Choose a random move
            move_type = rng.random()
            if move_type < 0.5 and missing:
                # Add a missing edge
                u, v = rng.choice(missing)
                if creates_q3(cur_adj, u, v):
                    continue
                cur_adj[u] |= (1 << v)
                cur_adj[v] |= (1 << u)
                delta = 1
                missing.remove((u, v))
                existing.append((u, v))
            elif move_type < 0.8 and existing:
                # Remove an existing edge
                u, v = rng.choice(existing)
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                delta = -1
                existing.remove((u, v))
                missing.append((u, v))
            elif existing and missing:
                # Swap: remove one existing, add one missing
                u1, v1 = rng.choice(existing)
                u2, v2 = rng.choice(missing)
                # Remove (u1,v1)
                cur_adj[u1] &= ~(1 << v1)
                cur_adj[v1] &= ~(1 << u1)
                # Try add (u2,v2)
                if not creates_q3(cur_adj, u2, v2):
                    cur_adj[u2] |= (1 << v2)
                    cur_adj[v2] |= (1 << u2)
                    delta = 0
                    existing.remove((u1, v1))
                    missing.remove((u2, v2))
                    existing.append((u2, v2))
                    missing.append((u1, v1))
                else:
                    # Revert removal
                    cur_adj[u1] |= (1 << v1)
                    cur_adj[v1] |= (1 << u1)
                    continue
            else:
                continue

            cur_count += delta
            new_count = cur_count

            if new_count > best_count:
                best_count = new_count
                best_adj = cur_adj[:]
            elif rng.random() > math.exp(-delta / max(T, 0.001)):
                # Reject: revert
                if delta == 1:
                    cur_adj[u] &= ~(1 << v)
                    cur_adj[v] &= ~(1 << u)
                    missing.append((u, v))
                    existing.remove((u, v))
                    cur_count -= 1
                elif delta == -1:
                    cur_adj[u1] |= (1 << v1)
                    cur_adj[v1] |= (1 << u1)
                    missing.remove((u1, v1))
                    existing.append((u1, v1))
                    cur_count += 1
                # delta == 0 swap: revert both
                else:
                    cur_adj[u1] &= ~(1 << v1)
                    cur_adj[v1] &= ~(1 << u1)
                    cur_adj[u2] &= ~(1 << v2)
                    cur_adj[v2] &= ~(1 << u2)
                    missing.append((u1, v1))
                    missing.remove((u2, v2))
                    existing.append((u1, v1))
                    existing.remove((u2, v2))

        return best_adj, best_count

    # ---------- Main ----------
    rng = random.Random(42)
    best_adj = None
    best_count = 0

    # Multiple restarts with different removal strategies
    for restart in range(15):
        if time.time() > deadline - 60:
            break
        seed_val = 100 + restart * 7
        rng = random.Random(seed_val)
        adj = make_full_adj()

        # Phase 1: remove to Q3-free
        adj, _ = remove_to_q3_free(adj, rng)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

        # Phase 2: SA refinement
        if time.time() < deadline - 30:
            sa_time = min(deadline - 30, time.time() + 8)
            adj2, cnt2 = simulate_annealing(adj, rng, sa_time)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2[:]

    # Final greedy add on best
    if best_adj is not None and time.time() < deadline - 5:
        missing = [(u, v) for u, v in all_edges if not (best_adj[u] & (1 << v))]
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if not creates_q3(best_adj, u, v):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
        best_count = count_edges(best_adj)

    if best_adj is None:
        # Fallback: empty graph
        return []

    return edges_from_adj(best_adj)
