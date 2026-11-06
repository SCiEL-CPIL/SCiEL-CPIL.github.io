import random
import time

def construct():
    n = 20
    deadline = time.time() + 130

    def count_bits(x):
        return bin(x).count('1')

    def try_add_edge(adj_masks, u, v):
        """Try to add edge (u,v). Returns True if K3,3-free is preserved."""
        if (adj_masks[u] >> v) & 1:
            return True
        # Tentatively add
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        # Check all triples involving u and v
        for w in range(n):
            if w == u or w == v:
                continue
            if ((adj_masks[u] >> w) & 1) and ((adj_masks[v] >> w) & 1):
                if count_bits(adj_masks[u] & adj_masks[v] & adj_masks[w]) >= 3:
                    # Revert
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
                    return False
        return True

    def edge_count(adj_masks):
        return count_bits(sum(adj_masks)) // 2

    def build_edges(adj_masks):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if (adj_masks[i] >> j) & 1]

    random.seed(42)
    best_edges = None
    best_count = 0
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for restart in range(2000):
        if time.time() > deadline:
            break

        # Greedy construction with random edge ordering
        adj_masks = [0] * n
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for (u, v) in shuffled:
            if time.time() > deadline:
                break
            try_add_edge(adj_masks, u, v)

        cur_count = edge_count(adj_masks)

        # 2-swap local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for (ri, rj) in all_edges:
                if time.time() > deadline:
                    break
                if not ((adj_masks[ri] >> rj) & 1):
                    continue
                # Remove edge (ri, rj)
                adj_masks[ri] &= ~(1 << rj)
                adj_masks[rj] &= ~(1 << ri)
                for (ai, aj) in all_edges:
                    if time.time() > deadline:
                        break
                    if (ai, aj) == (ri, rj):
                        continue
                    if (adj_masks[ai] >> aj) & 1:
                        continue
                    if try_add_edge(adj_masks, ai, aj):
                        improved = True
                        break
                    # Restore removed edge
                    adj_masks[ri] |= (1 << rj)
                    adj_masks[rj] |= (1 << ri)
                if time.time() > deadline:
                    break
                if improved:
                    break
            if time.time() > deadline:
                break

        cur_count = edge_count(adj_masks)
        if cur_count > best_count:
            best_count = cur_count
            best_edges = build_edges(adj_masks)

        if best_count >= 46:
            break

    if best_edges is None:
        best_edges = [[a, b] for a in range(2) for b in range(2, 20)]

    return best_edges
