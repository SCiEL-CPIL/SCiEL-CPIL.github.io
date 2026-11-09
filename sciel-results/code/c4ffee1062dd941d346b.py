import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 115.0
    n = 16

    # Precompute all 8-vertex subsets
    all_8sets = list(combinations(range(n), 8))

    # Precompute the 35 bipartitions of 8 local indices into two sets of 4
    # We use these to test whether an 8-vertex induced subgraph contains a Q3.
    # A Q3 on a fixed 8-vertex set corresponds to a bipartition (A,B) of the 8 vertices
    # where the induced subgraph on A∪B has all cross-edges except a perfect matching.
    splits = []
    seen = set()
    for a in combinations(range(8), 4):
        b = tuple(i for i in range(8) if i not in a)
        key = frozenset([frozenset(a), frozenset(b)])
        if key not in seen:
            seen.add(key)
            splits.append((a, b))
    # splits has 35 entries; each is (tuple of 4 local indices, tuple of 4 local indices)

    # For each 8-set, precompute the local index mapping
    # local_idx[global_vertex] = position in the 8-tuple, or -1
    local_idx_cache = []
    for s in all_8sets:
        lm = {}
        for i, v in enumerate(s):
            lm[v] = i
        local_idx_cache.append(lm)

    # Precompute for each 8-set and each split, the list of (local_i, local_j) cross-pairs
    # and the required matching structure
    # Actually, we can just test directly.

    def has_q3_in_subset(adj_mat, subset, lm):
        """Check if the induced subgraph on 'subset' contains a Q3."""
        # Build local adjacency as bitmasks
        local_adj = [0] * 8
        for i in range(8):
            vi = subset[i]
            mask = 0
            for j in range(8):
                if i != j and adj_mat[vi][subset[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask

        # Count edges; need at least 12
        ec = sum(bin(m).count('1') for m in local_adj) // 2
        if ec < 12:
            return False

        for A, B in splits:
            # Check: all edges between A and B except a perfect matching are present
            # Count missing cross-edges
            missing = 0
            missing_list = []
            for i in A:
                for j in B:
                    if not (local_adj[i] & (1 << j)):
                        missing += 1
                        missing_list.append((i, j))
            if missing != 4:
                continue
            # Check if missing_list forms a perfect matching between A and B
            # Each vertex in A appears exactly once, each in B appears exactly once
            a_count = {}
            b_count = {}
            for i, j in missing_list:
                a_count[i] = a_count.get(i, 0) + 1
                b_count[j] = b_count.get(j, 0) + 1
            if all(c == 1 for c in a_count.values()) and all(c == 1 for c in b_count.values()):
                return True
        return False

    def is_q3_free(adj_mat):
        """Check if the graph is Q3-free."""
        for k, subset in enumerate(all_8sets):
            if has_q3_in_subset(adj_mat, subset, local_idx_cache[k]):
                return False
        return True

    def count_q3s(adj_mat):
        """Count number of Q3 subgraphs."""
        cnt = 0
        for k, subset in enumerate(all_8sets):
            if has_q3_in_subset(adj_mat, subset, local_idx_cache[k]):
                cnt += 1
        return cnt

    # Simulated Annealing
    # State: adjacency matrix (list of lists of 0/1)
    # Objective: maximize edges, penalize Q3s
    # Moves: flip a random edge

    def random_dense_seed():
        adj = [[0] * n for _ in range(n)]
        # Start with K_{3,13} + K3 as base
        for a in range(3):
            for b in range(3, n):
                adj[a][b] = adj[b][a] = 1
        adj[0][1] = adj[1][0] = 1
        adj[0][2] = adj[2][0] = 1
        adj[1][2] = adj[2][1] = 1
        # Randomly add some extra edges
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v] and random.random() < 0.1:
                    adj[u][v] = adj[v][u] = 1
        return adj

    def edge_count(adj):
        return sum(sum(row) for row in adj) // 2

    def objective(adj, q3_count):
        return edge_count(adj) - 100 * q3_count

    best_adj = None
    best_edges = 0
    best_q3 = 0

    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break

        adj = random_dense_seed()
        # Ensure Q3-free by removing edges that cause Q3s
        # Simple: while not Q3-free, remove a random edge involved in a Q3
        # This is slow; instead, start Q3-free
        # K_{3,13}+K3 is Q3-free, so let's verify and fix
        while not is_q3_free(adj):
            # Remove a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and adj[u][v]:
                adj[u][v] = adj[v][u] = 0
            if edge_count(adj) < 20:
                break

        if best_adj is None or edge_count(adj) > best_edges:
            best_adj = [row[:] for row in adj]
            best_edges = edge_count(adj)
            best_q3 = 0

        # SA
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        cur_edges = edge_count(adj)
        cur_q3 = count_q3s(adj)
        cur_obj = objective(adj, cur_q3)

        for step in range(50000):
            if time.time() > deadline - 5:
                break

            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            # Delta: only need to check Q3s containing this edge
            # But for simplicity, recompute
            old_val = adj[u][v]
            adj[u][v] = adj[v][u] = 1 - old_val
            new_edges = cur_edges + (1 if new_val == 1 else -1)
            new_q3 = count_q3s(adj)
            new_obj = objective(adj, new_q3)

            delta = new_obj - cur_obj
            if delta > 0 or random.random() < min(1.0, delta / T if T > 0 else 0):
                cur_edges = new_edges
                cur_q3 = new_q3
                cur_obj = new_obj
                if cur_q3 == 0 and cur_edges > best_edges:
                    best_adj = [row[:] for row in adj]
                    best_edges = cur_edges
                    best_q3 = 0
            else:
                adj[u][v] = adj[v][u] = old_val

            T *= cooling
            if T < T_min:
                T = T_min

        # Local polish: try adding edges greedily
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    adj[u][v] = adj[v][u] = 1
                    if is_q3_free(adj):
                        if edge_count(adj) > best_edges:
                            best_adj = [row[:] for row in adj]
                            best_edges = edge_count(adj)
                    else:
                        adj[u][v] = adj[v][u] = 0

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])

    return edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
