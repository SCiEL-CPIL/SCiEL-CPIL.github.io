import time
import random
import numpy as np

# Q3 edges (canonical, vertices 0..7)
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with sorted unassigned vertices
# For each edge (qa,qb) in Q3, we orient it as (gu,gv) and order the other 6 vertices
# by how many of {gu,gv} they are adjacent to in Q3 (desc) to prune early.
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb


def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 120.0  # safety margin before 150s kill

    full_mask = (1 << n) - 1

    def has_q3_after_add(adj, u, v, deadline):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph.
        Uses bitmask candidate intersection for fast pruning."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                # Precompute for each unassigned vertex which assigned Q3-neighbors it has
                # and their graph-neighbors, to build initial candidate set
                def bt(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == 6:
                        # All 6 unassigned are assigned; verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    # Candidates: not used, adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]

                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit

                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    # Spectral heuristic: start with K16, iteratively remove edges
    # that maximize reduction in trace(A^4) (number of 4-cycles)
    def spectral_remove_edges(adj, deadline):
        """Remove edges from adj to reduce 4-cycles, guided by spectral heuristic."""
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    A[u, v] = 1.0
                    A[v, u] = 1.0

        # Compute A^2 and A^4
        A2 = A @ A
        A4 = A2 @ A2
        # trace(A^4) counts closed walks of length 4; each 4-cycle contributes 8
        # diag(A^4)[i] = sum_j A^2[i,j]^2 = sum of (common neighbors)^2 for each i
        # We want to remove edges that reduce trace(A^4) the most

        # For edge (u,v), the change in trace(A^4) when removing it:
        # Removing A[u,v]=A[v,u]=1 changes A2 by -1 at positions (u,v),(v,u),(u,u)+(v,v) via paths
        # More precisely, A2_new = A2 - (e_u e_v^T + e_v e_u^T + e_u e_u^T * 0 + ...)
        # Actually: (A^2)_{ij} = sum_k A_{ik}A_{kj}
        # Removing edge (u,v): A'_{uv}=A'_{vu}=0
        # (A'^2)_{ij} = (A^2)_{ij} - A_{iu}A_{vj} - A_{iv}A_{uj} for i,j not u,v
        # (A'^2)_{u,j} = (A^2)_{u,j} - A_{uu}A_{vj} - A_{uv}A_{uj} = (A^2)_{u,j} - A_{vj} (since A_uv=1, A_uu=0)
        # Wait, let me think more carefully.
        # (A^2)_{ij} = sum_k A_{ik} A_{kj}
        # When we remove edge (u,v), A' differs from A only at (u,v) and (v,u)
        # (A'^2)_{ij} = (A^2)_{ij} - A_{iu}A_{vj} - A_{iv}A_{uj}
        # For trace: tr(A^4) = sum_i (A^2)_{ii}^2 ... no, tr(A^4) = sum_{i,j} (A^2)_{ij}(A^2)_{ji} = sum_{i,j} (A^2)_{ij}^2 since symmetric
        # So tr(A^4) = ||A^2||_F^2
        # Change in ||A^2||_F^2 when removing (u,v):
        # Delta_A2 = - (e_u e_v^T + e_v e_u^T)  ... wait
        # (A'^2)_{ij} - (A^2)_{ij} = -A_{iu}A_{vj} - A_{iv}A_{uj}
        # This is: -(column_u of A) * (row_v of A) - (column_v of A) * (row_u of A)
        # = -A[:,u] A[v,:] - A[:,v] A[u,:]
        # Let B = A[:,u] A[v,:] + A[:,v] A[u,:]  (this is the change with sign)
        # New A2 = A2 - B
        # New tr(A^4) = ||A2 - B||_F^2 = ||A2||_F^2 - 2<A2,B> + ||B||_F^2
        # So reduction = 2<A2,B> - ||B||_F^2
        # = 2 * sum_{i,j} A2_{ij} * B_{ij} - sum_{i,j} B_{ij}^2

        # B_{ij} = A_{iu}A_{vj} + A_{iv}A_{uj}
        # <A2, B> = sum_{i,j} A2_{ij} (A_{iu}A_{vj} + A_{iv}A_{uj})
        # = sum_{i,j} A2_{ij} A_{iu} A_{vj} + sum_{i,j} A2_{ij} A_{iv} A_{uj}
        # By symmetry these two terms are equal, so <A2,B> = 2 * sum_{i,j} A2_{ij} A_{iu} A_{vj}
        # sum_{i,j} A2_{ij} A_{iu} A_{vj} = sum_i A_{iu} (sum_j A2_{ij} A_{vj})
        # sum_j A2_{ij} A_{vj} = (A2 A)_{iv} ... no, sum_j A2_{ij} A_{vj} = (A2 @ A^T)_{iv} = (A2 @ A)_{iv} since symmetric
        # = (A^3)_{iv}
        # So sum_i A_{iu} (A^3)_{iv} = sum_i A_{iu} A^3_{iv} = (A^T A^3)_{uv} = (A^4)_{uv} since A symmetric and A commutes with its powers
        # Wait: sum_i A_{iu} A^3_{iv} = sum_i A_{ui} A^3_{iv} = (A @ A^3)_{uv} = (A^4)_{uv}
        # So <A2, B> = 2 * A^4_{uv}
        # ||B||_F^2 = sum_{i,j} (A_{iu}A_{vj} + A_{iv}A_{uj})^2
        # = sum_{i,j} [A_{iu}^2 A_{vj}^2 + 2 A_{iu}A_{vj}A_{iv}A_{uj} + A_{iv}^2 A_{uj}^2]
        # = sum_{i,j} A_{iu}A_{vj} + 2 sum_{i,j} A_{iu}A_{iv}A_{uj}A_{vj} + sum_{i,j} A_{iv}A_{uj}
        # First term: (sum_i A_{iu})(sum_j A_{vj}) = deg(u)*deg(v)
        # Third term: same = deg(u)*deg(v)
        # Middle term: 2 * sum_i sum_j A_{iu}A_{iv}A_{uj}A_{vj}
        # = 2 * sum_i (sum_j A_{iu}A_{iv}A_{uj}A_{vj})
        # For fixed i: sum_j A_{iu}A_{iv}A_{uj}A_{vj} = A_{iu}A_{iv} * (sum_j A_{uj}A_{vj}) = A_{iu}A_{iv} * (A^2)_{uv}
        # So middle = 2 * (A^2)_{uv} * sum_i A_{iu}A_{iv} = 2 * (A^2)_{uv} * (number of common neighbors of u,v)
        # = 2 * (A^2)_{uv}^2  since (A^2)_{uv} = number of common neighbors of u and v
        # So ||B||_F^2 = 2*deg(u)*deg(v) + 2*(A^2)_{uv}^2
        # Reduction in tr(A^4) = 2*A^4_{uv} - 2*deg(u)*deg(v) - 2*(A^2)_{uv}^2

        # But wait, we need to be careful: the formula assumes the edge exists.
        # Let me just compute it numerically for simplicity.

        while time.time() < deadline - 30:
            A2 = A @ A
            A4 = A2 @ A2
            # For each existing edge, compute the reduction in tr(A^4) if removed
            best_edge = None
            best_reduction = 0
            for u in range(n):
                for v in range(u+1, n):
                    if A[u, v] < 0.5:
                        continue
                    # Compute reduction in tr(A^4) = ||A2||_F^2
                    # B = A[:,u] A[v,:] + A[:,v] A[u,:]
                    B = np.outer(A[:, u], A[v, :]) + np.outer(A[:, v], A[u, :])
                    reduction = 2.0 * np.sum(A2 * B) - np.sum(B * B)
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_edge = (u, v)

            if best_edge is None or best_reduction <= 0:
                break

            u, v = best_edge
            A[u, v] = 0
            A[v, u] = 0
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

        return adj

    # Start with K16
    adj = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Apply spectral removal
    adj = spectral_remove_edges(adj, deadline)

    # Now verify no Q3 and try to add back edges
    # First, check if current graph has Q3 (it shouldn't if we removed enough, but verify)
    # We'll do a full Q3 check by trying to add all missing edges and seeing which ones cause Q3

    best_edges = None
    best_count = count_edges(adj)
    best_edges = edge_list(adj)

    # Try to add back missing edges greedily
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))

    # Multiple random orderings
    for attempt in range(3):
        if time.time() > deadline - 15:
            break

        if attempt == 0:
            pass
        else:
            random.shuffle(missing)

        # Reset to current adj
        adj_copy = adj[:]

        for u, v in missing:
            if time.time() > deadline - 15:
                break
            if not (adj_copy[u] & (1 << v)):
                if not has_q3_after_add(adj_copy, u, v, deadline):
                    adj_copy[u] |= (1 << v)
                    adj_copy[v] |= (1 << u)

        cnt = count_edges(adj_copy)
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj_copy)
            adj = adj_copy

    # Local search: try removing one edge and adding others
    if time.time() < deadline - 20:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        edges_list = [(u, v) for u, v in best_edges]
        random.shuffle(edges_list)

        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for a, b in edges_list:
                if time.time() > deadline - 20:
                    break
                # Try removing (a,b) and see if we can add more
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)

                # Try to add all missing edges
                added = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if time.time() > deadline - 20:
                            break
                        if not (adj[i] & (1 << j)):
                            if not has_q3_after_add(adj, i, j, deadline):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                added += 1

                if added > 1:
                    improved = True
                    new_count = count_edges(adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_edges = edge_list(adj)
                    edges_list = [(u, v) for u, v in best_edges]
                    random.shuffle(edges_list)
                else:
                    # Restore
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    # Remove any edges we added - restore from best
                    adj = [0] * n
                    for u, v in best_edges:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    # Final polish
    adj = [0] * n
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    for i in range(n):
        for j in range(i+1, n):
            if time.time() > deadline - 5:
                break
            if not (adj[i] & (1 << j)):
                if not has_q3_after_add(adj, i, j, deadline):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    final_count = count_edges(adj)
    if final_count > best_count:
        best_count = final_count
        best_edges = edge_list(adj)

    return best_edges
