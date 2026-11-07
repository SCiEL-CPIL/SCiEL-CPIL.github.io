import time
import random

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj_c = [0] * k
        adj_i = [0] * (n - k)
        
        for (u, v) in edges:
            if u < k and v < k:
                adj_c[u] |= (1 << v)
                adj_c[v] |= (1 << u)
            elif u < k and v >= k:
                idx_v = v - k
                adj_i[idx_v] |= (1 << u)
            elif u >= k and v < k:
                idx_v = u - k
                adj_i[idx_v] |= (1 << v)

        uncovered_adj = [0] * k
        for u in range(k):
            uncovered_adj[u] = adj_c[u]

        cliques = []

        def is_clique_in_graph(mask, adj_list):
            temp = mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                temp ^= lsb
                if (adj_list[u] & (mask ^ lsb)) != (mask ^ lsb):
                    return False
            return True

        def find_max_clique(candidates, adj_list):
            verts = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
            
            if not verts:
                return 0

            verts.sort(key=lambda v: (adj_list[v] & candidates).bit_count(), reverse=True)
            
            clique_mask = 0
            cand_mask = candidates
            
            while cand_mask:
                best_v = -1
                best_deg = -1
                temp = cand_mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = (adj_list[v] & cand_mask).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                clique_mask |= (1 << best_v)
                cand_mask = cand_mask & adj_list[best_v]
                cand_mask &= ~(1 << best_v)

            return clique_mask

        def mask_to_list(mask):
            result = []
            temp = mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                result.append(u)
                temp ^= lsb
            return result

        i_indices = list(range(n - k))
        i_indices.sort(key=lambda idx: adj_i[idx].bit_count(), reverse=True)

        for idx in i_indices:
            if _deadline_hit():
                temp = adj_i[idx]
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    cliques.append([k + idx, u])
                    temp ^= lsb
                break
            
            remaining_neighbors = adj_i[idx]
            
            while remaining_neighbors:
                if _deadline_hit():
                    temp = remaining_neighbors
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        cliques.append([k + idx, u])
                        temp ^= lsb
                    break

                S_mask = find_max_clique(remaining_neighbors, uncovered_adj)
                
                if S_mask == 0:
                    lsb = remaining_neighbors & -remaining_neighbors
                    u = lsb.bit_length() - 1
                    S_mask = lsb

                clique = [k + idx]
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    clique.append(u)
                    temp ^= lsb
                
                cliques.append(clique)

                verts_in_S = mask_to_list(S_mask)
                for i in range(len(verts_in_S)):
                    u = verts_in_S[i]
                    for j in range(i + 1, len(verts_in_S)):
                        w = verts_in_S[j]
                        uncovered_adj[u] &= ~(1 << w)
                        uncovered_adj[w] &= ~(1 << u)

                remaining_neighbors &= ~S_mask

        full_mask = (1 << k) - 1
        while True:
            if _deadline_hit():
                for u in range(k):
                    for w in range(u + 1, k):
                        if (uncovered_adj[u] >> w) & 1:
                            cliques.append([u, w])
                            uncovered_adj[u] &= ~(1 << w)
                            uncovered_adj[w] &= ~(1 << u)
                break
            
            has_edge = False
            for u in range(k):
                if uncovered_adj[u]:
                    has_edge = True
                    break
            
            if not has_edge:
                break
            
            S_mask = find_max_clique(full_mask, uncovered_adj)
            
            if S_mask.bit_count() < 2:
                u = 0
                while u < k and not uncovered_adj[u]:
                    u += 1
                if u == k: break
                lsb = uncovered_adj[u] & -uncovered_adj[u]
                w = lsb.bit_length() - 1
                S_mask = (1 << u) | (1 << w)
            
            clique = mask_to_list(S_mask)
            cliques.append(clique)
            
            verts_in_S = mask_to_list(S_mask)
            for i in range(len(verts_in_S)):
                u = verts_in_S[i]
                for j in range(i + 1, len(verts_in_S)):
                    w = verts_in_S[j]
                    uncovered_adj[u] &= ~(1 << w)
                    uncovered_adj[w] &= ~(1 << u)

        # Local improvement: remove redundant cliques
        def local_improve(cliques, adj_c, adj_i, k, n):
            if not cliques:
                return cliques
            
            # Build edge coverage: for each edge, which cliques cover it
            # Represent edges as frozensets
            edge_to_cliques = {}
            for i, c in enumerate(cliques):
                for a in range(len(c)):
                    for b in range(a+1, len(c)):
                        u, v = c[a], c[b]
                        edge = frozenset((u, v))
                        if edge not in edge_to_cliques:
                            edge_to_cliques[edge] = []
                        edge_to_cliques[edge].append(i)
            
            # Try to remove cliques whose edges are all covered by others
            improved = True
            while improved and not _deadline_hit():
                improved = False
                new_cliques = []
                for i, c in enumerate(cliques):
                    if _deadline_hit():
                        new_cliques.extend(cliques[i:])
                        break
                    
                    # Check if all edges of this clique are covered by other cliques
                    redundant = True
                    for a in range(len(c)):
                        for b in range(a+1, len(c)):
                            u, v = c[a], c[b]
                            edge = frozenset((u, v))
                            covering = edge_to_cliques.get(edge, [])
                            if all(j == i for j in covering):
                                # Only this clique covers this edge
                                redundant = False
                                break
                        if not redundant:
                            break
                    
                    if redundant:
                        improved = True
                        # Remove this clique from edge_to_cliques
                        for a in range(len(c)):
                            for b in range(a+1, len(c)):
                                u, v = c[a], c[b]
                                edge = frozenset((u, v))
                                if i in edge_to_cliques[edge]:
                                    edge_to_cliques[edge].remove(i)
                    else:
                        new_cliques.append(c)
                
                cliques = new_cliques
                # Rebuild edge_to_cliques
                edge_to_cliques = {}
                for i, c in enumerate(cliques):
                    for a in range(len(c)):
                        for b in range(a+1, len(c)):
                            u, v = c[a], c[b]
                            edge = frozenset((u, v))
                            if edge not in edge_to_cliques:
                                edge_to_cliques[edge] = []
                            edge_to_cliques[edge].append(i)
            
            return cliques

        cliques = local_improve(cliques, adj_c, adj_i, k, n)

        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
