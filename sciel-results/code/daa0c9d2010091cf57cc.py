import random
import time

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    start_time = time.time()
    timeout = 110  # Safety margin

    def adj_to_mask(adj):
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks[i] = m
        return masks

    def mask_to_adj(masks):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if masks[i] & (1 << j):
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def count_violations_mask(masks):
        # Returns list of (u1, u2, u3, common_mask) for each K3,3 violation
        violations = []
        for u1 in range(n):
            for u2 in range(u1 + 1, n):
                for u3 in range(u2 + 1, n):
                    common = masks[u1] & masks[u2] & masks[u3]
                    if bin(common).count('1') >= 3:
                        violations.append((u1, u2, u3, common))
        return violations

    def is_add_safe_mask(masks, u, v):
        # Check if adding edge (u,v) creates a K3,3
        # K3,3 exists if there are 3 vertices with 3 common neighbors
        # After adding (u,v), the only new K3,3s must involve the new edge
        # A K3,3 has bipartition {a1,a2,a3} and {b1,b2,b3}
        # If the new edge is in the K3,3, it connects one from each side
        # Case 1: u is in A-side, v is in B-side
        #   Then we need 2 more A-vertices (neighbors of v, not u) and 2 more B-vertices (neighbors of u, not v)
        #   such that all cross-edges exist
        # Actually simpler: just check all triples
        new_masks = masks[:]
        new_masks[u] |= (1 << v)
        new_masks[v] |= (1 << u)
        
        # Check if any triple now has >= 3 common neighbors
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    common = new_masks[a] & new_masks[b] & new_masks[c]
                    if bin(common).count('1') >= 3:
                        # Check if this violation involves the new edge
                        # It involves new edge if removing (u,v) fixes it
                        old_common = masks[a] & masks[b] & masks[c]
                        if bin(old_common).count('1') < 3:
                            return False
        return True

    def local_search(p):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)

        # Phase 1: Remove edges to fix K3,3
        while True:
            masks = adj_to_mask(adj)
            violations = count_violations_mask(masks)
            if not violations:
                break

            # Count edge frequencies in violations
            edge_counts = {}
            for u1, u2, u3, common_mask in violations:
                for c in range(n):
                    if common_mask & (1 << c):
                        for u in (u1, u2, u3):
                            e = (min(u, c), max(u, c))
                            edge_counts[e] = edge_counts.get(e, 0) + 1

            if not edge_counts:
                break

            max_count = -1
            best_edge = None
            for e, cnt in edge_counts.items():
                if cnt > max_count:
                    max_count = cnt
                    best_edge = e

            if best_edge:
                u, v = best_edge
                adj[u].discard(v)
                adj[v].discard(u)

        # Phase 2: Greedy add edges
        improved = True
        while improved:
            improved = False
            masks = adj_to_mask(adj)
            for i in range(n):
                for j in range(i + 1, n):
                    if j not in adj[i]:
                        if is_add_safe_mask(masks, i, j):
                            adj[i].add(j)
                            adj[j].add(i)
                            masks = adj_to_mask(adj)
                            improved = True

        # Phase 3: 2-opt swap local search
        # Try swapping one edge for another to improve or maintain edge count while fixing violations
        swap_improved = True
        while swap_improved:
            swap_improved = False
            masks = adj_to_mask(adj)
            current_count = sum(len(s) for s in adj) // 2
            
            # Try all possible edge swaps
            for i in range(n):
                for j in range(i + 1, n):
                    if j in adj[i]:  # existing edge to remove
                        for k in range(n):
                            for l in range(k + 1, n):
                                if l in adj[k]:  # existing edge to remove
                                    if (i, j) == (k, l) or (i, j) == (l, k) or (k, l) == (i, j) or (k, l) == (j, i):
                                        continue
                                    # Try removing (i,j) and (k,l), adding (i,k) and (j,l)
                                    # Or removing (i,j) and (k,l), adding (i,l) and (j,k)
                                    for new_edges in [((i, k), (j, l)), ((i, l), (j, k))]:
                                        ne1, ne2 = new_edges
                                        # Check if new edges are valid (not already present, not self-loops)
                                        if ne1[0] == ne1[1] or ne2[0] == ne2[1]:
                                            continue
                                        if ne2[0] in adj[ne2[1]] or ne1[0] in adj[ne1[1]]:
                                            continue
                                        
                                        # Apply swap
                                        adj[i].discard(j)
                                        adj[j].discard(i)
                                        adj[k].discard(l)
                                        adj[l].discard(k)
                                        adj[ne1[0]].add(ne1[1])
                                        adj[ne1[1]].add(ne1[0])
                                        adj[ne2[0]].add(ne2[1])
                                        adj[ne2[1]].add(ne2[0])
                                        
                                        new_masks = adj_to_mask(adj)
                                        violations = count_violations_mask(new_masks)
                                        new_count = sum(len(s) for s in adj) // 2
                                        
                                        if len(violations) == 0:
                                            if new_count > current_count:
                                                swap_improved = True
                                                # Keep the swap
                                                masks = new_masks
                                                break
                                            elif new_count == current_count and random.random() < 0.3:
                                                # Accept equal-count swap with some probability (simulated annealing-like)
                                                swap_improved = True
                                                masks = new_masks
                                                break
                                        
                                        # Revert
                                        adj[i].add(j)
                                        adj[j].add(i)
                                        adj[k].add(l)
                                        adj[l].add(k)
                                        adj[ne1[0]].discard(ne1[1])
                                        adj[ne1[1]].discard(ne1[0])
                                        adj[ne2[0]].discard(ne2[1])
                                        adj[ne2[1]].discard(ne2[0])
                                        
                                        if swap_improved:
                                            break
                                    if swap_improved:
                                        break
                                if swap_improved:
                                    break
                            if swap_improved:
                                break
                        if swap_improved:
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

        edge_count = sum(len(s) for s in adj) // 2
        return adj, edge_count

    # Try multiple restarts with different p
    for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for _ in range(10):
            if time.time() - start_time > timeout:
                break

            adj, edge_count = local_search(p)
            if edge_count > best_edges:
                best_edges = edge_count
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                best_graph = edges

        if time.time() - start_time > timeout:
            break

    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return best_graph
