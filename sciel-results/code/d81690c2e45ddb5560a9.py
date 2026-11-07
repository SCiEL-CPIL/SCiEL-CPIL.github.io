"""Cochromatic partition solver: model-and-optimize via branch-and-bound."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj


def construct():
    n, adj = _build_graph()
    deadline = time.time() + 85  # safety margin before 90s kill

    # Precompute for each vertex: neighbors and non-neighbors
    neighbors = [set(adj[i]) for i in range(n)]
    non_neighbors = [set(range(n)) - set(adj[i]) - {i} for i in range(n)]

    # Bitmask helpers
    full_mask = (1 << n) - 1

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for v in lst:
            m |= (1 << v)
        return m

    def is_clique_mask(mask):
        """Check if vertices in mask form a clique."""
        verts = mask_to_list(mask)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] not in neighbors[verts[i]]:
                    return False
        return True

    def is_independent_mask(mask):
        """Check if vertices in mask form an independent set."""
        verts = mask_to_list(mask)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] in neighbors[verts[i]]:
                    return False
        return True

    # Greedy upper bound
    def greedy_partition():
        parts = []  # list of (type, set_of_vertices)
        remaining = set(range(n))
        while remaining:
            # Try to extend an existing part
            placed = False
            for part_idx in range(len(parts)):
                ptype, pset = parts[part_idx]
                if ptype == 'clique':
                    for v in list(remaining):
                        if all(v in neighbors[u] for u in pset):
                            pset.add(v)
                            remaining.discard(v)
                            placed = True
                            break
                else:  # independent
                    for v in list(remaining):
                        if all(v not in neighbors[u] for u in pset):
                            pset.add(v)
                            remaining.discard(v)
                            placed = True
                            break
                if placed:
                    break
            if not placed:
                # Start a new part - try clique first with the vertex with highest degree
                v = max(remaining, key=lambda x: len(neighbors[x]))
                parts.append(('clique', {v}))
                remaining.discard(v)
        return [list(pset) for ptype, pset in parts]

    best_parts = greedy_partition()
    best_count = len(best_parts)

    # Lower bound: at least max(chromatic_number, independence_number_complement)
    # Use a simple lower bound: size of maximum clique in G or complement
    def max_clique_lower_bound():
        """Find a lower bound on parts needed via clique in G or clique in complement."""
        # Max clique in G gives lower bound on independent set packing
        # Max clique in complement gives lower bound on clique cover
        best_lb = 1
        
        # Greedy max clique in G
        def greedy_clique(adj_sets):
            clique = set()
            candidates = set(range(n))
            while candidates:
                # Pick vertex with max degree within candidates
                v = max(candidates, key=lambda x: len(adj_sets[x] & candidates))
                clique.add(v)
                candidates = candidates & adj_sets[v]
            return len(clique)
        
        lb1 = greedy_clique(neighbors)
        
        # Max clique in complement = max independent set in G
        comp_neighbors = [set(range(n)) - neighbors[i] - {i} for i in range(n)]
        lb2 = greedy_clique(comp_neighbors)
        
        return max(lb1, lb2)

    lb = max_clique_lower_bound()
    if lb >= best_count:
        # Can't do better
        result = [list(part) for part in best_parts]
        return result

    # Branch and bound search
    # State: (remaining_mask, parts as list of (type, mask))
    # We search for a valid partition with fewer parts than best_count

    def search(remaining_mask, parts, current_count):
        """DFS with branch and bound.
        Returns a valid partition if found, else None."""
        nonlocal best_parts, best_count

        if time.time() > deadline:
            return None

        if remaining_mask == 0:
            return [mask_to_list(pmask) for ptype, pmask in parts]

        if current_count >= best_count:
            return None

        # Lower bound pruning
        if current_count + lb > best_count:
            return None

        # Get remaining vertices sorted by degree (descending)
        rem_verts = mask_to_list(remaining_mask)
        rem_verts.sort(key=lambda v: len(neighbors[v]), reverse=True)

        for v in rem_verts:
            v_mask = 1 << v
            new_remaining = remaining_mask & ~v_mask

            # Option 1: add v to an existing clique part
            for i, (ptype, pmask) in enumerate(parts):
                if ptype == 'clique' and all(v in neighbors[u] for u in mask_to_list(pmask)):
                    new_pmask = pmask | v_mask
                    new_parts = list(parts)
                    new_parts[i] = ('clique', new_pmask)
                    result = search(new_remaining, new_parts, current_count)
                    if result is not None:
                        return result
                elif ptype == 'independent' and all(v not in neighbors[u] for u in mask_to_list(pmask)):
                    new_pmask = pmask | v_mask
                    new_parts = list(parts)
                    new_parts[i] = ('independent', new_pmask)
                    result = search(new_remaining, new_parts, current_count)
                    if result is not None:
                        return result

            # Option 2: start a new clique with v
            if current_count + 1 < best_count:
                new_parts = parts + [('clique', v_mask)]
                result = search(new_remaining, new_parts, current_count + 1)
                if result is not None:
                    return result

            # Option 3: start a new independent set with v
            if current_count + 1 < best_count:
                new_parts = parts + [('independent', v_mask)]
                result = search(new_remaining, new_parts, current_count + 1)
                if result is not None:
                    return result

        return None

    # Try to improve upon greedy
    target = best_count - 1
    if target >= 1:
        result = search(full_mask, [], 0)
        if result is not None:
            best_parts = result
            best_count = len(result)

        # Try even better
        target = best_count - 1
        if target >= 1:
            result = search(full_mask, [], 0)
            if result is not None:
                best_parts = result
                best_count = len(result)

    # Validate the result
    def validate(parts):
        seen = set()
        for part in parts:
            for v in part:
                if v in seen:
                    return False
                seen.add(v)
            if len(part) <= 1:
                continue
            is_cl = all(part[j] in neighbors[part[i]] for i in range(len(part)) for j in range(i+1, len(part)))
            is_ind = all(part[j] not in neighbors[part[i]] for i in range(len(part)) for j in range(i+1, len(part)))
            if not (is_cl or is_ind):
                return False
        if len(seen) != n:
            return False
        return True

    if not validate(best_parts):
        # Fallback: greedy
        best_parts = greedy_partition()

    # Ensure proper types
    result = [list(part) for part in best_parts]
    return result

# EVOLVE-BLOCK-END
