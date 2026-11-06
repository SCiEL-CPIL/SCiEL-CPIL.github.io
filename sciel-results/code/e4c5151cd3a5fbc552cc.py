"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 142  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks for fast intersection
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        # Track remaining edges via bitmask per vertex
        rem_mask = [0] * n
        for u, v in edges:
            rem_mask[u] |= (1 << v)
            rem_mask[v] |= (1 << u)

        remaining_count = len(edges)
        cliques = []

        def get_rem_neighbors(v):
            """Return list of vertices u such that (v,u) is in remaining."""
            mask = rem_mask[v]
            result = []
            while mask:
                lsb = mask & (-mask)
                u = lsb.bit_length() - 1
                result.append(u)
                mask ^= lsb
            return result

        def greedy_clique(start):
            """Build a clique from 'start' using only remaining edges."""
            clique = [start]
            # Candidates: neighbors of start via remaining edges
            candidates = set(get_rem_neighbors(start))

            while candidates:
                # Pick candidate with max connections to other candidates
                best_cand = None
                best_score = -1
                for cand in candidates:
                    score = 0
                    cand_neigh = rem_mask[cand]
                    for other in candidates:
                        if other != cand and (cand_neigh >> other) & 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_cand is None:
                    break

                clique.append(best_cand)
                candidates.discard(best_cand)
                # Intersect candidates with neighbors of best_cand
                bc_neigh = rem_mask[best_cand]
                candidates = {c for c in candidates if (bc_neigh >> c) & 1}

            return clique if len(clique) >= 2 else None

        def find_best_clique():
            """Try multiple starting vertices, return largest clique found."""
            best_clique = None
            best_size = 1

            # Try vertices in order of remaining degree
            degrees = []
            for v in range(n):
                deg = bin(rem_mask[v]).count('1')
                if deg >= 1:
                    degrees.append((deg, v))
            degrees.sort(reverse=True)

            for deg, v in degrees:
                if time.time() > deadline - 10:
                    break
                if deg < 1:
                    break
                clique = greedy_clique(v)
                if clique and len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
                    if best_size >= 10:  # good enough for dense graphs
                        break

            return best_clique

        while remaining_count > 0:
            if time.time() > deadline - 10:
                break

            clique = find_best_clique()

            if clique is None or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue

            # Remove all edges within clique from remaining
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    if (rem_mask[u] >> v) & 1:
                        rem_mask[u] &= ~(1 << v)
                        rem_mask[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue

            cliques.append(clique)
            remaining_count -= removed

        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = rem_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem_mask[u] &= ~(1 << v)
                    rem_mask[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
