"""Clique partition of two fixed dense random graphs via greedy set-cover with local improvement."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    TOTAL_BUDGET = 145.0  # safety margin from 150s kill

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Build adjacency sets
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        uncovered = set(edge_set)
        if not uncovered:
            return []

        cliques = []
        deadline = t0 + time_budget

        while uncovered:
            if time.time() > deadline:
                break

            # Strategy: prioritize edges with HIGH "clique density"
            # For each uncovered edge (u,v), compute the number of common neighbors w
            # such that (u,w) and (v,w) are both in uncovered.
            # This identifies edges that can be part of large cliques.
            
            best_edge = None
            best_score = -1

            for e in uncovered:
                u, v = e
                # Count common neighbors in the remaining (uncovered) graph
                common = 0
                # Intersect adjacency lists for speed
                for w in adj[u]:
                    if w == v:
                        continue
                    if w in adj[v]:
                        e1 = (u, w) if u < w else (w, u)
                        e2 = (v, w) if v < w else (w, v)
                        if e1 in uncovered and e2 in uncovered:
                            common += 1
                # Use common count as score (higher is better for larger cliques)
                # We can also add a small bonus for lower degree to handle ties
                score = common
                if score > best_score:
                    best_score = score
                    best_edge = e

            if best_edge is None:
                break

            u, v = best_edge

            # Build the largest clique containing (u,v) using only uncovered edges
            clique = [u, v]
            clique_set = {u, v}
            
            # Candidates: vertices adjacent to both u and v in the uncovered graph
            candidates = set()
            for w in adj[u]:
                if w == v:
                    continue
                if w in adj[v]:
                    e1 = (u, w) if u < w else (w, u)
                    e2 = (v, w) if v < w else (w, v)
                    if e1 in uncovered and e2 in uncovered:
                        candidates.add(w)

            # Greedily add candidates to form a maximal clique
            # Sort candidates by their "connectivity" to other candidates
            # to build a dense core first
            changed = True
            while changed:
                changed = False
                if not candidates:
                    break
                
                # Find the candidate with the most connections to other candidates
                # that are also adjacent to all current clique members
                best_cand = None
                best_conn = -1
                
                for w in candidates:
                    # Check if w is adjacent to all in clique
                    can_add = True
                    for x in clique_set:
                        if w not in adj[x]:
                            can_add = False
                            break
                        e = (w, x) if w < x else (x, w)
                        if e not in uncovered:
                            can_add = False
                            break
                    if not can_add:
                        continue
                    
                    # Count connections to other candidates
                    conn = 0
                    for y in candidates:
                        if y == w:
                            continue
                        if w in adj[y]:
                            e = (w, y) if w < y else (y, w)
                            if e in uncovered:
                                conn += 1
                    if conn > best_conn:
                        best_conn = conn
                        best_cand = w
                
                if best_cand is not None:
                    clique.append(best_cand)
                    clique_set.add(best_cand)
                    candidates.discard(best_cand)
                    # Prune candidates that are no longer adjacent to the new member
                    to_remove = []
                    for w in candidates:
                        if best_cand not in adj[w]:
                            to_remove.append(w)
                        else:
                            e = (w, best_cand) if w < best_cand else (best_cand, w)
                            if e not in uncovered:
                                to_remove.append(w)
                    for w in to_remove:
                        candidates.discard(w)
                    changed = True

            # Remove all edges within the clique from uncovered
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (a, b) if a < b else (b, a)
                    uncovered.discard(e)

            cliques.append(clique)

        # If we timed out, handle remaining uncovered edges
        if uncovered:
            for e in uncovered:
                cliques.append(list(e))

        return cliques

    def merge_cliques(cliques, n, adj):
        """Try to merge cliques that are fully connected to each other."""
        if not cliques:
            return cliques
        
        improved = True
        while improved:
            if time.time() > t0 + TOTAL_BUDGET - 5:
                break
            improved = False
            new_cliques = []
            merged = False
            for i in range(len(cliques)):
                current = set(cliques[i])
                for j in range(i + 1, len(cliques)):
                    if j - i > 0 and j < len(cliques) and i < len(cliques) and j < len(cliques):
                        other = set(cliques[j])
                        # Check if all vertices in current connect to all in other
                        can_merge = True
                        for a in current:
                            for b in other:
                                if b not in adj[a]:
                                    can_merge = False
                                    break
                            if not can_merge:
                                break
                        if can_merge:
                            current.update(other)
                            # Mark for removal
                            if j < len(cliques):
                                cliques[j] = []
                            merged = True
                if current:
                    new_cliques.append(list(current))
            if merged:
                cliques = [c for c in new_cliques if c]
                improved = True
            else:
                # No merges found, break
                break
        return cliques

    # Allocate time: give more to the larger instance (n=64)
    # n=32 instance is faster, n=64 is slower
    time1 = 40.0
    time2 = 90.0

    result = []
    
    # Instance 1: (32, 0.6, 601)
    if time.time() < t0 + TOTAL_BUDGET - 5:
        cliques1 = solve_instance(32, 0.6, 601, time1)
        # Local improvement: merge cliques
        edges1 = random_graph(32, 0.6, 601)
        adj1 = [set() for _ in range(32)]
        for u, v in edges1:
            adj1[u].add(v)
            adj1[v].add(u)
        cliques1 = merge_cliques(cliques1, 32, adj1)
        result.append(cliques1)
    else:
        edges1 = random_graph(32, 0.6, 601)
        result.append([list(e) for e in edges1])

    # Instance 2: (64, 0.55, 602)
    if time.time() < t0 + TOTAL_BUDGET - 5:
        cliques2 = solve_instance(64, 0.55, 602, time2)
        edges2 = random_graph(64, 0.55, 602)
        adj2 = [set() for _ in range(64)]
        for u, v in edges2:
            adj2[u].add(v)
            adj2[v].add(u)
        cliques2 = merge_cliques(cliques2, 64, adj2)
        result.append(cliques2)
    else:
        edges2 = random_graph(64, 0.55, 602)
        result.append([list(e) for e in edges2])

    return result
# EVOLVE-BLOCK-END
