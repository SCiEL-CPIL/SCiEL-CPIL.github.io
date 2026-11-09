# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find the largest subset of Z/205Z with no two elements differing by a
    nonzero square mod 205.

    Uses exact maximum independent set via branch-and-bound with greedy
    coloring bound (Tomita algorithm) on the conflict graph, with a time
    deadline to return the best partial solution.
    """
    m = 205
    deadline = time.time() + 590  # safety margin before 630s

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = forbidden

    # Build conflict graph: adj[v] = list of vertices conflicting with v
    adj = [[] for _ in range(m)]
    for v in range(m):
        for u in range(m):
            if u == v:
                continue
            d = (v - u) % m
            if d in forbidden_set:
                adj[v].append(u)

    # Degree of each vertex
    degree = [len(adj[v]) for v in range(m)]

    # Greedy coloring bound: color the subgraph induced by candidates
    # Returns (color_count, color_of_vertex_dict)
    def greedy_color(candidates):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns (num_colors, color_assignment_dict).
        Uses largest-first ordering for a tighter bound.
        """
        if not candidates:
            return 0, {}

        # Order by degree in the induced subgraph, descending
        cand_set = set(candidates)
        ordered = sorted(candidates, key=lambda v: -sum(1 for u in adj[v] if u in cand_set))

        color = {}
        color_classes = []  # list of sets

        for v in ordered:
            # Find smallest color not used by neighbors
            used = set()
            for u in adj[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)

        return len(color_classes), color

    # Branch-and-bound for maximum independent set
    # We use the "include/exclude" branching with coloring bound.
    best_size = [0]
    best_set = [set()]
    time_up = [False]

    def branch_and_bound(candidates, current_set, current_size):
        """
        Recursive branch-and-bound.
        candidates: list of remaining candidate vertices
        current_set: set of vertices in the current independent set
        current_size: len(current_set)
        """
        if time_up[0]:
            return
        if time.time() > deadline:
            time_up[0] = True
            return

        if not candidates:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return

        # Upper bound via greedy coloring
        num_colors, _ = greedy_color(candidates)
        if current_size + num_colors <= best_size[0]:
            return

        # Choose a vertex to branch on: the one with highest degree in candidates
        cand_set = set(candidates)
        # Pick vertex with max degree in the induced subgraph
        v = max(candidates, key=lambda x: sum(1 for u in adj[x] if u in cand_set))

        # Branch 1: include v
        new_set = current_set | {v}
        # Remove v and all its neighbors from candidates
        neighbors = set(adj[v])
        new_candidates = [u for u in candidates if u != v and u not in neighbors]
        branch_and_bound(new_candidates, new_set, current_size + 1)

        if time_up[0]:
            return

        # Branch 2: exclude v
        new_candidates2 = [u for u in candidates if u != v]
        branch_and_bound(new_candidates2, current_set, current_size)

    # Initialize with a greedy solution
    greedy_order = sorted(range(m), key=lambda v: -degree[v])
    initial_set = set()
    for v in greedy_order:
        if all((v - u) % m not in forbidden_set and (u - v) % m not in forbidden_set for u in initial_set):
            initial_set.add(v)
    best_size[0] = len(initial_set)
    best_set[0] = set(initial_set)

    # Run branch-and-bound starting from all vertices
    # Sort initial candidates by degree descending for better pruning
    all_vertices = sorted(range(m), key=lambda v: -degree[v])
    branch_and_bound(all_vertices, set(), 0)

    # If we timed out, best_set[0] holds the best found so far
    # Convert to sorted array
    result = sorted(best_set[0])

    # Safety: verify the result is valid
    if not result:
        # Fallback: return a small valid set
        result = [0]

    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
