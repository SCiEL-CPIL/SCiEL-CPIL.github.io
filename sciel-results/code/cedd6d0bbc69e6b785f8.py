"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose_graph(n, edges, time_limit):
        """
        Find an edge-disjoint path decomposition covering all edges.
        Uses greedy extraction with multiple restarts and post-processing.
        """
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        edge_set = set(edges)
        remaining_edges = set(edges)

        def extract_path_greedy(start, remaining, max_len=None):
            """Extract a simple path starting from start, using only edges in remaining."""
            if not remaining:
                return None

            path = [start]
            visited_nodes = {start}
            cur = start
            used_in_path = set()

            while True:
                if max_len is not None and len(path) >= max_len:
                    break

                candidates = []
                for nb in adj[cur]:
                    e = (min(cur, nb), max(cur, nb))
                    if e in remaining and e not in used_in_path:
                        candidates.append(nb)

                if not candidates:
                    break

                # Prefer unvisited nodes
                unvisited = [nb for nb in candidates if nb not in visited_nodes]
                if unvisited:
                    # Pick the one with highest degree among unvisited
                    nb = max(unvisited, key=lambda x: len(adj[x]))
                else:
                    # All neighbors visited; pick one that doesn't create issues
                    # Since we track edges, we can't reuse edges, so any unused edge is fine
                    # But we must avoid revisiting nodes for simplicity
                    break

                e = (min(cur, nb), max(cur, nb))
                used_in_path.add(e)
                path.append(nb)
                visited_nodes.add(nb)
                cur = nb

            if len(path) < 2:
                return None

            return path

        def extract_path_dfs(start, remaining, depth_limit=15):
            """Extract a simple path using DFS with depth limit."""
            if not remaining:
                return None

            best_path = None
            stack = [(start, [start], {start}, set())]

            while stack:
                cur, path, visited, used_edges = stack.pop()

                if len(path) > (len(best_path) if best_path else 0):
                    best_path = list(path)

                if len(path) >= depth_limit:
                    continue

                candidates = []
                for nb in adj[cur]:
                    e = (min(cur, nb), max(cur, nb))
                    if e in remaining and e not in used_edges and nb not in visited:
                        candidates.append((nb, e))

                # Sort by degree to explore promising branches first
                candidates.sort(key=lambda x: len(adj[x[0]]), reverse=True)

                for nb, e in candidates:
                    new_visited = visited | {nb}
                    new_used = used_edges | {e}
                    new_path = path + [nb]
                    stack.append((nb, new_path, new_visited, new_used))

            return best_path

        def try_decompose(remaining, rng, time_budget):
            """Try to decompose remaining edges into paths."""
            paths = []
            rem = set(remaining)

            while rem:
                # Pick an edge to start from
                e = rng.choice(list(rem))
                u, v = e

                best_path = None
                best_path_edges = None

                # Try extracting from both endpoints
                for start in [u, v]:
                    # Try greedy extraction
                    p = extract_path_greedy(start, rem)
                    if p:
                        path_edges = set()
                        valid = True
                        for i in range(len(p) - 1):
                            pe = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                            if pe not in rem:
                                valid = False
                                break
                            path_edges.add(pe)
                        if valid and (best_path is None or len(p) > len(best_path)):
                            best_path = p
                            best_path_edges = path_edges

                    # Try DFS extraction
                    p = extract_path_dfs(start, rem, depth_limit=min(12, len(rem) + 1))
                    if p:
                        path_edges = set()
                        valid = True
                        for i in range(len(p) - 1):
                            pe = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                            if pe not in rem:
                                valid = False
                                break
                            path_edges.add(pe)
                        if valid and (best_path is None or len(p) > len(best_path)):
                            best_path = p
                            best_path_edges = path_edges

                if best_path is None:
                    # Fallback: just use the single edge
                    best_path = [u, v]
                    best_path_edges = {e}
                else:
                    # Recompute edges to be safe
                    best_path_edges = set()
                    for i in range(len(best_path) - 1):
                        pe = (min(best_path[i], best_path[i+1]), max(best_path[i], best_path[i+1]))
                        best_path_edges.add(pe)

                rem -= best_path_edges
                paths.append(best_path)

            return paths

        def merge_paths(paths, remaining_edges):
            """Try to merge adjacent paths to reduce total count."""
            # Check if two paths can be merged (if end of one connects to start of another)
            merged = True
            new_paths = list(paths)

            while merged:
                merged = False
                for i in range(len(new_paths)):
                    for j in range(i + 1, len(new_paths)):
                        p1 = new_paths[i]
                        p2 = new_paths[j]

                        # Check if we can merge p1 and p2
                        # p1 ends at p1[-1], p2 starts at p2[0]
                        # If edge (p1[-1], p2[0]) exists and is in remaining, we can merge
                        e1 = (min(p1[-1], p2[0]), max(p1[-1], p2[0]))
                        if e1 in remaining_edges:
                            # Check if p1 and p2 are disjoint in vertices
                            if not (set(p1) & set(p2)):
                                new_paths[i] = p1 + p2
                                del new_paths[j]
                                remaining_edges.discard(e1)
                                merged = True
                                break
                        # Also check reverse: p2 ends at p2[-1], p1 starts at p1[0]
                        e2 = (min(p2[-1], p1[0]), max(p2[-1], p1[0]))
                        if e2 in remaining_edges:
                            if not (set(p1) & set(p2)):
                                new_paths[i] = p2 + p1
                                del new_paths[j]
                                remaining_edges.discard(e2)
                                merged = True
                                break
                    if merged:
                        break

            return new_paths

        # Main decomposition with multiple restarts
        best_paths = None
        best_count = float('inf')

        # Time allocation: try several restarts
        num_restarts = 5
        time_per_restart = time_limit / num_restarts

        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break

            rng = random.Random(42 + restart)
            t_start = time.time()

            paths = try_decompose(edge_set, rng, time_per_restart)

            # Post-process: merge paths
            # First, compute which edges are used
            used_edges = set()
            for p in paths:
                for i in range(len(p) - 1):
                    e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                    used_edges.add(e)

            # Find edges not covered (shouldn't happen, but safety)
            uncovered = edge_set - used_edges
            for e in uncovered:
                paths.append(list(e))

            # Now try to merge
            remaining_for_merge = set(uncovered)  # Only uncovered edges can be used for merging
            # Actually, we can only merge if the connecting edge exists in the graph
            # But it must not be used in any path
            all_used = set()
            for p in paths:
                for i in range(len(p) - 1):
                    e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                    all_used.add(e)

            # Edges available for merging
            available_for_merge = edge_set - all_used
            merged_paths = merge_paths(paths, available_for_merge)

            if len(merged_paths) < best_count:
                best_count = len(merged_paths)
                best_paths = merged_paths

            if time.time() - t_start > time_per_restart:
                break

        if best_paths is None:
            # Fallback: each edge as its own path
            best_paths = [[u, v] for u, v in edges]

        return best_paths

    results = []
    for idx, spec in enumerate(specs):
        if time.time() > DEADLINE - 10:
            # Not enough time; return trivial solution for remaining
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)

        # Allocate time: give more time to later graphs if they're harder
        # But for simplicity, divide remaining time equally
        remaining_time = DEADLINE - time.time() - 5
        time_for_this = remaining_time / (len(specs) - idx)

        t_start = time.time()
        paths = decompose_graph(n, edges, time_limit=time_for_this)

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
