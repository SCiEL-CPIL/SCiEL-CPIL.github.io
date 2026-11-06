import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _lower_bound(edges):
        if not edges:
            return 0
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd_count = sum(1 for d in degree.values() if d % 2 == 1)
        return max(1, (odd_count + 1) // 2)

    def _greedy_decompose(edges, rng, deadline):
        """Single greedy pass with randomized choices."""
        remaining = set(edges)
        if not remaining:
            return []

        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        degree = defaultdict(int)
        for u, v in remaining:
            degree[u] += 1
            degree[v] += 1

        paths = []

        while remaining:
            if time.time() > deadline:
                # Fallback: dump remaining edges as single-edge paths
                for e in remaining:
                    paths.append(list(e))
                return paths

            # Pick start vertex: prefer odd-degree, then low-degree (leaf-like)
            active = [v for v in adj if adj[v]]
            if not active:
                break

            odd_verts = [v for v in active if degree[v] % 2 == 1]
            if odd_verts:
                # Among odd-degree, prefer lowest degree (most leaf-like)
                start = min(odd_verts, key=lambda x: (degree[x], x))
                # Add randomness: with 30% chance pick a random odd vertex
                if rng.random() < 0.3:
                    start = rng.choice(odd_verts)
            else:
                # No odd vertices: pick lowest degree for leaf preference
                start = min(active, key=lambda x: (degree[x], x))
                if rng.random() < 0.3:
                    start = rng.choice(active)

            # Greedy path extension with Warnsdorff heuristic
            path = [start]
            visited = {start}
            current = start

            while True:
                avail = [nb for nb in adj[current] if nb not in visited]
                if not avail:
                    break

                # Warnsdorff: prefer neighbor with fewest unvisited neighbors
                # (dead-end preference to keep core for later)
                def score(nb):
                    return sum(1 for x in adj[nb] if x not in visited)

                avail.sort(key=lambda x: (score(x), x))

                # Randomized choice: 70% greedy, 30% random from top-3
                if len(avail) > 1 and rng.random() < 0.3:
                    top_k = min(3, len(avail))
                    next_v = rng.choice(avail[:top_k])
                else:
                    next_v = avail[0]

                # Remove edge and update state
                edge = tuple(sorted((current, next_v)))
                remaining.discard(edge)
                adj[current].discard(next_v)
                adj[next_v].discard(current)
                degree[current] -= 1
                degree[next_v] -= 1

                path.append(next_v)
                visited.add(next_v)
                current = next_v

            if len(path) >= 2:
                paths.append(path)
            else:
                # Single vertex, no edges left from it; skip
                pass

        return paths

    def _try_merge_paths(paths, edges_set):
        """
        Post-hoc: try to merge two paths if an endpoint of one connects to
        an endpoint of the other via an edge in edges_set, and the merged
        path is simple. Returns new list of paths if a merge was made.
        """
        if len(paths) < 2:
            return paths

        # Build endpoint lookup
        # For each path, endpoints are path[0] and path[-1]
        # Check all pairs of paths for mergeable endpoints
        merged = False
        new_paths = list(paths)

        for i in range(len(new_paths)):
            if merged:
                break
            for j in range(i + 1, len(new_paths)):
                if merged:
                    break
                p1 = new_paths[i]
                p2 = new_paths[j]
                if len(p1) < 2 or len(p2) < 2:
                    continue

                # Try 4 combinations:
                # p1_end -> p2_start, p1_end -> p2_end (reverse p2)
                # p1_start -> p2_start, p1_start -> p2_end (reverse p2)
                candidates = []
                # p1 tail to p2 head
                if tuple(sorted((p1[-1], p2[0]))) in edges_set:
                    candidates.append(p1 + p2[1:])
                # p1 tail to p2 tail (reverse p2)
                if tuple(sorted((p1[-1], p2[-1]))) in edges_set:
                    candidates.append(p1 + p2[::-1][1:])
                # p1 head to p2 head
                if tuple(sorted((p1[0], p2[0]))) in edges_set:
                    candidates.append(p2[1:] + p1)
                # p1 head to p2 tail (reverse p2)
                if tuple(sorted((p1[0], p2[-1]))) in edges_set:
                    candidates.append(p2[::-1][1:] + p1)

                for cand in candidates:
                    # Check simplicity: all vertices unique
                    if len(set(cand)) == len(cand):
                        # Verify all edges in cand are in edges_set
                        valid = True
                        for k in range(len(cand) - 1):
                            e = tuple(sorted((cand[k], cand[k+1])))
                            if e not in edges_set:
                                valid = False
                                break
                        if valid:
                            # Merge: remove i and j, add cand
                            # Remove j first (higher index) to preserve indices
                            new_paths.pop(j)
                            new_paths[i] = cand
                            merged = True
                            break

        return new_paths

    def _decompose_instance(edges, time_budget):
        """Try multiple restarts, keep best, apply merging."""
        if not edges:
            return []

        best_paths = None
        best_count = len(edges)  # worst case
        lb = _lower_bound(edges)

        max_attempts = 30
        attempt_deadline = start_time + time_budget

        for attempt in range(max_attempts):
            if time.time() > attempt_deadline:
                break

            rng = random.Random(attempt * 7919 + 13)
            paths = _greedy_decompose(edges, rng, attempt_deadline)

            if paths is None or not paths:
                continue

            # Try merging to reduce count
            edges_set = set(edges)
            # Only merge if it doesn't violate edge-disjointness
            # Since paths are edge-disjoint by construction, merging
            # uses an edge that is in the graph but not in any path.
            # Actually, the merge edge must be in edges_set but NOT used
            # by any existing path. Let's check: the merge connects two
            # path endpoints with an edge. That edge must not be used
            # by either path (it's not, since it connects endpoints of
            # different paths and paths are edge-disjoint). But it must
            # be in the graph.
            # However, we need to ensure the merge edge is actually
            # available (not used by any path). Since paths are
            # edge-disjoint and cover a subset of edges, the merge
            # edge is in edges_set. But if it's already used by a
            # path, we can't use it. Let's check.
            # Actually, the merge edge connects an endpoint of path i
            # to an endpoint of path j. Since paths are edge-disjoint,
            # no path uses this edge (it would have to be internal to
            # one path, but it connects two different paths). So it's
            # available.
            merged = _try_merge_paths(paths, edges_set)
            if len(merged) < len(paths):
                paths = merged

            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                if count <= lb:
                    break

        if best_paths is None:
            # Fallback: each edge as its own path
            best_paths = [list(e) for e in edges]

        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    # Allocate time budget per instance proportionally to edge count
    # but with a minimum
    total_time = deadline - start_time
    # Reserve 10s for overhead
    usable = total_time - 10

    # Estimate: smaller graphs get less time, larger get more
    # n=20: ~15s, n=25: ~20s, n=30: ~25s (roughly proportional to n)
    budgets = [usable * 0.3, usable * 0.35, usable * 0.35]

    for idx, spec in enumerate(specs):
        if time.time() > deadline - 5:
            # Not enough time; use fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        time_budget = budgets[idx]
        paths = _decompose_instance(edges, time_budget)
        results.append(paths)

    return results


if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
    print(f"Total: {sum(len(r) for r in res)} paths")
