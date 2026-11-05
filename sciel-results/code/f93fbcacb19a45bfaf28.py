"""Path cover via iterative fragment merging using maximum matching."""
import time
import heapq
from collections import deque
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin under 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_one(n, p, seed, deadline):
        edges = instance(n, p, seed)
        n_verts = n
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))

        # Build adjacency with edge tracking
        adj = [[] for _ in range(n_verts)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize: each edge is its own path
        # paths: list of lists of vertices
        # For each path, track which edges it uses
        paths = [[u, v] for u, v in edges]
        # path_edges: for each path, set of (min(u,v), max(u,v))
        path_edges = [set() for _ in paths]
        for i, (u, v) in enumerate(edges):
            path_edges[i].add((min(u, v), max(u, v)))

        # Track which edge belongs to which path
        # edge_to_path: (u,v) -> path_index
        edge_to_path = {}
        for i, (u, v) in enumerate(edges):
            edge_to_path[(u, v)] = i
            edge_to_path[(v, u)] = i

        # For merging, we need to check if two paths can be joined
        # Two paths can be merged if:
        #   1. An endpoint of path A is connected by an edge to an endpoint of path B
        #      (endpoint-to-endpoint merge)
        #   2. An endpoint of path A is connected by an edge to an internal vertex of path B
        #      (endpoint-to-internal merge, which splits path B)

        def get_endpoints(path):
            return path[0], path[-1]

        def try_merge(paths, path_edges, edge_to_path, adj, n_verts):
            """Try to merge paths. Returns (merged, new_paths, new_path_edges, new_edge_to_path).
            Uses greedy: find all valid merges, pick best ones."""
            if len(paths) <= 1:
                return False, paths, path_edges, edge_to_path

            # Find all possible merges
            # Type 1: endpoint-to-endpoint (simple concatenation)
            # Type 2: endpoint-to-internal (split and rejoin)

            # For efficiency, build vertex-to-paths mapping
            # vertex_to_paths: vertex -> list of (path_idx, position) where position is 0 (start), -1 (end), or internal
            vertex_positions = {}
            for pi, path in enumerate(paths):
                for pos, v in enumerate(path):
                    if v not in vertex_positions:
                        vertex_positions[v] = []
                    vertex_positions[v].append((pi, pos))

            # Try all pairs of paths for merging
            best_merges = []  # (cost, type, path_a, path_b, endpoint_a, endpoint_b, internal_pos)
            # cost = 0 for endpoint-endpoint, 1 for endpoint-internal (we prefer simpler merges)

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    path_a = paths[i]
                    path_b = paths[j]
                    ep_a_start, ep_a_end = path_a[0], path_a[-1]
                    ep_b_start, ep_b_end = path_b[0], path_b[-1]

                    # Check endpoint-to-endpoint merges
                    # ep_a_start connected to ep_b_start or ep_b_end
                    # ep_a_end connected to ep_b_start or ep_b_end
                    for ea in (ep_a_start, ep_a_end):
                        for eb in (ep_b_start, ep_b_end):
                            if ea != eb and (ea, eb) in edge_to_path:
                                # Edge exists and belongs to some path
                                # For endpoint-endpoint merge, the connecting edge must NOT be in path_a or path_b
                                # Actually, the edge between ea and eb must exist in the graph
                                # and must not already be used by path_a or path_b
                                # Wait - the edge is between endpoints. It's in the graph.
                                # We need to check it's not already used by either path.
                                edge_key = (min(ea, eb), max(ea, eb))
                                if edge_key not in path_edges[i] and edge_key not in path_edges[j]:
                                    # Valid endpoint-endpoint merge
                                    best_merges.append((0, 1, i, j, ea, eb, -1))
                                break
                            if ea != eb and (eb, ea) in edge_to_path:
                                edge_key = (min(ea, eb), max(ea, eb))
                                if edge_key not in path_edges[i] and edge_key not in path_edges[j]:
                                    best_merges.append((0, 1, i, j, ea, eb, -1))
                                break

                    # Check endpoint-to-internal merges
                    # An endpoint of path_a connects to an internal vertex of path_b
                    # This means we split path_b at that internal vertex and merge
                    for ea in (ep_a_start, ep_a_end):
                        for pos in range(1, len(path_b) - 1):
                            v_internal = path_b[pos]
                            if ea != v_internal and (ea, v_internal) in edge_to_path:
                                edge_key = (min(ea, v_internal), max(ea, v_internal))
                                if edge_key not in path_edges[i] and edge_key not in path_edges[j]:
                                    best_merges.append((1, 2, i, j, ea, v_internal, pos))
                            elif ea != v_internal and (v_internal, ea) in edge_to_path:
                                edge_key = (min(ea, v_internal), max(ea, v_internal))
                                if edge_key not in path_edges[i] and edge_key not in path_edges[j]:
                                    best_merges.append((1, 2, i, j, ea, v_internal, pos))

            if not best_merges:
                return False, paths, path_edges, edge_to_path

            # Sort by cost (prefer endpoint-endpoint)
            best_merges.sort(key=lambda x: x[0])

            # Greedily apply merges
            applied = set()
            new_paths = list(paths)
            new_path_edges = [set(pe) for pe in path_edges]
            new_edge_to_path = dict(edge_to_path)
            merged_any = False

            for cost, mtype, pi, pj, ea, eb, pos in best_merges:
                if pi in applied or pj in applied:
                    continue
                if time.time() > deadline:
                    break

                if mtype == 1:
                    # Endpoint-to-endpoint merge
                    path_a = new_paths[pi]
                    path_b = new_paths[pj]

                    # Determine orientation
                    # If ea is start of path_a, path_a starts with ea
                    # If ea is end of path_a, path_a ends with ea
                    a_starts = (path_a[0] == ea)
                    b_ends = (path_b[-1] == eb)

                    if a_starts and b_ends:
                        # path_a: ea ... , path_b: ... eb
                        # Merge: path_a + path_b[1:]  (if ea==eb, but they're different)
                        # Actually ea connects to eb, so new path = path_a + [eb] + path_b[1:]
                        # Wait, ea is endpoint of A, eb is endpoint of B, and edge (ea,eb) exists
                        # New path: path_a followed by path_b reversed if needed
                        # path_a ends at ea (if a_starts, it starts at ea, so it goes away from ea)
                        # Let me think more carefully.
                        # path_a = [ea, ..., X] if a_starts, or [X, ..., ea] if not
                        # path_b = [Y, ..., eb] if b_ends, or [eb, ..., Y] if not
                        # We want to connect ea to eb.
                        # If a_starts: path_a = [ea, ...]. We continue from ea to eb, so append path_b starting from eb.
                        #   If b_ends: path_b = [..., eb]. Reverse: [eb, ...]. New = path_a + path_b[1:] reversed... 
                        #   Actually: new = path_a + list(reversed(path_b))[1:]  NO
                        #   path_a = [ea, v1, ..., vk], path_b = [w1, ..., wm, eb]
                        #   Edge ea-eb. New path: [ea, v1, ..., vk, eb, wm, ..., w1]? No, that doesn't use edge ea-eb.
                        #   We need: ...ea, eb... so new = path_a + [eb] + path_b without eb
                        #   If b_ends: path_b[-1] = eb, so path_b without eb = path_b[:-1]
                        #   new = path_a + path_b[:-1]... but that skips the edge ea-eb!
                        #   
                        #   I think I'm overcomplicating. The merge means:
                        #   We have two separate paths. We add the edge between ea and eb.
                        #   New path = path_a (ending at ea) + edge to eb + path_b (starting at eb)
                        #   So we need path_a to end at ea and path_b to start at eb.
                        #   If path_a starts at ea, reverse it so it ends at ea.
                        #   If path_b ends at eb, reverse it so it starts at eb.

                        pass  # handled below

                    # Let's just handle all 4 cases uniformly
                    # We want the merged path to go through ea then eb (or eb then ea)
                    # Orient path_a so ea is at the end
                    if path_a[0] == ea:
                        path_a = list(reversed(path_a))
                    # Now path_a[-1] == ea
                    # Orient path_b so eb is at the start
                    if path_b[-1] == eb:
                        path_b = list(reversed(path_b))
                    # Now path_b[0] == eb
                    # New path = path_a + path_b[1:]
                    new_path = path_a + path_b[1:]
                    new_edge = (min(ea, eb), max(ea, eb))

                    # Verify: new_path is simple (no repeated vertices)
                    if len(set(new_path)) == len(new_path):
                        new_paths[pi] = new_path
                        new_path_edges[pi] = new_path_edges[pi] | new_path_edges[pj] | {new_edge}
                        new_edge_to_path[new_edge] = pi
                        new_edge_to_path[(ea, eb)] = pi
                        new_edge_to_path[(eb, ea)] = pi
                        # Remove path j
                        new_paths[pj] = None
                        new_path_edges[pj] = None
                        applied.add(pi)
                        applied.add(pj)
                        merged_any = True

                elif mtype == 2:
                    # Endpoint-to-internal merge
                    path_a = new_paths[pi]
                    path_b = new_paths[pj]
                    v_internal = path_b[pos]

                    # Orient path_a so ea is at the end
                    if path_a[0] == ea:
                        path_a = list(reversed(path_a))
                    # Now path_a[-1] == ea

                    # Split path_b at position pos
                    # path_b = [b0, b1, ..., bpos, ..., b_last]
                    # Two pieces: path_b[0:pos+1] and path_b[pos:]
                    # We connect ea to v_internal (= path_b[pos])
                    # New path = path_a + path_b[pos:]  (the part from v_internal to end)
                    # The other piece path_b[0:pos+1] becomes a separate path
                    # But wait, we're removing the edge between v_internal and its neighbor in path_b
                    # Actually no - we're adding a new edge (ea, v_internal) and keeping all existing edges
                    # The path_b gets split: the segment from v_internal to one end merges with path_a
                    # The other segment remains as a separate path

                    # path_b[pos:] = [v_internal, ..., b_last]
                    # path_b[:pos+1] = [b0, ..., v_internal]
                    # We merge path_a (ending at ea) with edge (ea, v_internal) with path_b[pos:]
                    # New combined: path_a + path_b[pos+1:]
                    # Remaining: path_b[:pos+1] (which ends at v_internal)

                    new_combined = path_a + path_b[pos + 1:]
                    remaining = path_b[:pos + 1]

                    new_edge = (min(ea, v_internal), max(ea, v_internal))

                    if len(set(new_combined)) == len(new_combined) and len(set(remaining)) == len(remaining):
                        new_paths[pi] = new_combined
                        new_paths[pj] = remaining
                        # Update edge sets
                        # The edge between path_b[pos] and path_b[pos+1] is now in 'remaining'
                        # The edge between path_b[pos-1] and path_b[pos] is now in 'remaining'
                        # Actually all edges of path_b are still there, just split
                        # new_combined gets: path_a's edges + (ea, v_internal) + edges from v_internal to end of path_b
                        # remaining gets: edges from start of path_b to v_internal

                        # Let's just recompute
                        old_a_edges = set(new_path_edges[pi])
                        old_b_edges = set(new_path_edges[pj])

                        # Edges in new_combined: all of old_a_edges + new_edge + edges of path_b from pos to end
                        # Edges in remaining: edges of path_b from 0 to pos
                        combined_edges = set(old_a_edges)
                        combined_edges.add(new_edge)
                        for k in range(pos, len(path_b) - 1):
                            e = (min(path_b[k], path_b[k + 1]), max(path_b[k], path_b[k + 1]))
                            combined_edges.add(e)

                        rem_edges = set()
                        for k in range(0, pos):
                            e = (min(path_b[k], path_b[k + 1]), max(path_b[k], path_b[k + 1]))
                            rem_edges.add(e)

                        new_path_edges[pi] = combined_edges
                        new_path_edges[pj] = rem_edges
                        new_edge_to_path[new_edge] = pi
                        new_edge_to_path[(ea, v_internal)] = pi
                        new_edge_to_path[(v_internal, ea)] = pi
                        applied.add(pi)
                        applied.add(pj)
                        merged_any = True

            # Clean up None entries
            if merged_any:
                new_paths = [p for p in new_paths if p is not None]
                new_path_edges = [pe for pe in new_path_edges if pe is not None]
                # Rebuild edge_to_path
                new_edge_to_path = {}
                for pi, pe in enumerate(new_path_edges):
                    for e in pe:
                        new_edge_to_path[e] = pi
                        new_edge_to_path[(e[1], e[0])] = pi
                return True, new_paths, new_path_edges, new_edge_to_path

            return False, paths, path_edges, edge_to_path

        # Iterative merging
        max_rounds = 50
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            if len(paths) <= 1:
                break
            merged, paths, path_edges, edge_to_path = try_merge(
                paths, path_edges, edge_to_path, adj, n_verts
            )
            if not merged:
                break

        # Filter out any empty paths
        paths = [p for p in paths if p is not None and len(p) >= 2]
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        remaining = DEADLINE - time.time()
        if remaining < 5.0:
            # Fallback: trivial edge-as-path cover
            results.append([[u, v] for u, v in instance(*spec)])
            continue
        paths = solve_one(*spec, deadline=time.time() + remaining * 0.9)
        results.append(paths)
    return results


if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
