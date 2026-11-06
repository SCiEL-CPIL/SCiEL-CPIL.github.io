"""Strong edge colouring: greedy + aggressive class merging + local search."""
import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, n):
        """Build the strong edge-colouring conflict graph as bitmasks."""
        m = len(edges)
        # vertex -> list of edge indices incident to it
        v2e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v2e[u].append(i)
            v2e[v].append(i)

        # Build adjacency sets for vertices
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # conflict[i] is a bitmask of edges that conflict with edge i
        conflict = [0] * m
        for i in range(m):
            u, v = edges[i]
            mask = 0
            # Edges sharing a vertex with (u,v)
            for j in v2e[u]:
                if j != i:
                    mask |= (1 << j)
            for j in v2e[v]:
                if j != i:
                    mask |= (1 << j)
            # Edges joined by an edge to (u,v)
            # For each vertex w adjacent to u (w != v), edges incident to w
            for w in adj[u]:
                if w == v:
                    continue
                for j in v2e[w]:
                    mask |= (1 << j)
            for w in adj[v]:
                if w == u:
                    continue
                for j in v2e[w]:
                    mask |= (1 << j)
            # Remove self
            mask &= ~(1 << i)
            conflict[i] = mask
        return conflict

    def greedy_color(conflict, order, m):
        """Assign colours using greedy first-fit in the given edge order."""
        colour = [0] * m
        # colour_to_edges[c] = bitmask of edges with colour c
        # We'll track used colours dynamically
        max_colour = 0
        for idx in order:
            forbidden_mask = 0
            # Get all colours used by conflicting edges
            c = 1
            while True:
                # Check if colour c is available
                # An edge with colour c conflicts if it's in conflict[idx]
                # We need to check if any edge with colour c is in conflict[idx]
                # Instead, let's track which colours are forbidden
                break

        # Better approach: track colour usage
        colour = [0] * m
        # For each colour, maintain a bitmask of edges using that colour
        # colour_masks[c] = bitmask of edges with colour c
        # We'll use a dict or list that grows as needed

        # Simpler: for each edge, find smallest colour not used by any conflict
        # To speed up, maintain for each colour a bitmask
        # But that's O(m) per colour check...

        # Let's just do it directly but efficiently
        colour = [0] * m
        max_colour = 0
        # colour_edge_mask[c] = bitmask of edges with colour c
        colour_edge_mask = defaultdict(int)

        for idx in order:
            forbidden_colours = set()
            # For each conflicting edge, add its colour to forbidden
            cm = conflict[idx]
            # Iterate through set bits in cm
            temp = cm
            while temp:
                # Get lowest set bit
                low = temp & (-temp)
                j = low.bit_length() - 1
                if colour[j] > 0:
                    forbidden_colours.add(colour[j])
                temp ^= low

            c = 1
            while c in forbidden_colours:
                c += 1
            colour[idx] = c
            colour_edge_mask[c] |= (1 << idx)
            if c > max_colour:
                max_colour = c

        return colour, max_colour, colour_edge_mask

    def can_merge(colour_edge_mask, c1, c2, conflict, m):
        """Check if colour classes c1 and c2 can be merged (no conflicts between them)."""
        mask1 = colour_edge_mask.get(c1, 0)
        mask2 = colour_edge_mask.get(c2, 0)
        # For each edge in class c1, check if it conflicts with any edge in class c2
        temp = mask1
        while temp:
            low = temp & (-temp)
            i = low.bit_length() - 1
            # Check if any conflicting edge of i is in mask2
            if (conflict[i] & mask2) != 0:
                return False
            temp ^= low
        return True

    def merge_classes(colour, colour_edge_mask, c1, c2, m):
        """Merge colour class c2 into c1. Returns updated structures."""
        mask2 = colour_edge_mask.get(c2, 0)
        # Re-colour all edges in c2 to c1
        temp = mask2
        while temp:
            low = temp & (-temp)
            i = low.bit_length() - 1
            colour[i] = c1
            temp ^= low
        # Update masks
        colour_edge_mask[c1] |= mask2
        if c2 in colour_edge_mask:
            del colour_edge_mask[c2]
        return colour, colour_edge_mask

    def try_local_improve(colour, colour_edge_mask, max_colour, conflict, m, time_limit=3.0):
        """Try to reduce colour count by merging or re-colouring."""
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            # Try to remove the highest colour by re-colouring its edges
            for target_c in range(max_colour, 1, -1):
                if time.time() > DEADLINE - 5.0:
                    return colour, colour_edge_mask, max_colour
                mask_c = colour_edge_mask.get(target_c, 0)
                if mask_c == 0:
                    continue
                # Try to re-colour each edge in this class to a lower colour
                temp = mask_c
                found = False
                while temp:
                    low = temp & (-temp)
                    i = low.bit_length() - 1
                    # Find forbidden colours
                    forbidden = set()
                    cm = conflict[i]
                    temp2 = cm
                    while temp2:
                        low2 = temp2 & (-temp2)
                        j = low2.bit_length() - 1
                        if colour[j] > 0 and colour[j] != target_c:
                            forbidden.add(colour[j])
                        temp2 ^= low2
                    # Try colours 1 to target_c-1
                    for c in range(1, target_c):
                        if c not in forbidden:
                            # Re-colour edge i to c
                            colour[i] = c
                            colour_edge_mask[c] |= (1 << i)
                            # Remove from target_c
                            colour_edge_mask[target_c] &= ~(1 << i)
                            if colour_edge_mask[target_c] == 0:
                                del colour_edge_mask[target_c]
                            # Check if target_c is now empty
                            if target_c not in colour_edge_mask:
                                max_colour -= 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
                    temp ^= low
                if improved:
                    break

            # Try merging colour classes
            if not improved and max_colour > 1:
                colours = list(colour_edge_mask.keys())
                for i in range(len(colours)):
                    for j in range(i + 1, len(colours)):
                        c1, c2 = colours[i], colours[j]
                        if c1 > c2:
                            c1, c2 = c2, c1
                        if can_merge(colour_edge_mask, c1, c2, conflict, m):
                            colour, colour_edge_mask = merge_classes(colour, colour_edge_mask, c1, c2, m)
                            max_colour = max(colour_edge_mask.keys()) if colour_edge_mask else 0
                            improved = True
                            break
                    if improved:
                        break

        return colour, colour_edge_mask, max_colour

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict(edges, n)

        # Build vertex degrees for edge ordering
        vdeg = [0] * n
        for u, v in edges:
            vdeg[u] += 1
            vdeg[v] += 1

        best_colour = None
        best_max = m + 1
        best_mask = None

        # Try multiple orderings
        orderings = []
        # 1. Original sorted order
        orderings.append(list(range(m)))
        # 2. Descending by max endpoint degree
        orderings.append(sorted(range(m), key=lambda i: max(vdeg[edges[i][0]], vdeg[edges[i][1]]), reverse=True))
        # 3. Descending by sum of endpoint degrees
        orderings.append(sorted(range(m), key=lambda i: vdeg[edges[i][0]] + vdeg[edges[i][1]], reverse=True))
        # 4. Random shuffles
        rng = random.Random(42)
        for _ in range(30):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        for order in orderings:
            if time.time() > DEADLINE - 10.0:
                break
            colour, max_c, c_mask = greedy_color(conflict, order, m)
            if max_c < best_max:
                best_max = max_c
                best_colour = colour[:]
                best_mask = {k: v for k, v in c_mask.items()}
            # Try local improvement
            if time.time() < DEADLINE - 15.0:
                colour_imp, mask_imp, max_imp = try_local_improve(colour, c_mask, max_c, conflict, m, time_limit=2.0)
                if max_imp < best_max:
                    best_max = max_imp
                    best_colour = colour_imp[:]
                    best_mask = {k: v for k, v in mask_imp.items()}

        # More random restarts if time permits
        if best_colour is not None and time.time() < DEADLINE - 20.0:
            for _ in range(100):
                if time.time() > DEADLINE - 10.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, max_c, c_mask = greedy_color(conflict, order, m)
                if max_c < best_max:
                    best_max = max_c
                    best_colour = colour[:]
                    best_mask = {k: v for k, v in c_mask.items()}
                if time.time() < DEADLINE - 15.0:
                    colour_imp, mask_imp, max_imp = try_local_improve(colour, c_mask, max_c, conflict, m, time_limit=1.5)
                    if max_imp < best_max:
                        best_max = max_imp
                        best_colour = colour_imp[:]
                        best_mask = {k: v for k, v in mask_imp.items()}

        if best_colour is None:
            # Fallback
            colour, _, _ = greedy_color(conflict, list(range(m)), m)
            return colour
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5.0:
            out.append([])
            continue
        result = solve_instance(*spec)
        out.append(result)

    return out
