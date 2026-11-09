"""Rewritten program: greedy edge-colouring with local 4-cycle checks."""
import random
import time
import itertools

N = 20
C = 20
DEADLINE_SECONDS = 115.0  # safety margin before 120s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best_edges = []
    best_count = 0

    # We'll try multiple random restarts with different orderings.
    # For each restart, we attempt to add edges one by one, checking
    # that no 4-cycle is formed with duplicate colours.

    # Precompute all 4-cycles? No, too many. Instead, when adding edge (u,v),
    # we check: for every pair of existing edges (a,b) and (c,d) such that
    # {a,b} and {c,d} together with (u,v) form a 4-cycle, the colours must be distinct.

    # A 4-cycle containing edge (u,v) has the form: u - x - y - v - u
    # where x != v, y != u, x != y, and (u,x), (x,y), (y,v) are existing edges.
    # So for each pair (x,y) where (u,x) in E, (x,y) in E, (y,v) in E, and x!=y, x!=v, y!=u:
    #   colours of (u,v), (u,x), (x,y), (y,v) must all be distinct.

    # To make this efficient, we maintain:
    #   adj: dict of (u,v) -> colour, with u < v
    #   adj_list: for each vertex, list of (neighbor, colour)

    # When trying to add edge (u,v) with colour c:
    #   For each common "path of length 2" from u to v: u-x-y-v
    #   Check that c, col(u,x), col(x,y), col(y,v) are all distinct.

    # Paths of length 2 from u to v:
    #   x is a neighbor of u, y is a neighbor of v, and (x,y) is an edge.
    #   Also need x != v, y != u, x != y.

    def try_random_construction(order_seed, time_limit):
        """Try one random construction. Returns list of [u,v,c]."""
        random.seed(order_seed)

        # Random order of edges
        all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(all_edges)

        # State
        edge_colour = {}  # (u,v) with u<v -> colour
        adj = {i: [] for i in range(N)}  # adj[i] = list of (j, colour)

        edges_result = []

        for (u, v) in all_edges:
            if time.time() > time_limit:
                break

            # Try each colour
            # For efficiency, try colours in random order
            colours = list(range(C))
            random.shuffle(colours)

            placed = False
            for c in colours:
                # Check: for all 4-cycles u-x-y-v, colours must be distinct
                # x in adj[u], y in adj[v], (x,y) in edge_colour
                valid = True
                for (x, cx) in adj[u]:
                    if x == v:
                        continue
                    for (y, cy) in adj[v]:
                        if y == u or y == x:
                            continue
                        # Check if (x,y) is an edge
                        if x < y:
                            key = (x, y)
                        else:
                            key = (y, x)
                        if key not in edge_colour:
                            continue
                        cyx = edge_colour[key]
                        # Colours: c (for u-v), cx (for u-x), cyx (for x-y), cy (for-v)
                        if len({c, cx, cyx, cy}) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Place edge
                    if u < v:
                        key = (u, v)
                    else:
                        key = (v, u)
                    edge_colour[key] = c
                    adj[u].append((v, c))
                    adj[v].append((u, c))
                    edges_result.append([u, v, c])
                    placed = True
                    break

        return edges_result

    # Try multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() >= deadline - 5:
            break

        # Give each restart a fair share of time
        remaining = deadline - time.time()
        time_per_restart = min(remaining / max(1, num_restarts - i), 20.0)
        if time_per_restart < 1.0:
            break

        result = try_random_construction(seed=i * 1000 + 42, time_limit=time.time() + time_per_restart)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
