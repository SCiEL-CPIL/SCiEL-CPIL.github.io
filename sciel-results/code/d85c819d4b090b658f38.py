import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # 255 = 3 * 5 * 17
    # Any element in our set must share a prime factor with 255, so must be divisible by 3, 5, or 17.
    # We need the largest subset of {x in [1,255] : gcd(x,255) > 1} that is pairwise non-coprime, containing 255.

    # Key structural insight: The multiples of 3 form a clique of size 85 (floor(255/3) = 85).
    # Can we do better? Let's use a smarter approach.

    # Build the candidate set
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)

    N = len(candidates)

    # Build adjacency using bitmasks for speed
    # adj_mask[i] is a bitmask where bit j is set if i and j are adjacent
    adj_mask = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    # Find index of 255
    idx_255 = candidates.index(255)

    # We need max clique containing 255, which is 1 + max clique in neighbors of 255
    # Get neighbors of 255 as a bitmask
    neighbors_255_mask = adj_mask[idx_255]

    # List of neighbor indices
    neighbor_indices = []
    for i in range(N):
        if i != idx_255 and (neighbors_255_mask >> i) & 1:
            neighbor_indices.append(i)

    # Now find max clique in the subgraph induced by neighbor_indices
    # Use branch and bound with coloring

    best_size = 0
    best_clique = []

    # Tomita's greedy coloring for upper bound
    def color_bound(vertices):
        """Assign colors to vertices to get an upper bound on clique size."""
        if not vertices:
            return 0

        colors = {}
        color_classes = []  # List of lists, each list is a set of vertices with same color

        for v in vertices:
            # Find the smallest color class where v is not adjacent to any vertex
            placed = False
            for ci, cc in enumerate(color_classes):
                compatible = True
                for u in cc:
                    if not ((adj_mask[v] >> u) & 1):
                        compatible = False
                        break
                if compatible:
                    cc.append(v)
                    placed = True
                    break
            if not placed:
                color_classes.append([v])

        return len(color_classes)

    # Branch and bound for max clique
    def bnb_max_clique(P_mask, C_size, C_list):
        nonlocal best_size, best_clique

        if time.time() > deadline:
            return

        if not P_mask:
            if C_size > best_size:
                best_size = C_size
                best_clique = C_list[:]
            return

        # Upper bound using coloring
        # Get list of vertices from P_mask
        P_list = []
        for i in range(N):
            if (P_mask >> i) & 1:
                P_list.append(i)

        ub = color_bound(P_list)
        if C_size + ub <= best_size:
            return

        # Choose a vertex to branch on (heuristic: highest degree in P)
        best_v = -1
        best_deg = -1
        for v in P_list:
            deg = 0
            for u in P_list:
                if u != v and ((adj_mask[v] >> u) & 1):
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = v

        if best_v == -1:
            return

        # Branch 1: Include best_v
        new_P = P_mask & adj_mask[best_v]
        C_list.append(best_v)
        bnb_max_clique(new_P, C_size + 1, C_list)
        C_list.pop()

        # Branch 2: Exclude best_v
        new_P2 = P_mask & ~(1 << best_v)
        bnb_max_clique(new_P2, C_size, C_list)

    # Run B&B on neighbors of 255
    bnb_max_clique(neighbors_255_mask, 0, [])

    # The result is best_clique + [255]
    result = [candidates[i] for i in best_clique]
    result.append(255)

    return result
