import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # safety margin before 90s

    # 255 = 3 * 5 * 17
    # Any element in the set must be adjacent to 255, i.e., gcd(x, 255) > 1
    # So x must be divisible by 3, 5, or 17

    # Build candidate list: all numbers in [1,255] with gcd(x,255)>1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)

    # 255 must be in the set
    # Find index of 255
    idx_255 = candidates.index(255)

    n = len(candidates)

    # Build adjacency using bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # We need max clique containing idx_255
    # All other vertices must be adjacent to 255

    # Greedy initial clique: start with multiples of 3 (a known clique)
    init_clique = set()
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_clique.add(i)

    best_clique = init_clique.copy()
    best_size = len(best_clique)

    # Popcount helper
    def popcount(x):
        return bin(x).count('1')

    best = [best_size]
    best_set = [best_clique.copy()]

    # Bron-Kerbosch with pivot for max clique
    def bron_kerbosch(clique, candidates_mask):
        if time.time() > DEADLINE:
            return

        if popcount(candidates_mask) + len(clique) <= best[0]:
            return

        if candidates_mask == 0:
            if len(clique) > best[0]:
                best[0] = len(clique)
                best_set[0] = set(clique)
            return

        # Choose pivot: vertex with max neighbors in candidates
        pivot = -1
        max_deg = -1
        temp = candidates_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = popcount(adj[v] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                pivot = v

        # Branch on candidates not adjacent to pivot
        branch_mask = candidates_mask & ~(adj[pivot] if pivot >= 0 else 0)

        temp = branch_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1

            new_clique = clique | {v}
            new_candidates = candidates_mask & adj[v]

            bron_kerbosch(new_clique, new_candidates)

            candidates_mask &= ~(1 << v)

            if time.time() > DEADLINE:
                return

    # Start: clique = {idx_255}, candidates = all neighbors of 255
    init_clique_set = {idx_255}
    init_candidates = adj[idx_255]

    bron_kerbosch(init_clique_set, init_candidates)

    # Greedy polish: try to add more vertices to the best clique found
    def try_add_vertex(clique_set, candidates_list):
        """Try to add a vertex from candidates_list to clique_set if it's adjacent to all."""
        for v in candidates_list:
            if v in clique_set:
                continue
            can_add = True
            for u in clique_set:
                if not (adj[v] & (1 << u)):
                    can_add = False
                    break
            if can_add:
                return v
        return None

    # Multiple polish passes
    polish_clique = best_set[0].copy()
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        # Find all vertices not in clique that are adjacent to 255
        not_in_clique = [i for i in range(n) if i not in polish_clique and i != idx_255 and (adj[idx_255] & (1 << i))]
        added = try_add_vertex(polish_clique, not_in_clique)
        if added is not None:
            polish_clique.add(added)
        else:
            break

    if len(polish_clique) > best[0]:
        best[0] = len(polish_clique)
        best_set[0] = polish_clique

    # Convert back to values
    result = [candidates[i] for i in best_set[0]]
    result.sort()

    # Verify 255 is in result
    if 255 not in result:
        result.append(255)

    return result
