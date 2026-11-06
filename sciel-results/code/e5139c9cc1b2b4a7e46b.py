import random
import time
from itertools import permutations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    # Build weight matrix
    weight = [[0] * n for _ in range(n)]
    for idx, (u, v) in enumerate(P['edge_order']):
        weight[u][v] = idx + 1
        weight[v][u] = idx + 1
    
    def max_weight(cycle):
        m = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if weight[u][v] > m:
                m = weight[u][v]
        return m

    def two_opt_improve(cycle):
        improved = True
        while improved:
            improved = False
            current_max = max_weight(cycle)
            for i in range(n):
                for j in range(i + 1, n):
                    # Edges (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                    u = cycle[i]
                    v = cycle[(i + 1) % n]
                    x = cycle[j]
                    y = cycle[(j + 1) % n]
                    
                    # Check if edges are distinct and not adjacent in the cycle structure
                    # (u,v) and (x,y) are the edges being swapped.
                    # New edges will be (u,x) and (v,y) after reversing segment i+1..j
                    # We must ensure we don't create a self-loop or duplicate edge
                    if u == x or u == y or v == x or v == y:
                        continue
                    if (i + 1) % n == j or (j + 1) % n == i:
                        # Adjacent edges, 2-opt on adjacent edges is trivial or invalid for simple cycle improvement in this context
                        # Actually, if i and j are adjacent, the segment is length 1 or 2.
                        # Let's just allow the check, the weight check will handle it.
                        pass

                    # Calculate new max weight
                    # Instead of rebuilding the whole cycle and checking all edges,
                    # we can just check the two new edges and keep the rest if they were <= current_max.
                    # But we need the TRUE max of the new cycle.
                    # Since only 2 edges change, the new max is max(max(old_max_except_removed, new_edge1, new_edge2))
                    # This is slightly complex to track "old_max_except_removed".
                    # Given N=24, calculating full max_weight is O(N) = O(24).
                    # Total operations per iteration: O(N^2 * N) = O(N^3) = 24^3 ~ 13k.
                    # This is very fast.
                    
                    new_cycle = cycle[:]
                    # Reverse segment from i+1 to j
                    new_cycle[i+1:j+1] = reversed(new_cycle[i+1:j+1])
                    
                    new_max = max_weight(new_cycle)
                    if new_max < current_max:
                        cycle = new_cycle
                        current_max = new_max
                        improved = True
        return cycle

    # Greedy initial construction
    def greedy_cycle(start):
        visited = [False] * n
        cycle = [start]
        visited[start] = True
        for _ in range(n - 1):
            last = cycle[-1]
            min_w = float('inf')
            next_v = -1
            for v in range(n):
                if not visited[v]:
                    if weight[last][v] < min_w:
                        min_w = weight[last][v]
                        next_v = v
            cycle.append(next_v)
            visited[next_v] = True
        return cycle

    best_cycle = None
    best_score = float('inf')
    
    # Try multiple random starts
    rng = random.Random(42)
    for _ in range(1000):
        # Random permutation
        perm = list(range(n))
        rng.shuffle(perm)
        # Greedy from first element of perm? Or just use perm as start?
        # Greedy is better to get a good local structure.
        # Let's pick a random start vertex for greedy
        start = rng.choice(perm)
        c = greedy_cycle(start)
        c = two_opt_improve(c)
        score = max_weight(c)
        if score < best_score:
            best_score = score
            best_cycle = c
            if best_score == 1: # Theoretical minimum
                break

    return best_cycle
