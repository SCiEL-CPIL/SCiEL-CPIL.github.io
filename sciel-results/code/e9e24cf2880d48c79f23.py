import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 20
    m = 10
    K = 20

    left = list(range(m))
    right = list(range(m, 2 * m))

    def build_hadamard_coloring(adjL, adjR):
        """
        Use a Hadamard-like structure for coloring.
        For K_{m,m} with m=10, we can use a 10x10 matrix where entries are distinct in rows/cols as much as possible.
        Actually, we just need every 4-cycle to be rainbow.
        A 4-cycle in bipartite graph: (l1, r1), (l1, r2), (l2, r1), (l2, r2).
        Colors: c(l1,r1), c(l1,r2), c(l2,r1), c(l2,r2).
        We need these 4 to be distinct.
        
        If we use c(i,j) = (i + j) % K, then for a 4-cycle:
        (i1+j1), (i1+j2), (i2+j1), (i2+j2).
        Are they distinct? 
        i1+j1 == i1+j2 => j1==j2 (no, j1!=j2)
        i1+j1 == i2+j1 => i1==i2 (no, i1!=i2)
        i1+j1 == i2+j2 => i1-i2 == j2-j1. 
        i1+j2 == i2+j1 => i1-i2 == j1-j2.
        So if i1-i2 == j2-j1, then c(l1,r1)==c(l2,r2) and c(l1,r2)==c(l2,r1).
        This creates pairs of equal colors, violating the rainbow condition.
        
        So simple addition doesn't work if the differences match.
        We need a stronger coloring. 
        If we use a Latin Square of order 10 (which exists), then in any 2x2 submatrix, the four entries are not necessarily distinct. 
        Wait, in a Latin Square, each symbol appears once in each row and column.
        In a 2x2 submatrix:
        a b
        c d
        a!=b, a!=c, b!=d, c!=d.
        But a could equal d. If a==d, then we have two a's.
        So a Latin Square does NOT guarantee 4 distinct colors in every 2x2.
        
        To guarantee 4 distinct colors in every 2x2, we need a "4-distinct" matrix.
        This is equivalent to a graph where no 4-cycle has non-distinct colors.
        
        Let's use random colors for the bipartite edges and check.
        Since m=10, the number of 4-cycles is C(10,2)*C(10,2) = 45*45 = 2025.
        We can just assign random colors and verify.
        If a 4-cycle is not rainbow, re-color one of its edges.
        """
        colors = {}
        for i in range(m):
            for j in range(m):
                colors[(i, j)] = random.randint(0, K - 1)
        
        # Verify and fix
        for _ in range(100):
            if time.time() > deadline - 30:
                break
            fixed = False
            for i1 in range(m):
                for i2 in range(i1 + 1, m):
                    for j1 in range(m):
                        for j2 in range(j1 + 1, m):
                            c11 = colors[(i1, j1)]
                            c12 = colors[(i1, j2)]
                            c21 = colors[(i2, j1)]
                            c22 = colors[(i2, j2)]
                            if len({c11, c12, c21, c22}) < 4:
                                # Pick one edge to re-color
                                # Re-color (i1, j1) to a random color not in {c12, c21, c22}
                                forbidden = {c12, c21, c22}
                                available = [c for c in range(K) if c not in forbidden]
                                if available:
                                    colors[(i1, j1)] = random.choice(available)
                                    fixed = True
                                if time.time() > deadline - 30:
                                    break
                        if time.time() > deadline - 30:
                            break
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if not fixed:
                break
        
        return colors

    def build_c4free_bipartite():
        """Build a C4-free bipartite graph greedily."""
        adjL = [set() for _ in range(m)]
        adjR = [set() for _ in range(m)]
        edges = []
        all_edges = [(i, j) for i in range(m) for j in range(m)]
        random.shuffle(all_edges)

        for i, j in all_edges:
            if time.time() > deadline - 30:
                break
            creates_c4 = False
            for i_prime in adjR[j]:
                if adjL[i] & adjL[i_prime]:
                    creates_c4 = True
                    break
            if not creates_c4:
                adjL[i].add(j)
                adjR[j].add(i)
                edges.append([left[i], right[j], 0])  # placeholder color

        return adjL, adjR, edges

    def is_bipartite_valid(adjL, adjR, colors):
        """Check if all 4-cycles in the bipartite graph are rainbow."""
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                # Common neighbors of i1 and i2
                common = adjL[i1] & adjL[i2]
                if len(common) >= 2:
                    j1, j2 = list(common)[:2]
                    c11 = colors.get((i1, j1))
                    c12 = colors.get((i1, j2))
                    c21 = colors.get((i2, j1))
                    c22 = colors.get((i2, j2))
                    if c11 is None or c12 is None or c21 is None or c22 is None:
                        return False
                    if len({c11, c12, c21, c22}) < 4:
                        return False
        return True

    best_edges = None
    best_count = 0

    # Strategy: Build complete bipartite K_{10,10} and color it to be rainbow on all 4-cycles.
    # If we can do that, we have 100 edges. Then we add intra-partition edges.
    
    # First, try to get a valid coloring for K_{10,10}
    adjL_full = [set(range(m)) for _ in range(m)]
    adjR_full = [set(range(m)) for _ in range(m)]
    
    colors_full = build_hadamard_coloring(adjL_full, adjR_full)
    
    if is_bipartite_valid(adjL_full, adjR_full, colors_full):
        bip_edges = []
        for i in range(m):
            for j in range(m):
                c = colors_full[(i, j)]
                bip_edges.append([left[i], right[j], c])
        
        if len(bip_edges) > best_count:
            best_count = len(bip_edges)
            best_edges = bip_edges[:]

    # If the full bipartite graph wasn't valid, fall back to C4-free greedy
    if best_count < 100:
        for attempt in range(20):
            if time.time() > deadline - 40:
                break
            adjL, adjR, bip_edges = build_c4free_bipartite()
            
            if len(bip_edges) <= best_count:
                continue
            
            # Color these edges
            # Since it's C4-free, any coloring is valid for bipartite 4-cycles!
            # Because there are no 4-cycles.
            # So we can use any colors.
            color_idx = 0
            for idx, (u, v, c) in enumerate(bip_edges):
                bip_edges[idx] = [u, v, color_idx % K]
                color_idx += 1
            
            if len(bip_edges) > best_count:
                best_count = len(bip_edges)
                best_edges = bip_edges[:]

    # Now, try to add intra-partition edges to the best solution
    if best_edges is not None:
        # Build adjacency and colors
        adj = {i: set() for i in range(n)}
        colors = {}
        for u, v, c in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            colors[(min(u, v), max(u, v))] = c

        # Candidate intra-partition edges
        candidates = []
        for i in range(m):
            for j in range(i + 1, m):
                candidates.append((left[i], left[j]))
                candidates.append((right[i], right[j]))
        
        random.shuffle(candidates)

        def find_valid_color(u, v):
            """Find a color for (u,v) such that all 4-cycles containing it are rainbow."""
            # A 4-cycle containing (u,v) must be u-w-x-v-u or u-w-v-x-u? No, simple cycle.
            # u-v is an edge. The other 3 edges must form a path from v to u of length 3.
            # v-a-b-u.
            # Edges: (u,v), (v,a), (a,b), (b,u).
            # We need colors[u,v], colors[v,a], colors[a,b], colors[b,u] to be distinct.
            
            forbidden = set()
            for a in adj[v]:
                if a == u:
                    continue
                cva = colors.get((min(u, a), max(u, a))) # Wait, (v,a)
                cva = colors.get((min(v, a), max(v, a)))
                if cva is None:
                    continue
                for b in adj[a]:
                    if b == u or b == v or b == a:
                        continue
                    if u not in adj[b]:
                        continue
                    cab = colors.get((min(a, b), max(a, b)))
                    cbu = colors.get((min(b, u), max(b, u)))
                    if cab is None or cbu is None:
                        continue
                    # The cycle is u-v-a-b-u
                    # Colors: c(u,v), c(v,a), c(a,b), c(b,u)
                    # c(u,v) is what we are choosing.
                    # It must not be in {c(v,a), c(a,b), c(b,u)}
                    forbidden.add(cva)
                    forbidden.add(cab)
                    forbidden.add(cbu)
                    if len(forbidden) == K:
                        return None
            
            for c in range(K):
                if c not in forbidden:
                    return c
            return None

        added = 0
        for u, v in candidates:
            if time.time() > deadline - 10:
                break
            if u == v or v in adj[u]:
                continue
            
            c = find_valid_color(u, v)
            if c is not None:
                adj[u].add(v)
                adj[v].add(u)
                colors[(min(u, v), max(u, v))] = c
                best_edges.append([u, v, c])
                best_count += 1
                added += 1

    # Final check and fallback
    if best_edges is None or best_count == 0:
        # Fallback to a simple valid graph
        # A star is C4-free
        best_edges = [[0, v, v % K] for v in range(1, n)]
        best_count = n - 1

    return best_edges
