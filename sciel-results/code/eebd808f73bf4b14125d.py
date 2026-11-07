import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _validate(partition, edges, n):
    """Check that partition is a valid biclique partition of edges."""
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    covered = set()
    for bic in partition:
        left, right = bic[0], bic[1]
        if not left or not right:
            return False
        if len(left) != len(set(left)) or len(right) != len(set(right)):
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    return False
                if (u, v) in covered:
                    return False
                covered.add((u, v))
    if covered != edge_set:
        return False
    return True

def _greedy_partition(adj, edges, n, deadline, seed=0):
    """Greedy: repeatedly pick largest valid biclique covering most uncovered edges."""
    rng = random.Random(seed)
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    uncovered = set(edge_set)
    partition = []
    while uncovered:
        # Find candidate bicliques by trying each vertex as a "center"
        best_biclique = None
        best_count = 0
        # Try each vertex as left side anchor
        candidates = list(uncovered)
        rng.shuffle(candidates)
        for (u, v) in candidates[:50]:  # limit candidates for speed
            # Try to build a biclique with u on left, v on right
            # Left side: vertices connected to all of right
            # Right side: vertices connected to all of left
            # Start with left={u}, right={v}
            left = {u}
            right = {v}
            # Expand: add vertices to left that connect to all of right
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in left and w not in right:
                        if all(w in adj[x] for x in right):
                            # Check w connects to all in right
                            left.add(w)
                            changed = True
                for w in range(n):
                    if w not in left and w not in right:
                        if all(w in adj[x] for x in left):
                            right.add(w)
                            changed = True
            # Count uncovered edges covered
            count = 0
            for x in left:
                for y in right:
                    a, b = (x, y) if x < y else (y, x)
                    if (a, b) in uncovered:
                        count += 1
            if count > best_count:
                best_count = count
                best_biclique = (list(left), list(right))
        if best_biclique is None or best_count == 0:
            # Fallback: single edge
            e = next(iter(uncovered))
            best_biclique = [e[0], e[1]]
            best_count = 1
        l, r = best_biclique
        partition.append([l, r])
        for x in l:
            for y in r:
                a, b = (x, y) if x < y else (y, x)
                uncovered.discard((a, b))
        if time.time() > deadline:
            break
    return partition

def _try_merge(partition, adj, edges, n, deadline):
    """Try to merge pairs of bicliques to reduce count."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(partition)
        for i in range(m):
            for j in range(i + 1, m):
                if time.time() > deadline:
                    return partition
                li, ri = partition[i][0], partition[i][1]
                lj, rj = partition[j][0], partition[j][1]
                # Try all 4 orientations of merging
                for l1, r1, l2, r2 in [(li, ri, lj, rj), (li, ri, rj, lj),
                                         (ri, li, lj, rj), (ri, li, rj, lj)]:
                    # Check: all cross pairs must be edges
                    valid = True
                    for x in l1:
                        for y in r1:
                            if y not in adj[x]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    for x in l2:
                        for y in r2:
                            if y not in adj[x]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Check no overlap in covered edges
                    # New biclique covers all pairs in l1×r1 ∪ l2×r2
                    # Old covered: li×ri ∪ lj×rj
                    # We need: new covers exactly old (no extra edges, no missing)
                    old_edges = set()
                    for x in li:
                        for y in ri:
                            a, b = (x, y) if x < y else (y, x)
                            old_edges.add((a, b))
                    for x in lj:
                        for y in rj:
                            a, b = (x, y) if x < y else (y, x)
                            old_edges.add((a, b))
                    new_edges = set()
                    for x in l1:
                        for y in r1:
                            a, b = (x, y) if x < y else (y, x)
                            new_edges.add((a, b))
                    for x in l2:
                        for y in r2:
                            a, b = (x, y) if x < y else (y, x)
                            new_edges.add((a, b))
                    if new_edges == old_edges:
                        # Merge successful
                        new_biclique = [list(l1), list(r2)]
                        # Remove i and j, add new
                        partition.pop(j)
                        partition.pop(i)
                        partition.append(new_biclique)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return partition

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    best_partition = None
    best_count = float('inf')
    
    # Multiple restarts with different seeds
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        part = _greedy_partition(adj, edges, n, deadline - 3, seed=seed)
        if not _validate(part, edges, n):
            continue
        # Local search: merge
        part = _try_merge(part, adj, edges, n, deadline - 2)
        if _validate(part, edges, n):
            cnt = len(part)
            if cnt < best_count:
                best_count = cnt
                best_partition = part
                if best_count <= 5:
                    break
    
    if best_partition is None:
        # Fallback: star partition
        best_partition = [[[u], [v for a, v in edges if a == u]] for u in range(n) if any(a == u for a, v in edges)]
    
    # Convert to proper Python types
    result = []
    for bic in best_partition:
        left = list(bic[0])
        right = list(bic[1])
        result.append([left, right])
    return result
