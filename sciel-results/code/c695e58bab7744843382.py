import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    q = 5
    # Build PG(2, q) points: 31 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))

    n = len(points)  # should be 31

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [set() for _ in range(n)]
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                edge_set.add((i, j))

    # Count initial 4-cycles in the polarity graph
    def count_4cycles(adj_sets, edge_set_local, n_local):
        """Count number of 4-cycles (subgraphs) in the graph."""
        count = 0
        for i in range(n_local):
            for j in range(i+1, n_local):
                if (i, j) not in edge_set_local and (j, i) not in edge_set_local:
                    continue
                # For each pair of non-adjacent vertices, count common neighbors
                # Actually, a 4-cycle is a set of 4 vertices {a,b,c,d} with edges
                # forming a cycle. We count each 4-cycle once.
                # A 4-cycle can be identified by its two "diagonal" pairs.
                # Alternative: for each pair of vertices (u,v), count common neighbors.
                # Each 4-cycle has exactly 2 pairs of opposite vertices (diagonals).
                # So total 4-cycles = sum over all pairs (u,v) of C(common_neighbors(u,v), 2) / 2
                # Wait, that's not quite right either.
                # Let me think again. A 4-cycle is a subgraph isomorphic to C4.
                # It has 4 vertices and 4 edges. The vertices can be labeled a-b-c-d-a.
                # The diagonals are (a,c) and (b,d). These are non-edges in the C4 subgraph.
                # So for each pair of non-adjacent vertices (u,v) in the graph, the number
                # of 4-cycles where u and v are opposite is C(|N(u) ∩ N(v)|, 2).
                # Each 4-cycle is counted exactly once this way (once for each diagonal pair).
                # So total = sum over all non-adjacent pairs (u,v) of C(common(u,v), 2)
                pass
        
        # Better approach: for each pair of vertices (u,v), let c(u,v) = |N(u) ∩ N(v)|.
        # A 4-cycle containing u and v as adjacent vertices... no, that's complicated.
        # Standard formula: number of C4 subgraphs = (1/2) * sum_{u,v} C(c(u,v), 2)
        # where the sum is over all unordered pairs {u,v}, and c(u,v) = |N(u) ∩ N(v)|.
        # Wait, each C4 has 4 pairs of adjacent vertex pairs and 2 pairs of opposite vertices.
        # Let me use: for each unordered pair {u,v}, let t(u,v) = |N(u) ∩ N(v)|.
        # Each C4 contributes to t(u,v) for its 4 pairs of opposite vertices? No.
        # In a C4 with vertices a,b,c,d (in order), the pairs with common neighbors are:
        # (a,c): common neighbors b,d -> t(a,c) >= 2
        # (b,d): common neighbors a,c -> t(b,d) >= 2
        # (a,b): common neighbors? a's neighbors are b,d; b's neighbors are a,c. Intersection: empty (if no other edges).
        # So only opposite pairs have common neighbors in a pure C4.
        # Thus: number of C4s = (1/2) * sum over all unordered pairs {u,v} of C(t(u,v), 2)
        # because each C4 is counted once for each of its 2 diagonal pairs, and we divide by... 
        # Actually each C4 is counted exactly once in the sum C(t(u,v),2) for the two diagonal pairs.
        # Wait: for a C4 with diagonals (a,c) and (b,d), C(t(a,c),2) includes the pair {b,d} as one of the
        # C(t(a,c),2) choices, and C(t(b,d),2) includes the pair {a,c}. So the C4 is counted twice.
        # Hence: #C4 = (1/2) * sum_{u<v} C(t(u,v), 2)
        
        total = 0
        for u in range(n_local):
            for v in range(u+1, n_local):
                common = len(adj_sets[u] & adj_sets[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    initial_c4 = count_4cycles(adj, edge_set, n)

    # Find the best non-edge to add (minimizing new 4-cycles)
    best_edge = None
    best_cost = float('inf')

    # Precompute adjacency matrix as numpy for speed
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in adj[i]:
            adj_matrix[i, j] = True
            adj_matrix[j, i] = True

    # For each vertex, get neighbor list as numpy array
    adj_lists_np = [np.array(sorted(adj[i]), dtype=int) for i in range(n)]

    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))

    # Compute cost for each non-edge: number of new 4-cycles created
    # When adding edge (i,j), new 4-cycles are those where (i,j) is part of the cycle.
    # A new 4-cycle containing edge (i,j) has the form i-a-b-j-i or i-b-a-j-i (same cycle).
    # So we need a ~ i, b ~ j, a ~ b, a != b, a != j, b != i.
    # Since (i,j) is not an edge, a != j and b != i are automatic (a~i means a!=j only if... 
    # actually a could be j if j~i, but (i,j) is not an edge, so a != j).
    # Similarly b != i.
    # So cost = number of pairs (a,b) with a in N(i), b in N(j), a~b, a!=b.

    # Sort non-edges by a quick heuristic: sum of degrees
    edge_costs = []
    for (i, j) in non_edges:
        # Quick lower bound: min(deg(i), deg(j))
        edge_costs.append((len(adj[i]) + len(adj[j]), i, j))
    edge_costs.sort()

    # Evaluate candidates
    for _, i, j in edge_costs:
        if time.time() > deadline:
            break
        
        # Count exact cost using numpy
        ni = adj_lists_np[i]
        nj = adj_lists_np[j]
        if len(ni) == 0 or len(nj) == 0:
            cost = 0
        else:
            # For each a in N(i), count b in N(j) such that a~b and a!=b
            cost = 0
            for a in ni:
                # Count neighbors of a that are in N(j) and not equal to a
                mask = adj_matrix[a, nj]
                # Exclude b == a
                if a in nj:
                    mask = np.where(nj != a, mask, False)
                cost += mask.sum()
        
        if cost < best_cost:
            best_cost = cost
            best_edge = [i, j]

    if best_edge is not None:
        edges.append(best_edge)
        # Update adjacency
        i, j = best_edge
        adj[i].add(j)
        adj[j].add(i)
        edge_set.add((i, j))
        adj_matrix[i, j] = True
        adj_matrix[j, i] = True

    # Ensure we have exactly 91 edges
    if len(edges) != 91:
        existing = set()
        for e in edges:
            existing.add(tuple(sorted(e)))
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i,j) not in existing:
                    edges.append([i,j])
                    existing.add((i,j))
            if len(edges) >= 91:
                break

    return edges[:91]
