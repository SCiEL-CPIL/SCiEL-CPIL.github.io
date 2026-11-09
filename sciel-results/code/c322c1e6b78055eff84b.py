import time
import itertools
import random

# EVOLVE-BLOCK-START

def _has_q3(adj):
    """
    Check if a graph with adjacency list 'adj' (list of sets) contains Q3 as a subgraph.
    Q3 is the 3-cube: 8 vertices, 12 edges, 3-regular, bipartite.
    We look for any set of 8 vertices that induces a supergraph of Q3.
    Since Q3 is 3-regular, we need 8 vertices where each has degree >= 3 within the set,
    and the induced subgraph is bipartite and isomorphic to Q3.
    
    A more efficient approach: Q3 has 12 edges. We can try all 8-subsets and check if
    they contain Q3. But C(16,8) = 12870, which is manageable.
    
    For each 8-subset, we check if the induced subgraph contains Q3 as a subgraph.
    Q3 is 3-regular bipartite with parts of size 4.
    
    Actually, let's use a smarter check: Q3 is characterized by being 3-regular, bipartite,
    with 8 vertices and 12 edges, and having girth 4. But checking isomorphism is complex.
    
    Alternative: enumerate all possible Q3 embeddings. Q3 has 8! / (symmetries) labelings.
    But a simpler approach: for each 8-subset S, check if there exists a bipartition of S
    into two sets A, B of size 4 each, such that every vertex in A is connected to at
    least 3 vertices in B, and every vertex in B is connected to at least 3 vertices in A,
    and the bipartite graph is isomorphic to Q3.
    
    Q3 as a bipartite graph between A={a0,a1,a2,a3} and B={b0,b1,b2,b3}:
    Each vertex in A connects to 3 vertices in B, and each vertex in B connects to 3 vertices in A.
    Specifically, if we label A as {0,1,2,3} and B as {4,5,6,7}, the edges of Q3 are:
    0-4, 0-5, 0-6
    1-4, 1-5, 1-7
    2-4, 2-6, 2-7
    3-5, 3-6, 3-7
    
    So the bipartite adjacency (as bitmasks for B-side neighbors of each A-vertex):
    a0: {4,5,6} -> 0b0111 = 7
    a1: {4,5,7} -> 0b1011 = 11
    a2: {4,6,7} -> 0b1101 = 13
    a3: {5,6,7} -> 0b1110 = 14
    
    We can check: for a given bipartition (A,B) of an 8-set, check if the bipartite
    subgraph is isomorphic to this pattern.
    """
    n = len(adj)
    # Precompute all 8-subsets
    for S in itertools.combinations(range(n), 8):
        S_set = set(S)
        # Try all ways to split S into two sets of size 4
        # C(8,4) = 70 partitions, but each partition counted twice, so 35 unique
        A_list = list(S)
        for A_idx in itertools.combinations(range(8), 4):
            A = [A_list[i] for i in A_idx]
            B = [A_list[i] for i in range(8) if i not in set(A_idx)]
            # Check bipartite subgraph between A and B
            # Build adjacency from A to B
            # For each a in A, find which b in B are neighbors
            A_to_B = []
            valid = True
            for a in A:
                nbrs = adj[a] & S_set
                # Only consider neighbors in B
                nbrs_in_B = [b for b in B if b in nbrs]
                if len(nbrs_in_B) != 3:
                    valid = False
                    break
                A_to_B.append(nbrs_in_B)
            if not valid:
                continue
            # For each b in B, find which a in A are neighbors
            B_to_A = []
            for b in B:
                nbrs = adj[b] & S_set
                nbrs_in_A = [a for a in A if a in nbrs]
                if len(nbrs_in_A) != 3:
                    valid = False
                    break
                B_to_A.append(nbrs_in_A)
            if not valid:
                continue
            # Now check if this bipartite graph is isomorphic to Q3
            # Q3 pattern: each vertex has degree 3, and the structure is specific.
            # We can check by trying all permutations of B and see if it matches the pattern.
            # The pattern for A-vertices (in order a0,a1,a2,a3) connecting to B-vertices:
            # a0 -> {b0,b1,b2}, a1 -> {b0,b1,b3}, a2 -> {b0,b2,b3}, a3 -> {b1,b2,b3}
            # So if we label B as [b0,b1,b2,b3], the neighbor sets for A are:
            # {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}
            # Check if there exists a permutation of B such that the neighbor sets match.
            target_patterns = [
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3)
            ]
            for perm in itertools.permutations(range(4)):
                # perm[i] gives the original index in B for the i-th position
                # Check if A_to_B matches target_patterns under this permutation
                match = True
                for i in range(4):
                    actual_nbrs = set(A_to_B[i])
                    expected_nbrs = set(B[target_patterns[i][j]] for j in range(3))
                    if actual_nbrs != expected_nbrs:
                        match = False
                        break
                if match:
                    return True
    return False


def _edge_creates_q3(adj, u, v):
    """Check if adding edge (u,v) to graph with adjacency 'adj' creates a Q3."""
    adj[u].add(v)
    adj[v].add(u)
    result = _has_q3(adj)
    adj[u].discard(v)
    adj[v].discard(u)
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n = 16
    # Start with K3,13 (the seed)
    adj = [set() for _ in range(n)]
    edges = []
    for a in range(3):
        for b in range(3, 16):
            adj[a].add(b)
            adj[b].add(a)
            edges.append([a, b])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try to add edges greedily
    # Generate all possible edges not already present
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                all_pairs.append((i, j))
    
    random.seed(42)
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if _edge_creates_q3(adj, u, v):
            continue
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    # Try more random restarts with different orderings
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        adj = [set() for _ in range(n)]
        edges = []
        for a in range(3):
            for b in range(3, 16):
                adj[a].add(b)
                adj[b].add(a)
                edges.append([a, b])
        
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    all_pairs.append((i, j))
        
        random.seed(attempt)
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if _edge_creates_q3(adj, u, v):
                continue
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    return best_edges

# EVOLVE-BLOCK-END
