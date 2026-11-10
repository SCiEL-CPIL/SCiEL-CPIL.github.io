import math
import time
import numpy as np

_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_INV  = [0, 1, 2, 3]


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix (numpy-vectorised)."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    L = len(sequences[labels[0]])
    # Encode all sequences as an int array of shape (n, L)
    mat = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # fast path: use a precomputed table
        mat[i] = np.fromiter((_BASE[c] for c in s), dtype=np.int8, count=L)
    # Pairwise hamming distances
    # diff[i,j] = number of positions where row i != row j
    # Use broadcasting: (n,1,L) vs (1,n,L) -> (n,n,L) may be large; do in chunks
    D = np.zeros((n, n), dtype=np.float64)
    # chunk over rows to limit memory
    chunk = max(1, min(n, 64))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        a = mat[start:end, :, None]   # (c, L, 1)
        b = mat[:, None, :]           # (1, n, L)
        diff = (a != b)               # (c, n, L)
        h = diff.sum(axis=2)          # (c, n)
        D[start:end, :] = h
    # Symmetrise and zero diagonal
    D = np.maximum(D, D.T)
    np.fill_diagonal(D, 0.0)
    # JC correction
    p = D / max(L, 1)
    # clip to avoid log(0)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log1p(-4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _nj_build(D):
    """
    Neighbor-Joining. Returns (leaf_order, internal_nodes) where:
      - leaf_order: list of n original leaf indices (0..n-1) in the order they appear as leaves
      - internal_nodes: list of (a, b, length) triples for each internal node created,
        where a and b are node ids in the working node space.
    We'll instead return a compact tree representation:
      nodes: list where entries 0..n-1 are leaves (value = leaf index),
             and entries n.. are internal nodes represented as (left, right) tuples.
      parent: list of (node, parent) edges? Simpler: return the adjacency.
    """
    n = D.shape[0]
    if n <= 2:
        return list(range(n)), []
    D = D.copy()
    active = list(range(n))
    # node representation: for leaves it's the int leaf index; for internal it's a tuple (l, r)
    node_rep = list(range(n))
    next_id = n
    edges = []  # list of (u, v) in node-id space (will be used to build adjacency later)

    while len(active) > 3:
        m = len(active)
        # r_i = sum_j D[i][j]
        sub = D[np.ix_(active, active)]
        r = sub.sum(axis=1)
        # B_ij = (m-2)*D_ij - r_i - r_j
        best_val = float('inf')
        best_i = best_j = -1
        for a in range(m):
            i = active[a]
            for b in range(a + 1, m):
                j = active[b]
                val = (m - 2) * sub[a, b] - r[a] - r[b]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        i, j = best_i, best_j
        # new node
        new_id = next_id
        next_id += 1
        node_rep.append((i, j))
        edges.append((new_id, i))
        edges.append((new_id, j))
        # compute new distances
        D_new = np.zeros((new_id + 1, new_id + 1), dtype=np.float64)
        # copy old
        D_new[:new_id, :new_id] = D[:new_id, :new_id]
        for k in active:
            if k == i or k == j:
                continue
            d_ik = D[i, k]
            d_jk = D[j, k]
            d_ij = D[i, j]
            d_new_k = 0.5 * (d_ik + d_jk - d_ij)
            D_new[new_id, k] = d_new_k
            D_new[k, new_id] = d_new_k
        D = D_new
        active.remove(i)
        active.remove(j)
        active.append(new_id)

    # final 3 nodes form a star
    a, b, c = active
    root = next_id
    next_id += 1
    node_rep.append((a, b))
    node_rep.append((root, c))  # not quite; let's just record the three-way
    # For simplicity, we'll build the final tree by adding a central node connected to a,b,c
    # But NJ typically returns a binary tree; the last step is a 3-way star.
    # We'll represent it as: new internal node X with children a and (b,c) -- arbitrary rooting.
    # For unrooted topology, any binary representation of the 3-star works.
    # Let's make X = (a, b), then Y = (X, c).
    x = next_id; next_id += 1
    y = next_id; next_id += 1
    node_rep.append((a, b))   # x
    node_rep.append((x, c))   # y
    edges.append((x, a)); edges.append((x, b)); edges.append((y, x)); edges.append((y, c))

    return node_rep, edges


def _build_adjacency(node_rep, edges, n_leaves):
    """
    Build adjacency list for the unrooted tree.
    node_rep: list of length N where entries < n_leaves are leaves, else (l, r) tuples.
    We'll return:
      adj: dict node -> list of (neighbor, edge_length_placeholder)
      is_leaf: list of bool
      leaf_to_node: dict leaf_index -> node_id
    """
    N = len(node_rep)
    adj = {i: [] for i in range(N)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    is_leaf = [i < n_leaves for i in range(N)]
    leaf_to_node = {i: i for i in range(n_leaves)}
    return adj, is_leaf, leaf_to_node, N


def _leaf_set(node, adj, is_leaf, n_leaves):
    """Return the set of leaf indices in the subtree rooted at node (away from parent)."""
    # We'll do a BFS/DFS avoiding the parent.
    pass


def _nni_refine(sequences, labels, node_rep, edges, deadline):
    """
    Perform NNI refinement. We work on the tree represented by node_rep and edges.
    For each internal edge, test the 3 possible topologies and keep the best.
    We use a simple scoring: sum of pairwise distances (or better, a likelihood proxy).
    Since computing full JC likelihood is expensive, we use a fast proxy:
      score = sum over all pairs of leaves of D_ij (lower is better) -- but this is invariant to topology.
    Instead, we use the standard NNI criterion based on the four subtrees.
    
    Given an internal edge separating the tree into two parts, with the four subtrees A, B, C, D
    (A and B on one side, C and D on the other, with the edge connecting (A,B) to (C,D)),
    the three topologies are:
      T1: (A,B)(C,D)  -- current
      T2: (A,C)(B,D)
      T3: (A,D)(B,C)
    The NNI criterion (for NJ-like scoring) is:
      For T1: S1 = sum_{a in A, c in C} d_ac + sum_{a in A, d in D} d_ad + sum_{b in B, c in C} d_bc + sum_{b in B, d in D} d_bd
      For T2: S2 = sum_{a in A, b in B} d_ab + sum_{a in A, d in D} d_ad + sum_{c in C, b in B} d_cb + sum_{c in C, d in D} d_cd
      For T3: S3 = sum_{a in A, b in B} d_ab + sum_{a in A, c in C} d_ac + sum_{d in D, b in B} d_db + sum_{d in D, c in C} d_dc
    We pick the topology with minimum S.
    """
    start = time.time()
    n_leaves = len(labels)
    N = len(node_rep)
    adj, is_leaf, leaf_to_node, N = _build_adjacency(node_rep, edges, n_leaves)
    
    # Precompute leaf membership for each node: leaf_sets[node] = frozenset of leaf indices
    # Do a DFS from each node, but that's O(N^2). Instead, do a post-order traversal.
    # Since the tree is unrooted, we'll root it arbitrarily at node 0 and do a DFS.
    # For each node, compute the set of leaves in its subtree (away from parent).
    # Then for an edge (u, v), the two sides are leaf_sets[u] (when parent is v) and leaf_sets[v] (when parent is u).
    
    # Root at 0
    parent = {i: -1 for i in range(N)}
    order = []
    stack = [0]
    parent[0] = -2  # mark as root
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                stack.append(v)
    
    # Post-order: process in reverse order
    leaf_sets = [None] * N
    for u in reversed(order):
        if is_leaf[u]:
            leaf_sets[u] = frozenset([u])  # leaf node id == leaf index
        else:
            s = frozenset()
            for v in adj[u]:
                if v != parent[u]:
                    s = s | leaf_sets[v]
            leaf_sets[u] = s
    
    # Now for each edge (u, v) where both are internal, compute the NNI criterion.
    # We need to identify the four subtrees.
    # For edge (u, v): 
    #   Side U: the component containing u when edge is removed. This is leaf_sets[u] if parent[u] == v, 
    #           else (all leaves) - leaf_sets[v] if parent[v] == u.
    #   Actually, let's define:
    #   If parent[u] == v: side_u = leaf_sets[u], side_v = all_leaves - side_u
    #   If parent[v] == u: side_v = leaf_sets[v], side_u = all_leaves - side_v
    #   The four subtrees are the children of u (away from v) and children of v (away from u).
    
    all_leaves = frozenset(range(n_leaves))
    
    # Build a list of candidate edges to test
    # An internal edge is one where both endpoints are internal nodes.
    # For each such edge, we test the 3 topologies.
    
    # We'll represent the tree as a set of edges and a mapping from node to its children (in the rooted tree).
    # For NNI, we need to:
    #   1. Identify the four subtrees A, B, C, D
    #   2. Compute S1, S2, S3
    #   3. If a better topology exists, modify the tree.
    
    # Precompute pairwise distance sums for all pairs of disjoint leaf sets? That's expensive.
    # Instead, for each edge, we compute the four sums on the fly using the distance matrix.
    
    # Get the distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # For efficiency, precompute for each leaf set a vector of distances to all leaves? 
    # That's O(n^2) memory, which is fine for n <= 32.
    
    # Let's precompute: for each leaf index i, D[i, :] is already available.
    
    # For a set S of leaves, and a set T of leaves, sum_{s in S, t in T} D[s,t]
    # We can compute this as: sum_{s in S} (D[s, :][T].sum())
    # Using numpy: D[np.ix_(S_list, T_list)].sum()
    
    # To speed up, precompute for each leaf set a numpy array of sums to each leaf.
    # But leaf sets change as we modify the tree, so we can't precompute globally.
    
    # For now, do it on the fly. With n <= 32, this should be fast enough.
    
    improved = True
    while improved and (time.time() - start) < deadline - 5:
        improved = False
        # Rebuild parent and leaf_sets (since tree may have changed)
        parent = {i: -1 for i in range(N)}
        order = []
        stack = [0]
        parent[0] = -2
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if v != parent[u]:
                    parent[v] = u
                    stack.append(v)
        
        leaf_sets = [None] * N
        for u in reversed(order):
            if is_leaf[u]:
                leaf_sets[u] = frozenset([u])
            else:
                s = frozenset()
                for v in adj[u]:
                    if v != parent[u]:
                        s = s | leaf_sets[v]
                leaf_sets[u] = s
        
        # Test each internal edge
        tested = set()
        for u in range(N):
            if is_leaf[u]:
                continue
            for v in adj[u]:
                if is_leaf[v]:
                    continue
                if (u, v) in tested or (v, u) in tested:
                    continue
                tested.add((u, v))
                tested.add((v, u))
                
                # Determine the four subtrees
                # Children of u away from v
                children_u = [w for w in adj[u] if w != v]
                # Children of v away from u
                children_v = [w for w in adj[v] if w != u]
                
                # A, B are the leaf sets of children_u (should be 2 for binary)
                # C, D are the leaf sets of children_v (should be 2 for binary)
                if len(children_u) != 2 or len(children_v) != 2:
                    continue
                
                A = leaf_sets[children_u[0]]
                B = leaf_sets[children_u[1]]
                C = leaf_sets[children_v[0]]
                D_set = leaf_sets[children_v[1]]
                
                # Compute S1, S2, S3
                # S1 = sum(A,C) + sum(A,D) + sum(B,C) + sum(B,D)
                # S2 = sum(A,B) + sum(A,D) + sum(B,C) + sum(C,D)
                # S3 = sum(A,B) + sum(A,C) + sum(B,D) + sum(C,D)
                
                def sum_pairs(S1, S2):
                    if not S1 or not S2:
                        return 0.0
                    l1 = list(S1)
                    l2 = list(S2)
                    return D[np.ix_(l1, l2)].sum()
                
                sAC = sum_pairs(A, C)
                sAD = sum_pairs(A, D_set)
                sBC = sum_pairs(B, C)
                sBD = sum_pairs(B, D_set)
                sAB = sum_pairs(A, B)
                sCD = sum_pairs(C, D_set)
                
                S1 = sAC + sAD + sBC + sBD
                S2 = sAB + sAD + sBC + sCD
                S3 = sAB + sAC + sBD + sCD
                
                best = min(S1, S2, S3)
                if best == S1:
                    continue  # current is best
                
                # Modify the tree: remove edge (u, v) and rewire
                # If S2 is best: new topology is (A,C)(B,D)
                #   u's children become A and C, v's children become B and D
                #   But we need to create new internal nodes or modify existing ones.
                #   Simplest: replace the edge (u, v) with two new edges:
                #     u -- new_node1 -- v  where new_node1 connects A and C? 
                #   Actually, the standard NNI move:
                #     Current: u has children A, B; v has children C, D; edge u-v
                #     New (S2): u has children A, C; v has children B, D; edge u-v
                #     So we just need to swap: remove edge u-v, and reassign children.
                #     But u and v are still connected by an edge. We just change which subtrees hang off each.
                
                # For S2: u gets A and C, v gets B and D
                # For S3: u gets A and D, v gets B and C
                
                if best == S2:
                    # u: A, C; v: B, D
                    # Remove old edges
                    adj[u] = [children_u[0], children_v[0], v]  # A, C, v
                    adj[v] = [children_u[1], children_v[1], u]  # B, D, u
                    # Update node_rep
                    node_rep[u] = (children_u[0], children_v[0])
                    node_rep[v] = (children_u[1], children_v[1])
                else:  # S3
                    # u: A, D; v: B, C
                    adj[u] = [children_u[0], children_v[1], v]  # A, D, v
                    adj[v] = [children_u[1], children_v[0], u]  # B, C, u
                    node_rep[u] = (children_u[0], children_v[1])
                    node_rep[v] = (children_u[1], children_v[0])
                
                improved = True
                break  # restart the loop after a change
    
    return node_rep, adj, is_leaf


def _tree_to_newick(node_rep, adj, is_leaf, n_leaves, labels):
    """Convert the tree to a Newick string."""
    N = len(node_rep)
    # Root at 0
    parent = {i: -1 for i in range(N)}
    stack = [0]
    parent[0] = -2
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                stack.append(v)
    
    def to_newick(u):
        if is_leaf[u]:
            return labels[u]
        children = [v for v in adj[u] if v != parent[u]]
        if not children:
            return labels[u]
        return "(" + ",".join(to_newick(c) for c in children) + ")"
    
    return to_newick(0) + ";"


def reconstruct(sequences):
    start_time = time.time()
    deadline = 265.0  # safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa == 1:
        return f"({labels[0]});"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Step 1: Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Step 2: Build initial tree with NJ
    node_rep, edges = _nj_build(D)
    
    # Step 3: NNI refinement
    if time.time() - start_time < deadline - 10:
        node_rep, adj, is_leaf = _nni_refine(sequences, labels, node_rep, edges, deadline)
        return _tree_to_newick(node_rep, adj, is_leaf, n_taxa, labels)
    else:
        # Fallback: return the NJ tree
        adj, is_leaf, leaf_to_node, N = _build_adjacency(node_rep, edges, n_taxa)
        return _tree_to_newick(node_rep, adj, is_leaf, n_taxa, labels)
