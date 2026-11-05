"""Phylogenetic topology recovery: NJ + SPR with quartet-based scoring."""
import time
import math
import random

import numpy as np

_START_TIME = None
_DEADLINE = 260.0  # safety margin under 270s wall-clock


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distance matrix using NumPy."""
    n = len(labels)
    # Encode sequences as integer arrays
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = []
    for lbl in labels:
        s = sequences[lbl]
        seqs.append(np.array([enc[c] for c in s], dtype=np.int8))
    seqs = np.array(seqs)  # shape (n, L)
    L = seqs.shape[1]
    
    # Pairwise mismatch counts
    # For memory efficiency with n<=32, L<=2000: n*n*L = 32*32*2000 = 2M, fine
    diff = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            diff[i, j] = np.sum(seqs[i] != seqs[j])
            diff[j, i] = diff[i, j]
    
    p = diff.astype(np.float64) / L
    # JC correction: -0.75 * ln(1 - 4p/3)
    np.clip(p, 0.0, 0.7499, out=p)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(dist, labels):
    """
    Neighbour-Joining. Returns (adj, edge_lengths, node_count).
    adj: list of sets, adj[i] = set of neighbors of node i
    edge_lengths: dict (i,j) -> length
    node_count: total number of nodes
    """
    n = len(labels)
    if n <= 3:
        # Trivial
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj[i].add(j)
                adj[j].add(i)
        return adj, {}, n
    
    # dist is n x n numpy array
    nodes = list(range(n))
    d = dist.copy()  # mutable copy, will grow
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        # r[i] = sum of distances from node i to all other nodes
        r = np.zeros(m)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    r[i] += d[a, b]
        
        best_crit = None
        best_pair = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[a, b] - r[i] - r[j]
                if best_crit is None or crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        
        a, b = best_pair
        # Branch lengths
        da = 0.5 * d[a, b]
        db = 0.5 * d[a, b]
        for x in nodes:
            if x != a and x != b:
                da += 0.5 * (d[a, x] - d[b, x])
                db += 0.5 * (d[b, x] - d[a, x])
        da = max(da, 0.0)
        db = max(db, 0.0)
        
        new = next_id
        next_id += 1
        rest = [x for x in nodes if x not in (a, b)]
        
        # Update distance matrix
        if new >= d.shape[0]:
            # Expand d
            new_d = np.zeros((next_id + 1, next_id + 1))
            new_d[:d.shape[0], :d.shape[1]] = d
            d = new_d
        
        for x in rest:
            d[new, x] = d[x, new] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        
        nodes = rest + [new]
        trees[new] = (a, b, da, db)  # store structure
    
    # Build adjacency and edge lengths from trees
    adj = [set() for _ in range(next_id)]
    edge_lengths = {}
    
    def build(node):
        if isinstance(node, str):
            return
        a, b, da, db = node
        adj[node].add(a)
        adj[a].add(node)
        adj[node].add(b)
        adj[b].add(node)
        edge_lengths[(node, a)] = da
        edge_lengths[(a, node)] = da
        edge_lengths[(node, b)] = db
        edge_lengths[(b, node)] = db
        build(a)
        build(b)
    
    # The final 3 nodes form a triangle
    # Find the root of the tree structure
    # Actually, let's track the parent-child relationships
    # The last 'new' node created is the top
    # Let me redo: track the full tree structure
    
    # Rebuild: the trees dict has internal nodes mapping to (a, b, da, db)
    # The top node is the last one created
    top = next_id - 1
    build(top)
    
    # The 3 remaining nodes form a triangle - connect them
    # Actually in NJ, when 3 nodes remain, they form a triangle
    # We need to add edges between them
    remaining = nodes  # 3 nodes
    # Compute edge lengths for the triangle
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = remaining[i], remaining[j]
            # The distance between a and b in the tree
            # In the final triangle, we need to figure out the edge lengths
            # Actually, NJ just connects the last 3 nodes as a triangle
            # But we need edge lengths. Let's use the d matrix.
            if a not in adj[b]:
                adj[a].add(b)
                adj[b].add(a)
                # Edge length: we need to compute
                # For the final triangle, the edge lengths are determined by the distances
                # d[a,b] is the distance between a and b
                # But in a tree with 3 leaves, the edge lengths are:
                # Actually, the 3 nodes might be leaves or internal nodes
                # Let's just set edge length to 0.5 * d[a,b] for simplicity
                # This is approximate but topology is what matters
                edge_lengths[(a, b)] = 0.5 * d[a, b]
                edge_lengths[(b, a)] = 0.5 * d[a, b]
    
    return adj, edge_lengths, next_id


def _adj_to_splits(adj, n_leaves, labels):
    """
    Given adjacency list, compute all non-trivial splits.
    A split is a partition of the leaf set.
    Returns a set of frozensets (each representing one side of a split).
    """
    splits = set()
    leaf_set = set(range(n_leaves))
    
    for edge_node in range(len(adj)):
        # For each edge, remove it and see the two components
        # Find the edge by looking at each node's neighbors
        # Actually, let's iterate over all edges
        pass
    
    # Better: for each edge (u,v), remove it, find component containing u
    # Use BFS
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:  # each edge once
                # Remove edge u-v, find component of u
                visited = set()
                stack = [u]
                visited.add(u)
                while stack:
                    node = stack.pop()
                    for nb in adj[node]:
                        if nb != v and nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
                # Leaf set in this component
                leaf_part = visited & leaf_set
                if 0 < len(leaf_part) < n_leaves:
                    # Non-trivial split
                    # Canonical form: the side with smaller first element
                    sorted_part = sorted(leaf_part)
                    if sorted_part[0] < n_leaves - 1 - sorted_part[-1]:
                        # Use the smaller side
                        pass
                    # Just use frozenset of the part
                    splits.add(frozenset(leaf_part))
    
    return splits


def _quartet_score(adj, dist, n_leaves, labels):
    """
    Score a tree by the fraction of quartets it resolves correctly.
    For each set of 4 leaves, determine which of the 3 possible quartet topologies
    the tree implies, and compare with the one supported by the distance matrix.
    """
    from itertools import combinations
    
    n = n_leaves
    if n < 4:
        return 1.0
    
    # For each quartet of leaves, determine the tree's topology
    # and the distance-supported topology
    
    # First, build a function to get the path between two leaves in the tree
    # and determine which split separates them
    
    # For a quartet (a,b,c,d), the tree implies one of:
    # ab|cd, ac|bd, ad|bc
    # The tree's topology is determined by which pairs are separated by the same internal edge
    
    # Method: for leaves a,b,c,d, find the minimal subtree
    # The quartet topology is: if the path a-c and b-d share an internal edge, then ab|cd
    # More precisely: the quartet is ab|cd if the path from a to c and from b to d
    # cross (share an edge in the minimal subtree)
    
    # Simpler: for 4 leaves, the tree topology is determined by which pair of leaves
    # are "closest" in the sense that the path between them doesn't cross the others
    
    # Let me use: for quartet (a,b,c,d), compute the 6 pairwise distances in the tree
    # (using edge lengths). The quartet topology is the one where the sum of the
    # two opposite edges is smallest.
    
    # But we need tree path lengths. Let's compute them.
    
    # Actually, a cleaner method:
    # For 4 leaves a,b,c,d, the tree implies topology X|Y where X and Y are the two
    # pairs that are separated by the central edge.
    # We can find this by: remove each internal edge, see which 4-leaf splits it induces.
    
    # For efficiency, let me compute: for each pair of leaves, the set of internal edges
    # on their path. Then for quartet (a,b,c,d):
    # - If the path a-c and path b-d share an internal edge, and path a-b and c-d don't
    #   (and similarly for the other pairing), then it's ab|cd.
    
    # This is getting complex. Let me use a simpler approach:
    # For each quartet, determine the tree topology by checking which split is induced.
    
    # Precompute: for each leaf, the set of internal nodes on its "path" to some root
    # Actually, let me just do BFS for each quartet.
    
    # For speed, let me precompute all-pairs tree distances (path lengths)
    # n <= 32, so n*(n-1)/2 = 496 pairs, each BFS is O(n+E) which is fine
    
    def tree_path_length(u, v):
        """BFS in tree to find path length between u and v."""
        if u == v:
            return 0.0
        visited = {u: 0.0}
        stack = [u]
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if nb not in visited:
                    w = edge_lengths.get((node, nb), 0.0)
                    visited[nb] = visited[node] + w
                    if nb == v:
                        return visited[nb]
                    stack.append(nb)
        return 0.0  # shouldn't happen in a tree
    
    # Precompute all-pairs tree distances
    tree_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = tree_path_length(i, j)
            tree_dist[i, j] = d
            tree_dist[j, i] = d
    
    correct = 0
    total = 0
    
    for a, b, c, d in combinations(range(n), 4):
        # Tree topology: which of ab|cd, ac|bd, ad|bc?
        # The quartet topology is determined by which pair of opposite edges has the
        # smallest sum. In a tree with 4 leaves:
        # d(a,b) + d(c,d) vs d(a,c) + d(b,d) vs d(a,d) + b,c)
        # The smallest sum indicates the quartet topology.
        
        s1 = tree_dist[a, b] + tree_dist[c, d]  # ab|cd
        s2 = tree_dist[a, c] + tree_dist[b, d]  # ac|bd
        s3 = tree_dist[a, d] + tree_dist[b, c]  # ad|bc
        
        min_s = min(s1, s2, s3)
        if s1 == min_s:
            tree_topo = (frozenset([a, b]), frozenset([c, d]))
        elif s2 == min_s:
            tree_topo = (frozenset([a, c]), frozenset([b, d]))
        else:
            tree_topo = (frozenset([a, d]), frozenset([b, c]))
        
        # Distance-supported topology: use the observed distance matrix
        d1 = dist[a, b] + dist[c, d]
        d2 = dist[a, c] + dist[b, d]
        d3 = dist[a, d] + dist[b, c]
        
        min_d = min(d1, d2, d3)
        if d1 == min_d:
            data_topo = (frozenset([a, b]), frozenset([c, d]))
        elif d2 == min_d:
            data_topo = (frozenset([a, c]), frozenset([b, d]))
        else:
            data_topo = (frozenset([a, d]), frozenset([b, c]))
        
        total += 1
        if tree_topo == data_topo:
            correct += 1
    
    return correct / total if total > 0 else 1.0


def _spr_refine(adj, edge_lengths, dist, n_leaves, labels, max_rounds=3):
    """
    Subtree Prune and Regraft local search.
    For each internal edge, try all regraftings, keep the one that improves quartet score.
    """
    n = n_leaves
    if n < 4:
        return adj, edge_lengths
    
    current_score = _quartet_score(adj, edge_lengths, n_leaves, labels)
    
    for _round in range(max_rounds):
        if _time_exceeded():
            break
        
        improved = False
        # Try SPR moves on each internal edge
        # An internal edge is one where both endpoints have degree > 1 (or one is leaf, one is internal with degree > 1)
        # Actually, in an unrooted tree with n leaves, internal nodes have degree >= 3
        # An edge is "internal" if removing it splits the tree into two parts, both containing at least 2 leaves
        
        for u in range(len(adj)):
            for v in adj[u]:
                if u >= v:
                    continue
                
                # Check if this is a valid edge to prune
                # Remove edge u-v, find the two components
                # Component containing u (without v)
                comp_u = set()
                stack = [u]
                comp_u.add(u)
                while stack:
                    node = stack.pop()
                    for nb in adj[node]:
                        if nb != v and nb not in comp_u:
                            comp_u.add(nb)
                            stack.append(nb)
                
                # Leaf count in comp_u
                leaf_count_u = len(comp_u & set(range(n)))
                if leaf_count_u < 2 or leaf_count_u > n - 2:
                    continue  # not a valid internal edge
                
                # Prune: remove edge u-v
                # The subtree at v (the side with fewer leaves, say) is pruned
                # Let's prune the side with fewer leaves for efficiency
                # Actually, let's just try regrafting the u-side to all other edges
                
                # Save current state
                adj_save = [set(s) for s in adj]
                el_save = dict(edge_lengths)
                
                # Remove edge u-v
                adj[u].discard(v)
                adj[v].discard(u)
                edge_lengths.pop((u, v), None)
                edge_lengths.pop((v, u), None)
                
                # The pruned subtree is the component containing u (without v)
                # Find the "attachment point" which is u
                # Try regrafting u to every other edge in the remaining tree
                
                best_new_score = current_score
                best_regraft = None
                
                # All edges in the remaining tree (the v-side)
                # Plus the possibility of creating a new edge
                for x in range(len(adj)):
                    for y in adj[x]:
                        if x >= y:
                            continue
                        # Can we regraft u to edge x-y?
                        # x and y must be in the v-side component
                        # Check: x and y should not be in comp_u
                        if x in comp_u or y in comp_u:
                            continue
                        # Also, x and y should be in the same component (the v-side)
                        # They are, since we only removed u-v
                        
                        # Try regrafting: remove edge x-y, add x-u, y-u (or similar)
                        # Actually, regrafting means: remove edge x-y, add edge x-u and y-u
                        # This creates a new internal node? No, in unrooted trees:
                        # Prune subtree at u (remove edge u-v), then regraft by
                        # removing edge x-y and adding edges x-u and y-u
                        # But this changes the degree of x and y
                        # Actually, standard SPR: 
                        # 1. Remove edge u-v, the subtree rooted at u is detached
                        # 2. For each edge x-y in the remaining tree, remove it
                        # 3. Add edges x-u and y-u (u becomes a new internal node? No)
                        # Actually, u is already a node. We add edges from u to x and u to y?
                        # No, that would make u have degree 3 (plus its original edges)
                        
                        # Let me think more carefully:
                        # Original: ... - u - v - ...
                        # After pruning: u is the root of the pruned subtree, v is in the rest
                        # Regraft: pick edge x-y in the rest, remove it, add x-u and y-u
                        # But u already has edges to its children in the pruned subtree
                        # So u's degree increases by 1 (remove v, add x and y: net +1)
                        # x's degree: remove y, add u: net 0
                        # y's degree: remove x, add u: net 0
                        # This works!
                        
                        # Save state for this regraft
                        adj_save2 = [set(s) for s in adj]
                        el_save2 = dict(edge_lengths)
                        
                        # Remove edge x-y
                        adj[x].discard(y)
                        adj[y].discard(x)
                        edge_lengths.pop((x, y), None)
                        edge_lengths.pop((y, x), None)
                        
                        # Add edges x-u and y-u
                        adj[x].add(u)
                        adj[u].add(x)
                        adj[y].add(u)
                        adj[u].add(y)
                        # Edge lengths: set to 0 for now (topology only)
                        edge_lengths[(x, u)] = 0.0
                        edge_lengths[(u, x)] = 0.0
                        edge_lengths[(y, u)] = 0.0
                        edge_lengths[(u, y)] = 0.0
                        
                        new_score = _quartet_score(adj, edge_lengths, n_leaves, labels)
                        
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_regraft = (x, y)
                        
                        # Restore
                        adj = [set(s) for s in adj_save2]
                        edge_lengths = dict(el_save2)
                
                if best_regraft is not None:
                    # Apply the best regraft
                    x, y = best_regraft
                    adj_save2 = [set(s) for s in adj]
                    el_save2 = dict(edge_lengths)
                    
                    adj[x].discard(y)
                    adj[y].discard(x)
                    edge_lengths.pop((x, y), None)
                    edge_lengths.pop((y, x), None)
                    
                    adj[x].add(u)
                    adj[u].add(x)
                    adj[y].add(u)
                    adj[u].add(y)
                    edge_lengths[(x, u)] = 0.0
                    edge_lengths[(u, x)] = 0.0
                    edge_lengths[(y, u)] = 0.0
                    edge_lengths[(u, y)] = 0.0
                    
                    current_score = best_new_score
                    improved = True
                else:
                    # Restore original state
                    adj = [set(s) for s in adj_save]
                    edge_lengths = dict(el_save)
        
        if not improved:
            break
    
    return adj, edge_lengths


def _adj_to_newick(adj, n_leaves, labels):
    """Convert adjacency list to Newick string."""
    n = n_leaves
    
    # Find a leaf to use as root
    root = 0
    # Build parent pointers via BFS from root
    parent = {root: None}
    children = {root: []}
    stack = [root]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in parent:
                parent[nb] = node
                children[node].append(nb)
                stack.append(nb)
    
    def build_newick(node):
        if node < n_leaves:
            return labels[node]
        kids = children[node]
        if len(kids) == 0:
            return labels[node] if node < n_leaves else ""
        return "(" + ",".join(build_newick(k) for k in kids) + ")"
    
    return build_newick(root) + ";"


def _star_newick(labels):
    """Return a star tree Newick."""
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Generate multiple NJ candidates
    candidates = []
    
    # Candidate 1: standard NJ
    adj, el, nc = _nj_tree(dist, labels)
    candidates.append((adj, el, nc))
    
    if _time_exceeded():
        return _adj_to_newick(adj, n, labels)
    
    # Candidate 2+: NJ with perturbed distances
    rng = random.Random(42)
    num_variants = 3
    for _ in range(num_variants):
        if _time_exceeded():
            break
        noise = np.abs(rng.gauss(0, 0.03, (n, n)))
        perturbed = dist + noise
        # Symmetrize
        perturbed = 0.5 * (perturbed + perturbed.T)
        np.fill_diagonal(perturbed, 0.0)
        adj, el, nc = _nj_tree(perturbed, labels)
        candidates.append((adj, el, nc))
    
    # Score all candidates and pick the best
    best_score = -1
    best_candidate = candidates[0]
    for adj, el, nc in candidates:
        score = _quartet_score(adj, el, n, labels)
        if score > best_score:
            best_score = score
            best_candidate = (adj, el, nc)
    
    adj, el, nc = best_candidate
    
    if _time_exceeded():
        return _adj_to_newick(adj, n, labels)
    
    # SPR refinement
    adj, el = _spr_refine(adj, el, dist, n, labels, max_rounds=2)
    
    if _time_exceeded():
        return _adj_to_newick(adj, n, labels)
    
    return _adj_to_newick(adj, n, labels)
