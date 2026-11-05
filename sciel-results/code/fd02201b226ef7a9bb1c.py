<<<<<<< SEARCH
"""Bootstrap portfolio: generate varied candidates, refine with SPR, keep best."""
import math
import random
import time

_START_TIME = None
_DEADLINE = 260.0  # seconds, safety margin under 270s


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    n = len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    n = len(labels)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_tree(labels, dist):
    """Neighbour Joining, returns adjacency list and internal node count."""
    n = len(labels)
    # adj: dict node -> list of (neighbor, length)
    # We track tree structure with parent pointers or adjacency
    nodes = list(range(n))  # internal indices 0..n-1 are leaves
    next_id = n
    # trees[i] for i in nodes: will be the subtree string or label
    # For topology-only, we track adjacency
    adj = {i: [] for i in nodes}
    # dist is indexed by original leaf indices; for internal nodes we extend
    # We'll maintain a distance dict
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]

    while len(nodes) > 3:
        m = len(nodes)
        r = [0.0] * m
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    r[i] += d[(a, b)]
        best_crit = None
        best_pair = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[(a, b)] - r[i] - r[j]
                if best_crit is None or crit < best_crit:
                    best_crit = crit
                    best_pair = (i, j, a, b)
        i, j, a, b = best_pair
        # Compute branch lengths
        da = 0.5 * d[(a, b)]
        db = 0.5 * d[(a, b)]
        for k in range(m):
            if k != i and k != j:
                c = nodes[k]
                da += 0.5 * (d[(a, c)] - d[(b, c)])
                db += 0.5 * (d[(b, c)] - d[(a, c)])
        da = max(da, 0.0)
        db = max(db, 0.0)
        new_node = next_id
        next_id += 1
        # Remove a and b, add new_node
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_node)
        # Update adj
        for (nb, _) in list(adj[a]):
            adj[nb].remove((a, 0))
        for (nb, _) in list(adj[b]):
            adj[nb].remove((b, 0))
        adj[new_node] = []
        adj[a] = []
        adj[b] = []
        # Connect a-new_node, b-new_node
        adj[a].append((new_node, da))
        adj[new_node].append((a, da))
        adj[b].append((new_node, db))
        adj[new_node].append((b, db))
        # Reconnect neighbors
        for (nb, _) in list(adj[new_node]):
            if nb not in (a, b):
                pass
        # Actually, a and b's neighbors (other than each other) connect to new_node
        # We already removed a,b from their neighbors' adj
        # Now add new_node to those neighbors
        # But we lost which neighbors. Let me redo:
        # Before removing, collect neighbors of a (excluding b) and b (excluding a)
        # This is getting complex. Let me use a different representation.
        break  # fallback to simple approach

    # This NJ is getting complex for SPR. Let me use a simpler tree representation.
    # For the purpose of this task, I'll use Newick strings and parse them for SPR.
    # Actually, let me just return the NJ result as a Newick string and do SPR on that.
    # For now, let me implement NJ properly returning a Newick string, then parse it.
    pass


def _newick_to_adj(newick, labels):
    """Parse Newick string to adjacency list. Returns (adj, node_labels, internal_count)."""
    # Simple recursive parse
    nodes = {}
    counter = [0]
    
    def parse(s, pos):
        # s[pos] should be '(' or a label
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child, pos = parse(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            node_id = ('int', counter[0])
            counter[0] += 1
            nodes[node_id] = ('internal', children)
            return node_id, pos
        else:
            # leaf label
            end = pos
            while end < len(s) and s[end] not in ',;)':
                end += 1
            label = s[pos:end]
            # Skip branch length if present
            if end < len(s) and s[end] == ':':
                end += 1
                while end < len(s) and s[end] not in ',;)':
                    end += 1
            node_id = ('leaf', label)
            nodes[node_id] = ('leaf', label)
            return node_id, end

    root, _ = parse(newick, 0)
    # Build adjacency
    adj = {}
    for nid in nodes:
        adj[nid] = set()
    for nid, (typ, data) in nodes.items():
        if typ == 'internal':
            for child in data:
                adj[nid].add(child)
                adj[child].add(nid)
    return adj, nodes


def _tree_sse(adj, nodes, labels, dist):
    """Compute sum of squared errors between dist matrix and tree path lengths."""
    n = len(labels)
    label_to_node = {lbl: ('leaf', lbl) for lbl in labels}
    
    # BFS/DFS to compute all-pairs path lengths in tree
    # Since tree is small, do BFS from each leaf
    def bfs_path_lengths(start):
        dists = {start: 0.0}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in dists:
                    dists[v] = dists[u] + 0.0  # We don't have branch lengths in adj!
                    stack.append(v)
        return dists
    
    # Problem: we don't have branch lengths. Let me include them.
    # This is getting complex. Let me use a different approach:
    # Instead of SSE, use the **quartet scoring** or just do SPR based on 
    # the NJ criterion locally.
    
    # Actually, for topology search, a common approach is to use the 
    # **sum of squared residuals** but we need branch lengths.
    # Let me simplify: use a heuristic SPR where we evaluate moves by 
    # checking if the local NJ criterion improves.
    
    # Alternative: use **FastUp** which uses the tree's own branch lengths.
    # But we'd need to optimize branch lengths.
    
    # For simplicity and given time constraints, let me use a simpler search:
    # 1. Generate multiple NJ trees with perturbed distances
    # 2. Do a few rounds of SPR using a simple scoring: 
    #    for each move, recompute the local distance criterion
    # 
    # Actually, let me just do: generate several NJ variants, and for each,
    # try a limited set of SPR moves scored by a simple proxy.
    pass


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    label_index = {lbl: i for i, lbl in enumerate(labels)}
    
    # Compute distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # If small, just do NJ
    if n <= 10:
        return _nj_newick(labels, dist)
    
    # Generate multiple candidates
    candidates = []
    
    # Candidate 1: standard NJ
    candidates.append(_nj_newick(labels, dist))
    
    # Check time
    if _time_exceeded():
        return candidates[0] if candidates else _trivial_tree(labels)
    
    # Candidate 2+: NJ with perturbed distances
    rng = random.Random(42)
    num_variants = 5
    for _ in range(num_variants):
        if _time_exceeded():
            break
        perturbed = [[dist[i][j] * (1.0 + rng.gauss(0, 0.05)) for j in range(n)] for i in range(n)]
        # Symmetrize
        for i in range(n):
            for j in range(i+1, n):
                avg = 0.5 * (perturbed[i][j] + perturbed[j][i])
                perturbed[i][j] = avg
                perturbed[j][i] = avg
        candidates.append(_nj_newick(labels, perturbed))
    
    # Now refine best candidates with limited SPR
    # For simplicity, do 1 round of SPR on the best candidate
    # Score: use SSE with least-squares branch length optimization
    # But that's expensive. Instead, use a simple local improvement:
    # For each internal edge, try all regraftings, keep best by a simple metric.
    
    # Given time constraints, let me just return the best NJ candidate
    # and do a very limited SPR if time permits.
    
    best = candidates[0]
    if len(candidates) > 1:
        # Pick best by some simple metric - use the NJ total criterion
        # Actually, without the true tree, we can't score directly.
        # Use SSE as proxy.
        # For now, just return the first NJ (standard)
        best = candidates[0]
    
    return best


def _nj_newick(labels, dist):
    """Standard NJ returning Newick string."""
    n = len(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    next_id = n
    while len(nodes) > 3:
        m = len(nodes)
        r = {}
        for a in nodes:
            r[a] = sum(d[(a, b)] for b in nodes if b != a)
        best_crit = None
        best_pair = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[(a, b)] - r[a] - r[b]
                if best_crit is None or crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        a, b = best_pair
        da = 0.5 * d[(a, b)]
        db = 0.5 * d[(a, b)]
        for x in nodes:
            if x != a and x != b:
                da += 0.5 * (d[(a, x)] - d[(b, x)])
                db += 0.5 * (d[(b, x)] - d[(a, x)])
        da = max(da, 0.0)
        db = max(db, 0.0)
        new = next_id
        next_id += 1
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d[(new, x)] = d[(x, new)] = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
        # Remove old entries
        for x in rest:
            del d[(a, x)]
            del d[(b, x)]
            del d[(x, a)]
            del d[(x, b)]
        del d[(a, b)]
        del d[(b, a)]
        nodes = rest + [new]
        trees[new] = f"({trees[a]},{trees[b]})"
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _trivial_tree(labels):
    """Star tree as fallback."""
    return "(" + ",".join(labels) + ");"
=======
"""Phylogenetic tree reconstruction with NJ + SPR refinement."""
import math
import time
import numpy as np

_START_TIME = None
_DEADLINE = 250.0  # safety margin under 270s


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Vectorized JC distance matrix computation."""
    n = len(labels)
    seqs = np.array([list(sequences[lbl]) for lbl in labels], dtype='|S1')
    # Convert to integer codes
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_seqs = np.zeros((n, seqs.shape[1]), dtype=np.int8)
    for i in range(n):
        for j in range(seqs.shape[1]):
            int_seqs[i, j] = code_map[seqs[i, j].decode()]
    
    # Compute pairwise differences
    diff = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff_count = np.sum(int_seqs[i] != int_seqs[j])
            p = diff_count / seqs.shape[1]
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            diff[i, j] = d
            diff[j, i] = d
    return diff


def _nj_tree(labels, dist):
    """Neighbour Joining. Returns (parent, left, right) arrays for internal nodes."""
    n = len(labels)
    # parent[i] for internal node i (n..2n-2)
    # left[i], right[i] are children of internal node i
    # For leaves 0..n-1, parent will be set when they're joined
    
    next_id = n
    active = list(range(n))
    # trees tracks the Newick subtree for each active node
    trees = {i: labels[i] for i in range(n)}
    
    d = dist.copy()
    
    while len(active) > 3:
        m = len(active)
        # Compute r vector
        r = np.zeros(m)
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i != j:
                    r[i] += d[a, b]
        
        best_crit = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                crit = (m - 2) * d[a, b] - r[i] - r[j]
                if crit < best_crit:
                    best_crit = crit
                    best_i, best_j = i, j
        
        a, b = active[best_i], active[best_j]
        
        # Compute branch lengths
        da = 0.5 * d[a, b]
        db = 0.5 * d[a, b]
        for k in range(m):
            if k != best_i and k != best_j:
                c = active[k]
                da += 0.5 * (d[a, c] - d[b, c])
                db += 0.5 * (d[b, c] - d[a, c])
        da = max(da, 0.0)
        db = max(db, 0.0)
        
        new = next_id
        next_id += 1
        
        # Update distances
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            d[new, x] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[x, new] = d[new, x]
        
        active = rest + [new]
        trees[new] = f"({trees[a]},{trees[b]})"
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def _parse_newick(newick, labels):
    """
    Parse Newick into a tree structure.
    Returns:
        parent: list where parent[i] is the parent of node i
        left: list where left[i] is left child of node i (or -1 for leaves)
        right: list where right[i] is right child of node i (or -1 for leaves)
        is_leaf: list of booleans
        label_to_node: dict mapping label to node index
        total_nodes: total number of nodes
    """
    # Build adjacency by parsing
    # We'll create a tree where leaves are 0..n-1 and internal nodes are n..2n-2
    n = len(labels)
    
    # Parse recursively
    pos = [0]
    node_count = [n]  # internal nodes start at n
    
    def parse():
        nonlocal pos
        # Skip whitespace
        while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
            pos[0] += 1
        
        if pos[0] >= len(newick):
            return -1
        
        if newick[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            # Skip comma
            while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
                pos[0] += 1
            if pos[0] < len(newick) and newick[pos[0]] == ',':
                pos[0] += 1
            right = parse()
            # Skip closing paren
            while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
                pos[0] += 1
            if pos[0] < len(newick) and newick[pos[0]] == ')':
                pos[0] += 1
            
            # Create internal node
            node = node_count[0]
            node_count[0] += 1
            return node
        else:
            # Leaf
            start = pos[0]
            while pos[0] < len(newick) and newick[pos[0]] not in ',):':
                pos[0] += 1
            label = newick[start:pos[0]].strip()
            # Skip branch length
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(newick) and newick[pos[0]] not in ',);':
                    pos[0] += 1
            
            # Find this leaf's index
            for i, lbl in enumerate(labels):
                if lbl == label:
                    return i
            return -1
    
    root = parse()
    total = node_count[0]
    
    # Now we need to build parent/left/right arrays
    # Re-parse to build the structure
    pos[0] = 0
    parent = [-1] * total
    left = [-1] * total
    right = [-1] * total
    is_leaf = [False] * total
    label_to_node = {}
    
    for i in range(n):
        is_leaf[i] = True
        label_to_node[labels[i]] = i
    
    def parse2():
        nonlocal pos
        while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
            pos[0] += 1
        
        if pos[0] >= len(newick):
            return -1
        
        if newick[pos[0]] == '(':
            pos[0] += 1
            l = parse2()
            while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
                pos[0] += 1
            if pos[0] < len(newick) and newick[pos[0]] == ',':
                pos[0] += 1
            r = parse2()
            while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
                pos[0] += 1
            if pos[0] < len(newick) and newick[pos[0]] == ')':
                pos[0] += 1
            
            node = node_count[0] - 1  # The last created internal node
            # Actually, we need to track which internal node this corresponds to
            # This approach is flawed. Let me use a different strategy.
            return node
        else:
            start = pos[0]
            while pos[0] < len(newick) and newick[pos[0]] not in ',):':
                pos[0] += 1
            label = newick[start:pos[0]].strip()
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(newick) and newick[pos[0]] not in ',);':
                    pos[0] += 1
            
            for i, lbl in enumerate(labels):
                if lbl == label:
                    return i
            return -1
    
    # Simpler approach: just extract the splits directly from Newick
    # For SPR, we work with splits representation
    return None  # We'll use a different approach


def _get_splits_from_newick(newick, labels):
    """Extract all non-trivial splits from a Newick string."""
    n = len(labels)
    label_set = set(labels)
    
    # Parse to get all clades
    pos = [0]
    clades = []
    
    def parse_clade():
        nonlocal pos
        while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
            pos[0] += 1
        
        if pos[0] >= len(newick):
            return set()
        
        if newick[pos[0]] == '(':
            pos[0] += 1
            s = set()
            while True:
                c = parse_clade()
                s |= c
                while pos[0] < len(newick) and newick[pos[0]] in ' \t\n':
                    pos[0] += 1
                if pos[0] < len(newick) and newick[pos[0]] == ',':
                    pos[0] += 1
                    continue
                elif pos[0] < len(newick) and newick[pos[0]] == ')':
                    pos[0] += 1
                    break
                else:
                    break
            # Skip branch length
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(newick) and newick[pos[0]] not in ',);':
                    pos[0] += 1
            if len(s) > 1 and len(s) < n:
                clades.append(frozenset(s))
            return s
        else:
            start = pos[0]
            while pos[0] < len(newick) and newick[pos[0]] not in ',):':
                pos[0] += 1
            label = newick[start:pos[0]].strip()
            # Skip branch length
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(newick) and newick[pos[0]] not in ',);':
                    pos[0] += 1
            return {label}
    
    parse_clade()
    
    # Convert to canonical form (smaller side first)
    splits = set()
    for c in clades:
        if len(c) < n - len(c):
            splits.add(c)
        else:
            splits.add(frozenset(labels) - c)
    return splits


def _rf_distance(splits1, splits2):
    """Robinson-Foulds distance between two sets of splits."""
    return len(splits1 - splits2) + len(splits2 - splits1)


def _newick_from_splits(labels, splits):
    """Reconstruct a Newick string from a set of splits (for small n)."""
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Find a compatible set of splits and build tree
    # This is complex; for now, just return a star tree if we can't build properly
    # In practice, we should track the full tree structure, not just splits
    
    # For now, fallback to star
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Generate NJ tree
    newick = _nj_tree(labels, dist)
    
    if _time_exceeded() or n <= 8:
        return newick
    
    # Try SPR refinement if time permits
    # For small trees, do limited SPR
    if n <= 20 and not _time_exceeded():
        # Simple local search: try pruning each leaf and regrafting
        # This is a very limited SPR
        pass
    
    return newick
>>>>>>> REPLACE

Now let me add a proper SPR implementation that actually improves the topology:

