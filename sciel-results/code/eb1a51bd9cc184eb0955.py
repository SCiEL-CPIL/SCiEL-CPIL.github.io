"""Phylogenetic topology recovery via NJ + NNI refinement + bootstrap voting."""
import time
import math
import numpy as np
import re

_DEADLINE_FRACTION = 0.92
_start_time = None


def _encode_sequences(sequences, labels):
    """Encode sequences as uint8 array for fast operations."""
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    return seqs


def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from encoded sequences."""
    n = seqs.shape[0]
    L = seqs.shape[1]
    
    # Pairwise differences
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / L
    np.fill_diagonal(pmat, 0.0)
    
    # JC correction
    pmat_clipped = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat_clipped / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat, pmat


def _neighbor_joining(dmat, labels):
    """
    Perform Neighbor-Joining to reconstruct a tree.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
        
    # Store labels in a set for O(1) lookup in build_newick
    leaf_set = set(labels)
    
    # dist[(u, v)]
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(labels[i], labels[j])] = dmat[i, j]
            
    active_nodes = list(labels)
    tree_nodes = {}
    next_node_id = 0
    
    while len(active_nodes) > 2:
        m = len(active_nodes)
        
        # Compute r_i
        r = {}
        for i in active_nodes:
            r[i] = sum(dist[(i, k)] for k in active_nodes if k != i)
            
        # Compute Q and find min
        min_q = float('inf')
        i_node = None
        j_node = None
        
        for i in active_nodes:
            for j in active_nodes:
                if i == j:
                    continue
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    i_node = i
                    j_node = j
                    
        if i_node is None:
            break
            
        d_ij = dist[(i_node, j_node)]
        li = 0.5 * d_ij + (r[i_node] - r[j_node]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[i_node] - r[j_node]) / (2 * (m - 2))
        
        w_name = f"w{next_node_id}"
        next_node_id += 1
        
        tree_nodes[w_name] = (i_node, j_node, li, lj)
        
        # Update distances for w
        for k in active_nodes:
            if k == i_node or k == j_node:
                continue
            d_wk = 0.5 * (dist[(i_node, k)] + dist[(j_node, k)] - d_ij)
            dist[(w_name, k)] = d_wk
            dist[(k, w_name)] = d_wk
            
        # Remove i, j and add w
        active_nodes.remove(i_node)
        active_nodes.remove(j_node)
        active_nodes.append(w_name)
        
    # Connect the last two
    if len(active_nodes) == 2:
        u, v = active_nodes
        d_uv = dist.get((u, v), 0.0)
        w_name = f"w{next_node_id}"
        next_node_id += 1
        tree_nodes[w_name] = (u, v, d_uv / 2.0, d_uv / 2.0)
        active_nodes = [w_name]
        
    # Build Newick string
    def build_newick(node):
        if node in leaf_set:
            return node
        if node not in tree_nodes:
            return node  # Fallback
            
        child1, child2, len1, len2 = tree_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
        
    # The root is the last node created
    root = active_nodes[0]
    return build_newick(root)


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _parse_newick(newick, labels):
    """
    Parse a Newick string into a tree structure for NNI.
    Returns:
        - adjacency: dict node -> set of neighbors
        - leaf_sets: dict node -> set of leaf labels under that node (for rooted view)
        - all_leaves: set of all leaf labels
    """
    # Remove branch lengths
    newick_clean = re.sub(r':[0-9.\-eE+]+', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    # We'll build an unrooted tree representation
    # Each internal node has exactly 3 neighbors in an unrooted binary tree
    # We'll use a recursive parser that builds the tree bottom-up
    
    # First, let's build a rooted tree, then convert to unrooted
    # Actually, for NNI we need an unrooted representation
    # Let's build it differently: parse into a dict of nodes with their children
    
    # Simple approach: build a nested dict, then flatten to adjacency
    def parse_node(s, idx):
        """Parse a node starting at s[idx]. Returns (node_id, set_of_leaves, idx_after)."""
        nonlocal _next_id
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child_id, child_leaves, idx = parse_node(s, idx)
                children.append((child_id, child_leaves))
                if s[idx] == ',':
                    idx += 1
            idx += 1  # skip ')'
            
            # Create internal node
            node_id = f"i{_next_id}"
            _next_id += 1
            all_leaves = set()
            for cid, cleaves in children:
                all_leaves.update(cleaves)
            return node_id, all_leaves, idx
        else:
            # Leaf
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return leaf, {leaf}, j
    
    _next_id = 0
    root_id, root_leaves, _ = parse_node(newick_clean, 0)
    
    # Now build adjacency from the tree structure
    # We need to re-parse to get the structure
    def parse_struct(s, idx):
        """Returns (node_id, children_list, idx_after)."""
        nonlocal _next_id2
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child_id, child_children, idx = parse_struct(s, idx)
                children.append(child_id)
                if s[idx] == ',':
                    idx += 1
            idx += 1
            node_id = f"i{_next_id2}"
            _next_id2 += 1
            return node_id, children, idx
        else:
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return leaf, [], j
    
    _next_id2 = 0
    # Re-parse with same ID scheme
    # We need consistent IDs, so let's do it in one pass
    
    # Actually, let's rebuild properly
    adjacency = {}
    leaf_sets = {}
    
    def build_adj(s, idx, parent):
        nonlocal _next_id3
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child_id, child_adj, idx = build_adj(s, idx, None)
                children.append(child_id)
                if s[idx] == ',':
                    idx += 1
            idx += 1
            node_id = f"i{_next_id3}"
            _next_id3 += 1
            
            # Add edges
            if node_id not in adjacency:
                adjacency[node_id] = set()
            for child in children:
                adjacency[node_id].add(child)
                if child not in adjacency:
                    adjacency[child] = set()
                adjacency[child].add(node_id)
            
            # Compute leaf set
            all_leaves = set()
            for child in children:
                if child in leaf_sets:
                    all_leaves.update(leaf_sets[child])
            leaf_sets[node_id] = all_leaves
            
            return node_id, adjacency, idx
        else:
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            if leaf not in adjacency:
                adjacency[leaf] = set()
            if parent is not None:
                adjacency[leaf].add(parent)
                adjacency[parent].add(leaf)
            leaf_sets[leaf] = {leaf}
            return leaf, adjacency, j
    
    _next_id3 = 0
    root, adjacency, _ = build_adj(newick_clean, 0, None)
    
    # The root in a rooted binary tree has 2 children, so in unrooted view it has degree 2
    # But we want an unrooted tree where all internal nodes have degree 3
    # For the root, we need to "suppress" it or handle it specially
    # Actually, for NNI on unrooted trees, we should use an unrooted representation
    # Let's just use the rooted representation and handle the root specially
    
    return adjacency, leaf_sets, root


def _nni_refine(adjacency, leaf_sets, dmat, labels, max_rounds=50):
    """
    Perform NNI refinement using the distance matrix.
    The score for a quartet (a,b,c,d) is based on the sum of squared distances.
    We try all NNI moves and accept if it reduces the total score.
    """
    n = len(labels)
    if n < 5:
        return adjacency, leaf_sets
    
    all_leaves = set(labels)
    
    # Precompute leaf set for each node
    # Ensure all nodes have leaf_sets
    for node in adjacency:
        if node not in leaf_sets:
            # Compute by BFS
            visited = set()
            queue = [node]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                if cur in all_leaves:
                    if cur not in leaf_sets:
                        leaf_sets[cur] = {cur}
                    else:
                        leaf_sets[cur].add(cur)
                for nb in adjacency[cur]:
                    if nb not in visited:
                        queue.append(nb)
    
    # Recompute leaf_sets properly via post-order
    def recompute_leaf_sets():
        # Do a DFS from an arbitrary leaf
        start_leaf = labels[0]
        stack = [(start_leaf, None, False)]
        visited = set()
        while stack:
            node, parent, processed = stack.pop()
            if processed:
                all_leaves = set()
                for nb in adjacency[node]:
                    if nb != parent and nb in leaf_sets:
                        all_leaves.update(leaf_sets[nb])
                if node in all_leaves:
                    all_leaves.add(node)
                leaf_sets[node] = all_leaves
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, parent, True))
            for nb in adjacency[node]:
                if nb != parent:
                    stack.append((nb, node, False))
    
    recompute_leaf_sets()
    
    def quartet_score(leaves1, leaves2, leaves3, leaves4):
        """Score for a quartet configuration. Lower is better."""
        # For unrooted quartet with topology ((a,b),(c,d)), the score is:
        # sum of squared distances for the "correct" pairing vs the alternative
        # We use the criterion: S = (d_ab + d_cd)^2 + (d_ac + d_bd)^2 + (d_ad + d_bc)^2
        # But for NNI, we compare the current topology to the two alternatives
        # Current: ((A,B),(C,D)) where A,B are on one side, C,D on the other
        # The distance-based criterion for NNI:
        # Compute the sum of distances across the "split"
        # Actually, let's use the standard NNI criterion based on the distance matrix:
        # For edge (U,V) with neighbors A,B (of U) and C,D (of V):
        # Current split: AB | CD
        # Alt 1: AC | BD
        # Alt 2: AD | BC
        # Score = sum of d(x,y) for x in one part, y in other
        # We want to minimize the sum of squared distances within groups or something
        # Let's use: for each pair of leaves, compute the path length
        # Simpler: use the sum of pairwise distances within each cluster
        def cluster_score(cluster):
            s = 0.0
            cl = list(cluster)
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    s += dmat[labels.index(cl[i]), labels.index(cl[j])]
            return s
        return 0  # Placeholder
    
    # Let's implement a simpler NNI using the "quartet score" based on distances
    # For a candidate edge (U,V), with U connected to {A,B} and V connected to {C,D}:
    # The three possible quartet topologies have different "costs"
    # Cost for ((A,B)|(C,D)): proportional to d(A,C)+d(A,D)+d(B,C)+d(B,D) - d(A,B) - d(C,D)
    # We want to minimize this cost
    # Actually, the standard NNI criterion compares:
    # S_curr = d(A,C) + d(A,D) + d(B,C) + d(B,D)
    # S_alt1 = d(A,B) + d(C,D)  [this is for the alternative where A and B are together]
    # Hmm, let me think again.
    
    # For NNI on an unrooted tree with edge (U,V):
    # U has neighbors {V, A, B}, V has neighbors {U, C, D}
    # Current topology: the tree with edge (U,V)
    # NNI move 1: replace (U,V) with (A,C) and (B,D) -- wait, that's not right
    # NNI move 1: connect A-C and B-D, removing U-V, A-U, B-U, C-V, D-V
    # New: A connected to C, B connected to D, and the "middle" connection
    # Actually: U connects to {A, B, C} and V connects to {D, ...} no...
    
    # Let me use the standard formulation:
    # Edge (U,V). U's other neighbors: A, B. V's other neighbors: C, D.
    # Three quartet topologies:
    # 1. (A,B)|(C,D) - current
    # 2. (A,C)|(B,D)
    # 3. (A,D)|(B,C)
    # Distance-based score for each:
    # S1 = d(A,C) + d(A,D) + d(B,C) + d(B,D)  [sum of cross distances]
    # S2 = d(A,B) + d(C,D) + ... hmm
    # Actually, the standard criterion is:
    # For topology (AB|CD), the "length" is proportional to d(AB, CD) where d(X,Y) is the sum of distances
    # Let's just compute the total sum of pairwise distances and see which topology minimizes it
    # Or better: use the criterion from the literature
    
    # I'll use a simpler approach: compute the "quartet likelihood" proxy
    # For each quartet, compute the three possible topology scores and pick the best
    # Score for (AB|CD) = d(A,B) + d(C,D) - d(A,C) - d(A,D) - d(B,C) - d(B,D) + d(A,C)+d(B,D) ... 
    # This is getting complicated. Let me just use a simple heuristic.
    
    # For now, let's implement NNI by trying all moves and using a simple distance-based score
    # Score = sum of squared edge lengths (approximated by distances)
    
    # Actually, let me just do a simplified NNI that checks if swapping reduces the
    # sum of distances between leaves in the same cluster
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        
        # Find all internal edges (edges between two internal nodes)
        internal_edges = []
        for u in list(adjacency.keys()):
            if u in all_leaves:
                continue
            for v in adjacency[u]:
                if v in all_leaves:
                    continue
                if u < v:  # Avoid duplicates
                    internal_edges.append((u, v))
        
        for u, v in internal_edges:
            # Check if u and v are still connected
            if v not in adjacency.get(u, set()):
                continue
            
            # Get other neighbors
            u_neighbors = [x for x in adjacency[u] if x != v]
            v_neighbors = [x for x in adjacency[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            
            a, b = u_neighbors
            c, d = v_neighbors
            
            # Get leaf sets
            sa = leaf_sets.get(a, set())
            sb = leaf_sets.get(b, set())
            sc = leaf_sets.get(c, set())
            sd = leaf_sets.get(d, set())
            
            # Current topology: (ab)|(cd)
            # Compute score for each of the three topologies
            # Score = sum of d(x,y) for x in first cluster, y in second cluster
            # We want to MINIMIZE this (closer leaves should be together)
            
            def cross_score(cl1, cl2):
                s = 0.0
                for x in cl1:
                    for y in cl2:
                        if x in labels and y in labels:
                            s += dmat[labels.index(x), labels.index(y)]
                return s
            
            # Current: sa|sb vs sc|sd -- no, the split is (sa∪sb) | (sc∪sd)
            curr_split1 = sa | sb
            curr_split2 = sc | sd
            score_curr = cross_score(curr_split1, curr_split2)
            
            # Alt 1: sa|sc vs sb|sd -> (sa∪sc) | (sb∪sd)
            alt1_split1 = sa | sc
            alt1_split2 = sb | sd
            score_alt1 = cross_score(alt1_split1, alt1_split2)
            
            # Alt 2: sa|sd vs sb|sc -> (sa∪sd) | (sb∪sc)
            alt2_split1 = sa | sd
            alt2_split2 = sb | sc
            score_alt2 = cross_score(alt2_split1, alt2_split2)
            
            if score_alt1 < score_curr - 1e-9:
                # Perform NNI: replace edge (u,v) with edges (a,c) and (b,d)
                # Remove u-v, a-u, b-u, c-v, d-v
                # Add a-c, b-d
                adjacency[u].discard(v)
                adjacency[v].discard(u)
                adjacency[u].discard(a)
                adjacency[a].discard(u)
                adjacency[u].discard(b)
                adjacency[b].discard(u)
                adjacency[v].discard(c)
                adjacency[c].discard(v)
                adjacency[v].discard(d)
                adjacency[d].discard(v)
                
                adjacency[a].add(c)
                adjacency[c].add(a)
                adjacency[b].add(d)
                adjacency[d].add(b)
                
                # Remove u and v if they have no neighbors
                if not adjacency[u]:
                    del adjacency[u]
                if not adjacency[v]:
                    del adjacency[v]
                
                improved = True
                break
            elif score_alt2 < score_curr - 1e-9:
                # Perform NNI: replace edge (u,v) with edges (a,d) and (b,c)
                adjacency[u].discard(v)
                adjacency[v].discard(u)
                adjacency[u].discard(a)
                adjacency[a].discard(u)
                adjacency[u].discard(b)
                adjacency[b].discard(u)
                adjacency[v].discard(c)
                adjacency[c].discard(v)
                adjacency[v].discard(d)
                adjacency[d].discard(v)
                
                adjacency[a].add(d)
                adjacency[d].add(a)
                adjacency[b].add(c)
                adjacency[c].add(b)
                
                if not adjacency[u]:
                    del adjacency[u]
                if not adjacency[v]:
                    del adjacency[v]
                
                improved = True
                break
        
        if improved:
            recompute_leaf_sets()
    
    return adjacency, leaf_sets


def _adj_to_newick(adjacency, leaf_sets, labels):
    """Convert adjacency representation to Newick string."""
    if not adjacency:
        return ""
    
    # Find a root (arbitrary leaf)
    root = labels[0]
    
    # Do a DFS to build Newick
    def build_newick(node, parent):
        children = [nb for nb in adjacency.get(node, set()) if nb != parent]
        if not children:
            return node
        parts = []
        for child in children:
            parts.append(build_newick(child, node))
        return "(" + ",".join(parts) + ")"
    
    return build_newick(root, None)


def _parse_newick_splits(newick, labels):
    """Parse a Newick string into a set of unrooted splits."""
    import re
    newick_clean = re.sub(r':[0-9.\-eE+]+', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    splits = set()
    labels_set = set(labels)
    
    def parse(s, idx):
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child, idx = parse(s, idx)
                children.append(child)
                if s[idx] == ',':
                    idx += 1
            idx += 1
            
            combined = set()
            for child in children:
                combined.update(child)
            
            for child in children:
                if 0 < len(child) < len(labels):
                    if len(child) <= len(labels) - len(child):
                        splits.add(tuple(sorted(child)))
                    else:
                        other = labels_set - child
                        splits.add(tuple(sorted(other)))
            
            return combined, idx
        else:
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
    
    parse(newick_clean, 0)
    return splits


def _bootstrap_vote(seqs, labels, num_bootstraps, deadline):
    """
    Perform bootstrap resampling and vote on topologies.
    """
    n = len(labels)
    L = seqs.shape[1]
    
    topology_counts = {}
    topology_newick = {}
    
    for b in range(num_bootstraps):
        boot_indices = np.random.randint(0, L, size=L)
        boot_seqs = seqs[:, boot_indices]
        
        dmat_boot, pmat_boot = _jc_distance_matrix(boot_seqs)
        
        try:
            tree = _neighbor_joining(dmat_boot, labels)
        except Exception:
            continue
        
        splits = _parse_newick_splits(tree, labels)
        key = frozenset(splits)
        
        if key not in topology_counts:
            topology_counts[key] = 0
            topology_newick[key] = tree
        topology_counts[key] += 1
        
        if time.time() > deadline:
            break
    
    if not topology_counts:
        return None
    
    best_key = max(topology_counts, key=topology_counts.get)
    return topology_newick[best_key]


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    deadline = _start_time + _DEADLINE_FRACTION * 270.0
    
    # Encode sequences
    seqs = _encode_sequences(sequences, labels)
    
    # Compute Distance Matrix
    dmat, pmat = _jc_distance_matrix(seqs)
    
    # Run NJ
    try:
        tree = _neighbor_joining(dmat, labels)
    except Exception:
        tree = _star_tree(labels)
    
    # NNI refinement
    try:
        adjacency, leaf_sets, root = _parse_newick(tree, labels)
        adjacency, leaf_sets = _nni_refine(adjacency, leaf_sets, dmat, labels, max_rounds=20)
        refined_tree = _adj_to_newick(adjacency, leaf_sets, labels)
        if refined_tree:
            tree = refined_tree
    except Exception:
        pass
    
    # Bootstrap voting if time permits
    elapsed = time.time() - _start_time
    remaining = deadline - elapsed
    
    if remaining > 0.1:
        per_boot_time = 0.003 * (n / 16.0) ** 1.5
        num_bootstraps = int(remaining * 0.7 / per_boot_time)
        num_bootstraps = max(5, min(num_bootstraps, 100))
        
        boot_tree = _bootstrap_vote(seqs, labels, num_bootstraps, deadline)
        if boot_tree is not None:
            # Apply NNI to bootstrap tree
            try:
                adjacency, leaf_sets, root = _parse_newick(boot_tree, labels)
                adjacency, leaf_sets = _nni_refine(adjacency, leaf_sets, dmat, labels, max_rounds=10)
                refined_tree = _adj_to_newick(adjacency, leaf_sets, labels)
                if refined_tree:
                    tree = refined_tree
            except Exception:
                tree = boot_tree
    
    return tree

