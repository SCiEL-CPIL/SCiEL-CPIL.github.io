import math
import time
import numpy as np
import re

_GLOBAL_START = None
_DEADLINE = 210.0  # Total budget for all instances

def _elapsed():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return time.time() - _GLOBAL_START

def _time_exceeded():
    return _elapsed() > _DEADLINE

def _is_valid_newick(newick, labels):
    if not newick:
        return False
    found_labels = re.findall(r't\d+', newick)
    return set(found_labels) == set(labels)

def _parse_newick_to_tree(newick):
    """
    Parse Newick string into an internal tree representation.
    Returns:
      - adj: adjacency dict
      - node_labels: list where node_labels[i] is label for leaf i, None for internal
      - is_leaf: boolean list
      - n_nodes: number of nodes
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    node_counter = [0]
    node_labels = []
    is_leaf = []
    children = []
    
    def create_node(label=None):
        nid = node_counter[0]
        node_counter[0] += 1
        node_labels.append(label)
        is_leaf.append(label is not None)
        children.append([])
        return nid
    
    def parse(pos):
        if s[pos] == '(':
            pos += 1
            internal_id = create_node(None)
            while True:
                child_id, pos = parse(pos)
                children[internal_id].append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
            return internal_id, pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',:;)':
                end += 1
            label = s[pos:end]
            leaf_id = create_node(label)
            pos = end
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
            return leaf_id, pos
    
    root_id, _ = parse(0)
    n_nodes = len(node_labels)
    
    adj = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        for c in children[i]:
            adj[i].append(c)
            adj[c].append(i)
    
    return adj, node_labels, is_leaf, n_nodes

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    char_to_idx = np.array([0, 1, 2, 3], dtype=np.int8)
    lookup = np.zeros(256, dtype=np.int8)
    for c, i in zip('ACGT', range(4)):
        lookup[ord(c)] = i
    
    seqs = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        for j, c in enumerate(s):
            seqs[i, j] = lookup[ord(c)]
    
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(np.float64) / L
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) * 0.5
    return d

def _build_path_matrix(adj, leaf_nodes, labels, label_to_idx, dist):
    """
    Build the path matrix A and distance vector d_vec for SSE computation.
    Returns (A, d_vec, edges, edge_index)
    """
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return None, None, None, None
    
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    edges = list(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    num_edges = len(edges)
    num_pairs = n_leaves * (n_leaves - 1) // 2
    
    A = np.zeros((num_pairs, num_edges))
    d_vec = np.zeros(num_pairs)
    
    pair_idx = 0
    for i in range(n_leaves):
        li = leaf_nodes[i]
        idx_i = label_to_idx[labels[li]]
        
        dist_to_node = {li: 0}
        parent = {li: -1}
        queue = [li]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in dist_to_node:
                    dist_to_node[v] = dist_to_node[u] + 1
                    parent[v] = u
                    queue.append(v)
        
        for j in range(i + 1, n_leaves):
            lj = leaf_nodes[j]
            idx_j = label_to_idx[labels[lj]]
            d_vec[pair_idx] = dist[idx_i, idx_j]
            
            path = []
            curr = lj
            while curr != -1:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                edge = (u, v) if u < v else (v, u)
                A[pair_idx, edge_index[edge]] = 1.0
            
            pair_idx += 1
    
    return A, d_vec, edges, edge_index

def _compute_sse(A, d_vec):
    """Compute SSE from pre-built path matrix and distance vector."""
    if A is None or len(d_vec) == 0:
        return 0.0
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, d_vec, rcond=None)
    except:
        return float('inf')
    pred = A.dot(x)
    sse = np.sum((pred - d_vec) ** 2)
    return sse

def _tree_sse(adj, labels, dist, label_to_idx):
    """
    Compute SSE for a tree topology by solving for optimal branch lengths.
    """
    leaf_nodes = []
    for node in adj:
        if labels[node] is not None:
            leaf_nodes.append(node)
    
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return 0.0
    
    A, d_vec, edges, edge_index = _build_path_matrix(adj, leaf_nodes, labels, label_to_idx, dist)
    if A is None:
        return 0.0
    
    return _compute_sse(A, d_vec)

def _nj_tree(labels, dist):
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

def _quartet_tree(labels, dist):
    """
    Quartet-based tree construction: count quartet topologies and build tree from splits.
    For each 4-taxon subset, determine the most likely quartet topology from distances,
    then derive the set of splits and assemble a tree.
    """
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        return "(" + ",".join(labels) + ");"
    
    # For each quartet (i,j,k,l), determine the topology
    # Under JC model, the correct topology has the smallest sum of the two shortest distances
    # Topology (i,j)|(k,l): score = d(i,k)+d(i,l)+d(j,k)+d(j,l) - d(i,j)-d(k,l)
    # Actually, the quartet score for topology (ab)|(cd) is d(a,c)+d(a,d)+d(b,c)+d(b,d)
    # The topology with minimum score is the most likely one
    
    # Count votes for each split
    # A split is represented by a frozenset of taxa on one side
    split_votes = {}
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _time_exceeded():
                        break
                    # Three possible topologies for taxa i,j,k,l:
                    # (i,j)|(k,l), (i,k)|(j,l), (i,l)|(j,k)
                    # Score for (i,j)|(k,l): d(i,k)+d(i,l)+d(j,k)+d(j,l)
                    # Score for (i,k)|(j,l): d(i,j)+d(i,l)+d(j,k)+d(k,l)
                    # Score for (i,l)|(j,k): d(i,j)+d(i,k)+d(j,l)+d(k,l)
                    
                    s1 = dist[i][k] + dist[i][l] + dist[j][k] + dist[j][l]
                    s2 = dist[i][j] + dist[i][l] + dist[j][k] + dist[k][l]
                    s3 = dist[i][j] + dist[i][k] + dist[j][l] + dist[k][l]
                    
                    scores = [s1, s2, s3]
                    min_score = min(scores)
                    
                    # Count votes for each split
                    if s1 <= s2 and s1 <= s3:
                        # (i,j)|(k,l)
                        split = frozenset({i, j})
                        split_votes[split] = split_votes.get(split, 0) + 1
                    elif s2 <= s1 and s2 <= s3:
                        # (i,k)|(j,l)
                        split = frozenset({i, k})
                        split_votes[split] = split_votes.get(split, 0) + 1
                    else:
                        # (i,l)|(j,k)
                        split = frozenset({i, l})
                        split_votes[split] = split_votes.get(split, 0) + 1
    
    if not split_votes:
        return "(" + ",".join(labels) + ");"
    
    # Now we have a set of splits. We need to assemble a tree from them.
    # Use the splits to build a tree via recursive partitioning
    all_taxa = frozenset(range(n))
    
    # Filter splits: keep only those with sufficient support and that are compatible
    # A split is compatible with another if they don't cross
    # Two splits A|B and C|D are compatible if one of A∩C, A∩D, B∩C, B∩D is empty
    
    # Simple approach: greedily add splits that are compatible with existing ones
    compatible_splits = []
    for split in split_votes.keys():
        if len(split) < 2 or len(split) > n - 2:
            continue
        # Check compatibility with existing splits
        compatible = True
        for existing in compatible_splits:
            # Check if split and existing are compatible
            # split = frozenset(S), existing = frozenset(T)
            # They are compatible if S∩T=∅, S∩T=S, S∩T=T, or S∪T=all_taxa
            inter = split & existing
            if len(inter) > 0 and inter != split and inter != existing and (split | existing) != all_taxa:
                compatible = False
                break
        if compatible:
            compatible_splits.append(split)
    
    if not compatible_splits:
        return "(" + ",".join(labels) + ");"
    
    # Build tree from compatible splits using recursive partitioning
    def build_tree(taxa_set):
        if len(taxa_set) <= 1:
            return labels[list(taxa_set)[0]] if taxa_set else ""
        if len(taxa_set) == 2:
            return f"({labels[list(taxa_set)[0]]},{labels[list(taxa_set)[1]]})"
        
        # Find a split that partitions taxa_set
        best_split = None
        best_size = 0
        for split in compatible_splits:
            inter = taxa_set & split
            if len(inter) > 0 and len(inter) < len(taxa_set):
                # This split partitions taxa_set
                if len(inter) < best_size or best_split is None:
                    best_split = inter
                    best_size = len(inter)
        
        if best_split is None or len(best_split) == 0 or len(best_split) == len(taxa_set):
            # No valid split, make a star
            return "(" + ",".join(labels[i] for i in sorted(taxa_set)) + ")"
        
        left = build_tree(best_split)
        right = build_tree(taxa_set - best_split)
        return f"({left},{right})"
    
    tree_str = build_tree(all_taxa)
    return tree_str + ";"

def _get_edges(adj):
    """Get list of unique edges from adjacency dict."""
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    return list(edges)

def _nni_refine(adj, node_labels, dist, label_to_idx, max_rounds=3):
    """
    Perform NNI refinement on a tree represented by adjacency list.
    Returns the best adjacency list found.
    """
    leaf_nodes = [node for node in adj if node_labels[node] is not None]
    n_leaves = len(leaf_nodes)
    
    if n_leaves < 4:
        return adj
    
    # Precompute initial path matrix
    A, d_vec, edges, edge_index = _build_path_matrix(adj, leaf_nodes, node_labels, label_to_idx, dist)
    if A is None:
        return adj
    
    current_sse = _compute_sse(A, d_vec)
    
    improved = True
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        if _time_exceeded():
            break
        
        best_sse = current_sse
        best_adj = None
        
        edges = _get_edges(adj)
        
        for edge in edges:
            if _time_exceeded():
                break
            u, v = edge
            
            if len(adj[u]) < 2 or len(adj[v]) < 2:
                continue
            
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            for a in u_neighbors:
                for b in v_neighbors:
                    if _time_exceeded():
                        break
                    
                    # Create new adjacency
                    new_adj = {x: list(y) for x, y in adj.items()}
                    
                    new_adj[u].remove(v)
                    new_adj[v].remove(u)
                    new_adj[u].remove(a)
                    new_adj[v].remove(b)
                    
                    new_adj[u].append(b)
                    new_adj[b].append(u)
                    new_adj[v].append(a)
                    new_adj[a].append(v)
                    
                    # Compute SSE for new tree
                    new_leaf_nodes = [node for node in new_adj if node_labels[node] is not None]
                    new_A, new_d_vec, _, _ = _build_path_matrix(new_adj, new_leaf_nodes, node_labels, label_to_idx, dist)
                    if new_A is None:
                        continue
                    sse = _compute_sse(new_A, new_d_vec)
                    
                    if sse < best_sse - 1e-10:
                        best_sse = sse
                        best_adj = new_adj
                        improved = True
        
        if best_adj is not None:
            adj = best_adj
            current_sse = best_sse
    
    return adj

def _adj_to_newick(adj, node_labels):
    """
    Convert adjacency list to Newick string.
    """
    n_nodes = len(node_labels)
    
    root = None
    for i in range(n_nodes):
        if node_labels[i] is not None:
            root = i
            break
    
    if root is None:
        return ""
    
    visited = set()
    
    def gen(node, parent):
        if node in visited:
            return ""
        visited.add(node)
        
        if node_labels[node] is not None:
            return node_labels[node]
        else:
            children = []
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                child_str = gen(neighbor, node)
                if child_str:
                    children.append(child_str)
            
            if len(children) == 0:
                return ""
            elif len(children) == 1:
                return "(" + children[0] + ")"
            else:
                return "(" + ",".join(children) + ")"
                
    s = gen(root, -1)
    return s + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    dist = _jc_distance_matrix(sequences, labels)
    
    # Try both NJ and quartet-based methods, pick the better one
    nj_newick = _nj_tree(labels, dist)
    
    if not _is_valid_newick(nj_newick, labels):
        nj_newick = "(" + ",".join(labels) + ");"
    
    quartet_newick = None
    if not _time_exceeded() and n <= 20:
        try:
            quartet_newick = _quartet_tree(labels, dist)
            if not _is_valid_newick(quartet_newick, labels):
                quartet_newick = None
        except Exception:
            quartet_newick = None
    
    # Pick the better initial tree based on SSE
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    
    nj_adj, nj_labels, _, _ = _parse_newick_to_tree(nj_newick)
    nj_sse = _tree_sse(nj_adj, nj_labels, dist, label_to_idx)
    
    best_newick = nj_newick
    best_adj = nj_adj
    best_labels = nj_labels
    best_sse = nj_sse
    
    if quartet_newick is not None:
        q_adj, q_labels, _, _ = _parse_newick_to_tree(quartet_newick)
        q_sse = _tree_sse(q_adj, q_labels, dist, label_to_idx)
        if q_sse < nj_sse:
            best_newick = quartet_newick
            best_adj = q_adj
            best_labels = q_labels
            best_sse = q_sse
    
    # NNI refinement
    if not _time_exceeded():
        try:
            refined_adj = _nni_refine(best_adj, best_labels, dist, label_to_idx, max_rounds=3)
            refined_newick = _adj_to_newick(refined_adj, best_labels)
            if _is_valid_newick(refined_newick, labels):
                best_newick = refined_newick
        except Exception:
            pass
    
    if not _is_valid_newick(best_newick, labels):
        return "(" + ",".join(labels) + ");"
        
    return best_newick

def _spr_refine(newick, labels, dist):
    """
    Perform SPR refinement.
    """
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    refined_adj = _nni_refine(adj, node_labels, dist, label_to_idx, max_rounds=3)
    return _adj_to_newick(refined_adj, node_labels)

def _spr_refine_impl(newick, labels, dist, label_to_idx):
    """
    Implement NNI refinement.
    """
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    refined_adj = _nni_refine(adj, node_labels, dist, label_to_idx, max_rounds=3)
    return _adj_to_newick(refined_adj, node_labels)
