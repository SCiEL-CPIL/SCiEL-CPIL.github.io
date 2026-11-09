import numpy as np
import math
import time
from itertools import combinations

class Node:
    __slots__ = ['left', 'right', 'label', 'parent']
    
    def __init__(self, left=None, right=None, label=None):
        self.left = left
        self.right = right
        self.label = label
        self.parent = None
        
    def is_leaf(self):
        return self.label is not None

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized operations."""
    n = len(labels)
    L = len(sequences[labels[0]])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Encode sequences as integer matrix
    seq_matrix = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            seq_matrix[i, j] = base_map[seq[j]]
    
    # Compute pairwise Hamming distances vectorized
    diff = (seq_matrix[:, None, :] != seq_matrix[None, :, :])
    hamming = diff.sum(axis=2)
    
    # JC correction
    p = hamming.astype(float) / L
    p = np.clip(p, 0, 0.74)  # Avoid log(0) or negative args
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    
    return dist, seq_matrix

def _quartet_scores(dist):
    """
    Compute all quartet scores from distance matrix.
    For each quartet (a,b,c,d), compute:
      score_ab_cd = d_ab + d_cd - d_ac - d_bd
    Positive score supports (ab|cd) topology.
    Returns dict mapping (frozenset({a,b}), frozenset({c,d})) to score,
    where frozensets are the two splits.
    """
    n = len(dist)
    scores = {}
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Three possible splits:
                    # (ab|cd): d_ab + d_cd - d_ac - d_bd
                    # (ac|bd): d_ac + d_bd - d_ab - d_cd
                    # (ad|bc): d_ad + d_bc - d_ab - d_cd
                    
                    s1 = dist[a,b] + dist[c,d] - dist[a,c] - dist[b,d]  # ab|cd
                    s2 = dist[a,c] + dist[b,d] - dist[a,b] - dist[c,d]  # ac|bd
                    s3 = dist[a,d] + dist[b,c] - dist[a,b] - dist[c,d]  # ad|bc
                    
                    # Store the best supported split
                    best_score = max(s1, s2, s3)
                    if best_score > 0:
                        if best_score == s1:
                            key = (frozenset({a,b}), frozenset({c,d}))
                            # Canonicalize: sort the two frozensets
                            if key[0] > key[1]:
                                key = (key[1], key[0])
                            scores[key] = scores.get(key, 0) + best_score
                        elif best_score == s2:
                            key = (frozenset({a,c}), frozenset({b,d}))
                            if key[0] > key[1]:
                                key = (key[1], key[0])
                            scores[key] = scores.get(key, 0) + best_score
                        else:
                            key = (frozenset({a,d}), frozenset({b,c}))
                            if key[0] > key[1]:
                                key = (key[1], key[0])
                            scores[key] = scores.get(key, 0) + best_score
    
    return scores

def _fdq_build_tree(dist, labels, quartet_scores):
    """
    Build tree using Fast Distance-based Quartet approach.
    Start with a star tree, then iteratively add internal edges
    that are most strongly supported by quartet scores.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return Node(label=labels[0])
        elif n == 2:
            root = Node()
            l = Node(label=labels[0])
            r = Node(label=labels[1])
            l.parent = root
            r.parent = root
            root.left = l
            root.right = r
            return root
        else:
            # 3 taxa: any binary tree is fine
            root = Node()
            l = Node(label=labels[0])
            mid = Node()
            r1 = Node(label=labels[1])
            r2 = Node(label=labels[2])
            mid.left = r1
            mid.right = r2
            r1.parent = mid
            r2.parent = mid
            root.left = l
            root.right = mid
            l.parent = root
            mid.parent = root
            return root
    
    # Start with NJ tree as initial estimate, then refine with quartet info
    nj_tree = _nj_build_tree(dist, labels)
    
    # Use quartet scores to identify potential topological corrections
    # For each internal edge in the NJ tree, check if quartet evidence
    # supports a different topology nearby (NNI moves)
    
    refined_tree = _quartet_refine(nj_tree, dist, labels, quartet_scores)
    return refined_tree

def _nj_build_tree(dist, labels):
    """Standard Neighbor-Joining tree construction."""
    n = len(labels)
    if n == 1:
        return Node(label=labels[0])
    if n == 2:
        root = Node()
        l = Node(label=labels[0])
        r = Node(label=labels[1])
        l.parent = root
        r.parent = root
        root.left = l
        root.right = r
        return root
    
    # Track current nodes (indices into dist, or new internal node indices)
    current_nodes = list(range(n))
    current_dist = dist.copy()
    tree_edges = {}  # maps internal node index to (child1, child2, d1, d2)
    
    next_node_idx = n
    
    while len(current_nodes) > 2:
        n_curr = len(current_nodes)
        
        # Compute r values (sum of distances to all other current nodes)
        # Build a temporary distance matrix for current nodes
        curr_dist = np.zeros((n_curr, n_curr))
        for i in range(n_curr):
            for j in range(n_curr):
                if i != j:
                    curr_dist[i, j] = current_dist[current_nodes[i], current_nodes[j]]
        
        r = curr_dist.sum(axis=1)
        
        # Find best pair
        best_q = float('inf')
        best_i, best_j = -1, -1
        for i in range(n_curr):
            for j in range(i+1, n_curr):
                q = (n_curr - 2) * curr_dist[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        node_i = current_nodes[best_i]
        node_j = current_nodes[best_j]
        
        # Compute branch lengths
        d_ij = current_dist[node_i, node_j]
        r_i = r[best_i]
        r_j = r[best_j]
        d_i = (r_i - r_j) / (2 * (n_curr - 2))
        d_j = d_ij - d_i
        
        # Create new internal node
        new_idx = next_node_idx
        next_node_idx += 1
        
        # Record the edge
        tree_edges[new_idx] = (node_i, node_j, d_i, d_j)
        
        # Compute distances from new node to all other current nodes
        new_dists = {}
        for k in range(n_curr):
            if k == best_i or k == best_j:
                continue
            node_k = current_nodes[k]
            d_new_k = 0.5 * (current_dist[node_i, node_k] + current_dist[node_j, node_k] - d_ij)
            new_dists[node_k] = d_new_k
        
        # Update current_nodes and current_dist
        new_current_nodes = []
        for k in range(n_curr):
            if k != best_i and k != best_j:
                new_current_nodes.append(current_nodes[k])
        new_current_nodes.append(new_idx)
        
        new_n = len(new_current_nodes)
        new_dist = np.zeros((new_n, new_n))
        for i_idx in range(new_n):
            for j_idx in range(new_n):
                if i_idx == j_idx:
                    continue
                ni = new_current_nodes[i_idx]
                nj = new_current_nodes[j_idx]
                if ni == new_idx:
                    new_dist[i_idx, j_idx] = new_dists[nj]
                elif nj == new_idx:
                    new_dist[i_idx, j_idx] = new_dists[ni]
                else:
                    new_dist[i_idx, j_idx] = current_dist[ni, nj]
        
        current_nodes = new_current_nodes
        current_dist = new_dist
    
    # Final two nodes
    final_i = current_nodes[0]
    final_j = current_nodes[1]
    d_final = current_dist[final_i, final_j]
    tree_edges['final'] = (final_i, final_j, d_final / 2, d_final / 2)
    
    # Build the tree from tree_edges
    # We need to resolve the hierarchy
    node_map = {}  # maps node index to Node object
    
    def resolve(node_idx):
        if node_idx in node_map:
            return node_map[node_idx]
        
        if node_idx < n:
            # It's a leaf
            node = Node(label=labels[node_idx])
            node_map[node_idx] = node
            return node
        
        if node_idx == 'final':
            # Handle the final merge separately - this shouldn't happen in this structure
            pass
        
        if node_idx in tree_edges:
            child1, child2, _, _ = tree_edges[node_idx]
            left = resolve(child1)
            right = resolve(child2)
            parent = Node(left=left, right=right)
            left.parent = parent
            right.parent = parent
            node_map[node_idx] = parent
            return parent
        
        # Should not reach here
        return Node(label=f"unknown_{node_idx}")
    
    # The root is the last internal node created
    root_idx = next_node_idx - 1
    root = resolve(root_idx)
    
    return root

def _get_internal_edges(node):
    """Get all internal edges (parent, child) where child is internal."""
    edges = []
    if node.is_leaf():
        return edges
    if node.left and not node.left.is_leaf():
        edges.append((node, node.left))
        edges.extend(_get_internal_edges(node.left))
    if node.right and not node.right.is_leaf():
        edges.append((node, node.right))
        edges.extend(_get_internal_edges(node.right))
    return edges

def _fitch_score(node, seq_masks, leaf_indices):
    """Compute Fitch parsimony score for a tree."""
    if node.is_leaf():
        idx = leaf_indices[node.label]
        return seq_masks[:, idx], 0
    
    left_masks, left_score = _fitch_score(node.left, seq_masks, leaf_indices)
    right_masks, right_score = _fitch_score(node.right, seq_masks, leaf_indices)
    
    intersect = left_masks & right_masks
    union = left_masks | right_masks
    steps = np.sum(intersect == 0)
    masks = np.where(intersect == 0, union, intersect)
    
    return masks, left_score + right_score + steps

def _quartet_refine(tree, dist, labels, quartet_scores, max_rounds=5):
    """
    Refine tree topology using quartet scores to guide NNI moves.
    For each NNI move, evaluate if it's supported by quartet evidence.
    """
    n = len(labels)
    if n < 4:
        return tree
    
    leaf_indices = {label: i for i, label in enumerate(labels)}
    
    # Compute initial Fitch score
    # Need seq_masks - recompute from labels and dist isn't enough, 
    # so we use a proxy: use quartet-based scoring
    
    current_tree = tree
    
    for round_num in range(max_rounds):
        improved = False
        edges = _get_internal_edges(current_tree)
        
        for u, v in edges:
            # Identify the four subtrees around the internal edge (u,v)
            if u.left == v:
                w = u.right
            else:
                w = u.left
            
            x = v.left
            y = v.right
            
            # Get leaf sets for each subtree
            def get_leaves(node):
                leaves = []
                if node.is_leaf():
                    leaves.append(node.label)
                else:
                    leaves.extend(get_leaves(node.left))
                    leaves.extend(get_leaves(node.right))
                return leaves
            
            leaves_w = get_leaves(w)
            leaves_x = get_leaves(x)
            leaves_y = get_leaves(y)
            
            # For each NNI move, check quartet support
            # Move 1: (w,x)|(y,rest) vs current (w,y)|(x,rest)
            # Move 2: (w,y)|(x,rest) vs current (w,x)|(y,rest)
            
            # Sample quartets to evaluate support
            # For efficiency, use a subset of quartets
            import random
            random.seed(42)
            
            def quartet_support(move_type):
                """Estimate quartet support for a given NNI move."""
                score = 0
                count = 0
                
                # Sample quartets involving one leaf from each of w, x, y
                sample_size = min(50, len(leaves_w) * len(leaves_x) * len(leaves_y))
                
                for _ in range(sample_size):
                    lw = random.choice(leaves_w)
                    lx = random.choice(leaves_x)
                    ly = random.choice(leaves_y)
                    
                    # Need a fourth leaf - use one from the rest of the tree
                    # For simplicity, just use the three and check consistency
                    # Actually, let's use a different approach:
                    # Check if the distance matrix supports the new topology
                    
                    # For move 1: new topology has (w,x) together
                    # Check d(lw,lx) + d(ly, something) vs d(lw,ly) + d(lx, something)
                    # This is getting complex; let's use a simpler heuristic
                    
                    # Use the distance matrix directly
                    i_w = leaf_indices[lw]
                    i_x = leaf_indices[lx]
                    i_y = leaf_indices[ly]
                    
                    # For the current topology (w is separate from x,y which are together):
                    # Expected: d(w,x) + d(w,y) >= d(x,y)
                    # For new topology 1 (w,x together):
                    # Expected: d(w,x) + d(y,rest) < d(w,y) + d(x,rest)
                    
                    # Simplified: just use the quartet score from the matrix
                    # Find a fourth leaf from outside {w,x,y}
                    all_leaves = set(labels)
                    rest = all_leaves - set(leaves_w + leaves_x + leaves_y)
                    if not rest:
                        continue
                    lz = random.choice(list(rest))
                    i_z = leaf_indices[lz]
                    
                    # Current topology: (w, (x, y)) - w is outside the (x,y) clade
                    # New topology 1: ((w, x), y) - w and x are together
                    # New topology 2: ((w, y), x) - w and y are together
                    
                    # For move 1 (w,x together):
                    # Compare d(w,x) + d(y,z) vs d(w,y) + d(x,z) vs d(w,z) + d(x,y)
                    s_wx_yz = dist[i_w, i_x] + dist[i_y, i_z]
                    s_wy_xz = dist[i_w, i_y] + dist[i_x, i_z]
                    s_wz_xy = dist[i_w, i_z] + dist[i_x, i_y]
                    
                    if move_type == 1:
                        # w,x together: s_wx_yz should be smallest
                        if s_wx_yz < s_wy_xz and s_wx_yz < s_wz_xy:
                            score += 1
                        elif s_wx_yz > s_wy_xz and s_wx_yz > s_wz_xy:
                            score -= 1
                    else:
                        # w,y together: s_wy_xz should be smallest
                        if s_wy_xz < s_wx_yz and s_wy_xz < s_wz_xy:
                            score += 1
                        elif s_wy_xz > s_wx_yz and s_wy_xz > s_wz_xy:
                            score -= 1
                    
                    count += 1
                
                return score / max(count, 1)
            
            support1 = quartet_support(1)
            support2 = quartet_support(2)
            
            # If a move has strong positive support, apply it
            if support1 > 0.3:
                # Apply NNI move 1: swap w and x
                # New structure: u = (x, v'), v' = (w, y)
                v.left = w
                v.right = y
                w.parent = v
                y.parent = v
                
                u.left = x
                u.right = v
                x.parent = u
                v.parent = u
                
                improved = True
                break
            elif support2 > 0.3:
                # Apply NNI move 2: swap w and y
                # New structure: u = (y, v'), v' = (w, x)
                v.left = w
                v.right = x
                w.parent = v
                x.parent = v
                
                u.left = y
                u.right = v
                y.parent = u
                v.parent = u
                
                improved = True
                break
        
        if not improved:
            break
    
    return current_tree

def _node_to_newick(node):
    """Convert tree to Newick string."""
    if node.is_leaf():
        return node.label
    left = _node_to_newick(node.left)
    right = _node_to_newick(node.right)
    return f"({left},{right})"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 240  # Safety margin before 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    # 1. Compute distance matrix
    dist, seq_matrix = _jc_distance_matrix(sequences, labels)
    
    # 2. Compute quartet scores
    quartet_scores = _quartet_scores(dist)
    
    # 3. Build initial tree using NJ
    tree = _nj_build_tree(dist, labels)
    
    # 4. Refine using quartet-guided NNI
    if time.time() < deadline - 10:
        tree = _quartet_refine(tree, dist, labels, quartet_scores, max_rounds=3)
    
    # 5. Final NNI refinement using Fitch parsimony as tiebreaker
    if time.time() < deadline - 5:
        tree = _fitch_nni_refine(tree, seq_matrix, labels)
    
    return _node_to_newick(tree) + ";"

def _fitch_nni_refine(tree, seq_matrix, labels, max_rounds=2):
    """Final NNI refinement using Fitch parsimony score."""
    n = len(labels)
    if n < 4:
        return tree
    
    leaf_indices = {label: i for i, label in enumerate(labels)}
    
    # Compute initial score
    _, current_score = _fitch_score(tree, seq_matrix, leaf_indices)
    
    for round_num in range(max_rounds):
        improved = False
        edges = _get_internal_edges(tree)
        
        for u, v in edges:
            if u.left == v:
                w = u.right
            else:
                w = u.left
            
            x = v.left
            y = v.right
            
            # Move 1: swap w and x
            # Save state
            u_left_orig, u_right_orig = u.left, u.right
            v_left_orig, v_right_orig = v.left, v.right
            
            # Apply move 1
            v.left = w
            v.right = y
            w.parent = v
            y.parent = v
            u.left = x
            u.right = v
            x.parent = u
            v.parent = u
            
            _, score1 = _fitch_score(tree, seq_matrix, leaf_indices)
            gain1 = current_score - score1
            
            # Revert
            u.left, u.right = u_left_orig, u_right_orig
            v.left, v.right = v_left_orig, v_right_orig
            w.parent, x.parent, y.parent = u, v, v
            v.parent = u
            
            # Move 2: swap w and y
            v.left = w
            v.right = x
            w.parent = v
            x.parent = v
            u.left = y
            u.right = v
            y.parent = u
            v.parent = u
            
            _, score2 = _fitch_score(tree, seq_matrix, leaf_indices)
            gain2 = current_score - score2
            
            # Revert
            u.left, u.right = u_left_orig, u_right_orig
            v.left, v.right = v_left_orig, v_right_orig
            w.parent, x.parent, y.parent = u, v, v
            v.parent = u
            
            # Apply best move
            if gain1 > gain2 and gain1 > 0:
                v.left = w
                v.right = y
                w.parent = v
                y.parent = v
                u.left = x
                u.right = v
                x.parent = u
                v.parent = u
                current_score -= gain1
                improved = True
                break
            elif gain2 > 0:
                v.left = w
                v.right = x
                w.parent = v
                x.parent = v
                u.left = y
                u.right = v
                y.parent = u
                v.parent = u
                current_score -= gain2
                improved = True
                break
        
        if not improved:
            break
    
    return tree
