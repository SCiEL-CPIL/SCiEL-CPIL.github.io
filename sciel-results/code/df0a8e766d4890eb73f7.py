import numpy as np
import math
import time

class Node:
    __slots__ = ['left', 'right', 'label', 'parent']
    
    def __init__(self, left=None, right=None, label=None):
        self.left = left
        self.right = right
        self.label = label
        self.parent = None
        
    def is_leaf(self):
        return self.label is not None

def _jc_distance(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.75:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_nj_tree(matrix, labels):
    # matrix: 2D numpy array
    # labels: list of strings
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

    # Standard NJ
    # Use a list of active nodes
    # To keep it simple and fast for n<=32, we can use a dict or list
    # Let's use a list of "current" nodes, where each node is either a leaf or a joined cluster
    # We need to track the tree structure as we join.
    
    # This implementation is a bit complex to do with Node objects directly in NJ loop.
    # Easier: Run NJ to get a Newick string or nested tuples, then parse to Nodes.
    
    # Let's stick to the previous plan: NJ returns nested tuples.
    pass

def _nj_tuple(matrix, labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1])
        
    nodes = list(labels)
    trees = {name: name for name in labels}
    # matrix is dict or array? Let's use array for speed.
    # index_map
    idx = {name: i for i, name in enumerate(labels)}
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = matrix[labels[i], labels[j]]
                
    active = list(range(n))
    # trees is a dict from active_id to tuple/string
    # But active_id changes.
    # Let's keep a list of current "nodes" where each node is a tuple representation.
    # And a distance matrix between them.
    
    current_nodes = [labels[i] for i in range(n)]
    current_dist = dist.copy()
    
    while len(current_nodes) > 2:
        n_curr = len(current_nodes)
        # Calculate r[i] = sum(dist[i, j])
        r = current_dist.sum(axis=1)
        
        best_q = float('inf')
        best_i, best_j = -1, -1
        for i in range(n_curr):
            for j in range(i+1, n_curr):
                if i == j: continue
                q = (n_curr - 2) * current_dist[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
                    
        # Join best_i and best_j
        # New node
        new_node_tuple = (current_nodes[best_i], current_nodes[best_j])
        new_idx = len(current_nodes)
        
        # Update distances
        # d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        new_dists = np.zeros(n_curr)
        for k in range(n_curr):
            if k == best_i or k == best_j:
                new_dists[k] = 0
            else:
                new_dists[k] = 0.5 * (current_dist[best_i, k] + current_dist[best_j, k] - current_dist[best_i, best_j])
        
        # Remove best_i and best_j from current_nodes and current_dist
        # This is O(n^2) manipulation.
        
        # Create new distance matrix
        new_n = n_curr - 1
        new_dist = np.zeros((new_n, new_n))
        
        new_nodes_list = []
        new_nodes_list_map = {} # map old index to new index
        
        count = 0
        for k in range(n_curr):
            if k == best_i or k == best_j:
                continue
            new_nodes_list.append(current_nodes[k])
            new_nodes_list_map[k] = count
            count += 1
        
        # Copy distances
        for old_i in range(n_curr):
            if old_i == best_i or old_i == best_j: continue
            new_i = new_nodes_list_map[old_i]
            for old_j in range(n_curr):
                if old_j == best_i or old_j == best_j: continue
                new_j = new_nodes_list_map[old_j]
                new_dist[new_i, new_j] = current_dist[old_i, old_j]
        
        # Add new node
        new_nodes_list.append(new_node_tuple)
        new_nodes_list_map[new_idx] = len(new_nodes_list) - 1
        new_node_idx = len(new_nodes_list) - 1
        
        # Update new_dist with distances to new node
        for old_k in range(n_curr):
            if old_k == best_i or old_k == best_j: continue
            new_k = new_nodes_list_map[old_k]
            new_dist[new_k, new_node_idx] = new_dists[old_k]
            new_dist[new_node_idx, new_k] = new_dists[old_k]
            
        current_nodes = new_nodes_list
        current_dist = new_dist
        
    # Finally, join the last two
    return (current_nodes[0], current_nodes[1])

def _tuple_to_node(t):
    if isinstance(t, str):
        return Node(label=t)
    else:
        left_t, right_t = t
        left_n = _tuple_to_node(left_t)
        right_n = _tuple_to_node(right_t)
        root = Node(left=left_n, right=right_n)
        left_n.parent = root
        right_n.parent = root
        return root

def _node_to_newick(node):
    if node.is_leaf():
        return node.label
    left = _node_to_newick(node.left)
    right = _node_to_newick(node.right)
    return f"({left},{right})"

def _fitch_score(node, seq_masks, leaf_indices):
    # seq_masks: (L, n)
    # leaf_indices: dict label -> index
    # Returns score
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

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1: return f"{labels[0]};"
        return f"({labels[0]},{labels[1]});"
        
    # 1. Distance Matrix
    # Convert sequences to bitmasks for Fitch later
    base_map = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    L = len(sequences[labels[0]])
    seq_masks = np.zeros((L, n), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            seq_masks[j, i] = base_map[seq[j]]
            
    # Compute distances
    matrix = {}
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            matrix[(labels[i], labels[j])] = d
            matrix[(labels[j], labels[i])] = d
            
    # 2. NJ
    tree_tuple = _nj_tuple(matrix, labels)
    root = _tuple_to_node(tree_tuple)
    
    # 3. Initial Score
    leaf_indices = {label: i for i, label in enumerate(labels)}
    _, current_score = _fitch_score(root, seq_masks, leaf_indices)
    
    # 4. NNI Search
    # Helper to find internal edges
    # We will iterate through all nodes.
    
    def get_internal_edges(node):
        edges = []
        if node.is_leaf():
            return edges
        # Edge to left
        if not node.left.is_leaf():
            edges.append((node, node.left))
            edges.extend(get_internal_edges(node.left))
        # Edge to right
        if not node.right.is_leaf():
            edges.append((node, node.right))
            edges.extend(get_internal_edges(node.right))
        return edges
        
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_move = None
        
        edges = get_internal_edges(root)
        for u, v in edges:
            # u is parent, v is child
            # v is internal
            # u has other child w
            # v has children x, y
            
            # Identify w, x, y
            if u.left == v:
                w = u.right
            else:
                w = u.left
                
            x = v.left
            y = v.right
            
            # Save state
            # We need to swap subtrees
            
            # Move 1: Swap w and x
            # New structure:
            # u.left = x, u.right = v_new
            # v_new.left = w, v_new.right = y
            # (Assuming v was u.left)
            
            # Let's create a helper to apply and revert
            # It's safer to just calculate the score of the modified tree
            # But modifying and reverting is faster than rebuilding
            
            # State to save:
            # u.left, u.right, v.left, v.right
            # Parents: w.parent, x.parent, y.parent, v.parent
            
            orig_u_left = u.left
            orig_u_right = u.right
            orig_v_left = v.left
            orig_v_right = v.right
            
            orig_w_parent = w.parent
            orig_x_parent = x.parent
            orig_y_parent = y.parent
            orig_v_parent = v.parent
            
            # Apply Move 1
            if u.left == v:
                # u.left is v, u.right is w
                # Swap w and x
                # New v' = (w, y)
                # New u' = (x, v')
                
                # Create new node for v'? Or reuse v?
                # Reusing v is tricky because v's children change.
                # Let's just modify v and u in place.
                
                # v becomes (w, y)
                v.left = w
                v.right = y
                w.parent = v
                y.parent = v
                
                # u becomes (x, v)
                u.left = x
                u.right = v
                x.parent = u
                v.parent = u
                
                # Evaluate
                _, score1 = _fitch_score(root, seq_masks, leaf_indices)
                
                # Revert
                u.left = orig_u_left
                u.right = orig_u_right
                v.left = orig_v_left
                v.right = orig_v_right
                w.parent = orig_w_parent
                x.parent = orig_x_parent
                y.parent = orig_y_parent
                v.parent = orig_v_parent
                
                gain1 = current_score - score1
                if gain1 > best_gain:
                    best_gain = gain1
                    best_move = ('swap_wx', u, v, w, x, y)
                    
            # Apply Move 2: Swap w and y
            # Similar logic
            # ... (Implement similarly)
            
            # Check time
            if time.time() - start_time > 240:
                break
                
        if best_move:
            # Apply best move
            # ... (Implement apply logic)
            current_score -= best_gain
            improved = True
        else:
            break
            
    return _node_to_newick(root) + ";"
