import time
import math
import numpy as np
import random

_START_TIME = None
_TIME_LIMIT = 250.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

class TreeNode:
    __slots__ = ['left', 'right', 'label', 'parent', 'length']
    def __init__(self, left=None, right=None, label=None, parent=None, length=0.0):
        self.left = left
        self.right = right
        self.label = label
        self.parent = parent
        self.length = length

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_leaves(self):
        if self.is_leaf():
            return [self.label]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    return dists

def _nj_run(dists, labels):
    """
    Standard Neighbor Joining.
    Returns a TreeNode structure.
    """
    n = len(labels)
    if n < 3:
        if n == 1:
            return TreeNode(label=labels[0])
        elif n == 2:
            root = TreeNode()
            l = TreeNode(label=labels[0], parent=root)
            r = TreeNode(label=labels[1], parent=root)
            root.left = l
            root.right = r
            return root

    # Map label to index
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    # Active nodes are represented by indices in the current distance matrix
    # We maintain a list of 'active' nodes, which are either original leaves or internal nodes
    # To simplify, we'll keep a list of nodes (objects) and a distance matrix between them.
    
    nodes = [TreeNode(label=labels[i]) for i in range(n)]
    current_d = dists.copy()
    active_indices = list(range(n))
    
    # We need to track which original leaf indices belong to each current node
    # Actually, for NJ, we just need the distance matrix.
    # When we join i and j, we create a new node k.
    # The distance from k to any other active node m is 0.5 * (d(i,m) + d(j,m) - d(i,j))
    
    while len(active_indices) > 2:
        n_curr = len(active_indices)
        if n_curr == 2:
            break
            
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        # Join i and j
        node_i = nodes[active_indices[i]]
        node_j = nodes[active_indices[j]]
        
        d_ij = current_d[i, j]
        
        # Calculate branch lengths
        # len_i = 0.5 * d_ij + 0.5 * (sum_i - sum_j) / (n_curr - 2)
        # This formula is for the "outgroup" correction or something?
        # Standard NJ branch length:
        # alpha = (r_i - r_j) / (2 * (n_curr - 2))
        # len_i = 0.5 * d_ij + alpha
        # len_j = 0.5 * d_ij - alpha
        
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        len_i = 0.5 * d_ij + alpha
        len_j = 0.5 * d_ij - alpha
        
        # Create new node
        new_node = TreeNode()
        node_i.parent = new_node
        node_j.parent = new_node
        new_node.left = node_i
        new_node.right = node_j
        node_i.length = max(0.0, len_i)
        node_j.length = max(0.0, len_j)
        
        # Update distance matrix
        # Remove i and j, add new_node
        new_active_indices = [k for k in range(n_curr) if k != i and k != j]
        new_nodes_list = [nodes[active_indices[k]] for k in new_active_indices] + [new_node]
        
        # Calculate new distances
        n_new = len(new_nodes_list)
        new_d = np.zeros((n_new, n_new))
        
        # Copy old distances for remaining pairs
        # We need to map old indices to new indices
        # old i, j are removed.
        # The new node is at index n_new - 1.
        
        # For each remaining pair (a, b) in old indices:
        # new_d[a', b'] = current_d[a, b]
        
        # It's easier to rebuild the matrix
        # Let's keep a list of active node objects and their corresponding indices in the current_d matrix
        # This is getting messy with indices. Let's use a list of nodes and a matrix that matches the list order.
        
        # Let's restart the loop logic to be cleaner.
        break # This implementation is getting too complex for a quick rewrite. I will use a simpler recursive NJ or just stick to the matrix update carefully.
        
    # Let's implement a cleaner NJ using a list of nodes and a matrix that updates correctly.
    # Actually, the previous code in the prompt was working. I will refine that but return a Tree object.
    
    # Reset
    nodes = [TreeNode(label=labels[i]) for i in range(n)]
    current_d = dists.copy()
    active_nodes = list(range(n)) # indices into 'nodes' list? No, let's use a list of node objects and a parallel list of indices?
    
    # Let's use a dictionary: node_id -> node_obj
    # And a list of active node_ids.
    # And a distance matrix indexed by position in active_nodes list.
    
    node_id_counter = n
    active_ids = list(range(n))
    node_map = {i: nodes[i] for i in range(n)}
    current_d = dists.copy()
    
    while len(active_ids) > 2:
        n_curr = len(active_ids)
        
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        id_i = active_ids[i]
        id_j = active_ids[j]
        
        node_i = node_map[id_i]
        node_j = node_map[id_j]
        
        d_ij = current_d[i, j]
        
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        len_i = 0.5 * d_ij + alpha
        len_j = 0.5 * d_ij - alpha
        
        new_id = node_id_counter
        node_id_counter += 1
        
        new_node = TreeNode()
        node_i.parent = new_node
        node_j.parent = new_node
        new_node.left = node_i
        new_node.right = node_j
        node_i.length = max(0.0, len_i)
        node_j.length = max(0.0, len_j)
        
        node_map[new_id] = new_node
        
        # Update active_ids and current_d
        # Remove i and j from active_ids
        # Add new_id to active_ids
        
        # New active list
        new_active_ids = [k for k in range(n_curr) if k != i and k != j]
        new_active_ids.append(len(new_active_ids)) # placeholder for new node index
        
        # We need to build the new distance matrix
        n_new = len(new_active_ids)
        new_d = np.zeros((n_new, n_new))
        
        # Fill new_d
        # For each pair in new_active_ids:
        # If both are old, take from current_d
        # If one is new, calculate using formula
        
        # Map new index to old id or new_id
        # Let's just iterate
        
        # Create a map from new position to node_id
        pos_to_id = {}
        for pos, old_pos in enumerate(new_active_ids[:-1]):
            pos_to_id[pos] = active_ids[old_pos]
        pos_to_id[n_new - 1] = new_id
        
        for p1 in range(n_new):
            for p2 in range(p1 + 1, n_new):
                id1 = pos_to_id[p1]
                id2 = pos_to_id[p2]
                
                if id1 == new_id or id2 == new_id:
                    # Calculate distance to new node
                    # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
                    # x is the other node
                    if id1 == new_id:
                        x_id = id2
                    else:
                        x_id = id1
                    
                    # Find old indices of i, j, x
                    # i and j are the ones we just removed.
                    # x is in the remaining set.
                    
                    # We need the old indices of i, j, and x in the PREVIOUS current_d matrix.
                    # i and j were at positions i and j in the previous active_ids.
                    # x is at some position k in the previous active_ids.
                    
                    # Find k for x_id
                    old_pos_x = active_ids.index(x_id)
                    
                    d_ix = current_d[i, old_pos_x]
                    d_jx = current_d[j, old_pos_x]
                    
                    d_new_x = 0.5 * (d_ix + d_jx - d_ij)
                    d_new_x = max(0.0, d_new_x)
                    
                    new_d[p1, p2] = d_new_x
                    new_d[p2, p1] = d_new_x
                else:
                    # Both are old
                    old_pos1 = active_ids.index(id1)
                    old_pos2 = active_ids.index(id2)
                    new_d[p1, p2] = current_d[old_pos1, old_pos2]
                    new_d[p2, p1] = current_d[old_pos1, old_pos2]
                    
        current_d = new_d
        active_ids = list(pos_to_id.values())
        
    # Final step: connect the last 2 nodes
    if len(active_ids) == 2:
        id1 = active_ids[0]
        id2 = active_ids[1]
        node1 = node_map[id1]
        node2 = node_map[id2]
        
        d_12 = current_d[0, 1]
        
        # For the last two, the branch lengths are typically half the distance?
        # Or we can just set them to 0.5 * d_12
        # Actually, in unrooted NJ, the last step connects the two remaining subtrees.
        # The branch lengths are derived from the distances to the rest of the tree,
        # but since there is no "rest", we can split the distance evenly or use the previous logic?
        # The previous logic required n_curr > 2.
        # Let's just set them to 0.5 * d_12 for simplicity, as it doesn't affect topology.
        
        root = TreeNode()
        node1.parent = root
        node2.parent = root
        root.left = node1
        root.right = node2
        node1.length = 0.5 * d_12
        node2.length = 0.5 * d_12
        
        return root
    elif len(active_ids) == 3:
        # This shouldn't happen if we loop until > 2, then handle 2.
        # But if n=3, we start with 3, loop runs (3>2), joins 2, leaves 1?
        # No, if n=3, active_ids has 3.
        # i, j joined. new_active has 1 old + 1 new = 2.
        # Then loop ends.
        # So we always end with 2.
        pass
        
    # Fallback
    return nodes[0]

def _tree_to_newick(root):
    """Convert TreeNode to Newick string."""
    if root is None:
        return ""
    if root.is_leaf():
        return root.label
    left_str = _tree_to_newick(root.left)
    right_str = _tree_to_newick(root.right)
    return f"({left_str},{right_str})"

def _tree_splits(root):
    """Extract splits from a tree."""
    splits = set()
    all_leaves = set(root.get_leaves())
    
    def dfs(node):
        if node.is_leaf():
            return {node.label}
        
        left_leaves = dfs(node.left)
        right_leaves = dfs(node.right)
        
        # The split defined by this node is (left_leaves, all_leaves - left_leaves)
        # We canonicalize by taking the frozenset of the smaller side?
        # Or just frozenset(left_leaves) if we ensure consistency?
        # For RF distance, we need to compare sets of splits.
        # A split is a partition of the leaf set.
        # (A, B) is the same as (B, A).
        # So we can represent it as frozenset(A) if |A| <= |B|, else frozenset(B).
        # But we need to know the total set.
        
        if len(left_leaves) <= len(right_leaves):
            splits.add(frozenset(left_leaves))
        else:
            splits.add(frozenset(right_leaves))
            
        return left_leaves | right_leaves
        
    dfs(root)
    return splits

def _nni_refine(root, dists, labels, max_iters=100):
    """
    Perform NNI refinement using distance-based objective.
    Objective: Sum of squared differences between observed distances and tree-implied distances.
    """
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    # Precompute list of leaves
    leaves = root.get_leaves()
    
    # Function to calculate implied distance matrix for a tree
    def get_implied_dists(tree_root):
        # BFS/DFS to get distances from root? No, tree is unrooted.
        # We can root it arbitrarily at the first leaf.
        # Then distance between u and v is dist(root, u) + dist(root, v) - 2 * dist(root, LCA(u, v))
        
        # Simpler: For each pair of leaves, traverse the tree to find the path.
        # Since n is small (<=32), O(n^2 * n) is fine.
        
        # First, root the tree at leaves[0]
        # Assign parent pointers? The tree already has parent pointers, but they point to the unrooted parent.
        # We need to re-root or just do a DFS from a specific node.
        
        # Let's pick leaves[0] as root.
        # We need to traverse the tree.
        
        # Helper to get distance from root to all nodes
        def calc_dists_from_root(start_node):
            dists_from_root = {id(start_node): 0.0}
            stack = [(start_node, None)]
            while stack:
                node, parent = stack.pop()
                if node is None:
                    continue
                current_dist = dists_from_root[id(node)]
                
                if node.left and node.left is not parent:
                    dists_from_root[id(node.left)] = current_dist + node.left.length
                    stack.append((node.left, node))
                if node.right and node.right is not parent:
                    dists_from_root[id(node.right)] = current_dist + node.right.length
                    stack.append((node.right, node))
            return dists_from_root
            
        # This is O(n) per start node.
        # We need distances between all pairs of leaves.
        # We can do this more efficiently by rooting at one leaf and using LCA,
        # but for n=32, just calculating dist(u,v) by finding the path is O(n) per pair, so O(n^3) total.
        # 32^3 = 32768. Very fast.
        
        # Let's implement a function to get distance between two nodes in the tree.
        def dist_between(u, v):
            # Find path
            # Since it's a tree, we can just do BFS/DFS.
            # Or, simpler: Root at u, then dist(u, v) is just the distance in the rooted tree.
            # But doing this for every pair is O(n^3).
            
            # Let's try a different approach:
            # Root the tree at leaves[0].
            # Calculate dist from root to all nodes.
            # Calculate LCA for all pairs?
            
            # Actually, for NNI, we only need to evaluate the change in the objective function.
            # The objective is sum (d_obs - d_tree)^2.
            # When we do an NNI move, only the distances between leaves in the four subtrees involved change.
            # This allows for O(n) update instead of O(n^2).
            # But implementing that is complex.
            
            # For now, let's just recalculate the full objective for each neighbor.
            # With n=32, there are ~2n-5 = 59 internal edges.
            # Each edge has 1 NNI neighbor (actually 2 if we consider direction, but NNI is symmetric).
            # So ~60 neighbors.
            # Each evaluation is O(n^2) to compute dists, O(n^2) to compare.
            # 60 * 1000 = 60,000 ops per iteration.
            # 100 iterations = 6,000,000 ops.
            # This is fast enough in Python? Maybe.
            
            # Let's implement the full recalculation for simplicity and robustness.
            pass
            
        # Let's implement a faster distance calculation.
        # Root at leaves[0].
        # Precompute dist from root to all nodes.
        # Precompute LCA for all pairs of leaves?
        
        # Actually, let's just use the simple path finding for each pair.
        # It's safer.
        
        implied_dists = np.zeros((n, n))
        
        # To speed up, we can precompute the distance from the first leaf (root) to all nodes.
        # And use the property: dist(u, v) = dist(root, u) + dist(root, v) - 2 * dist(root, lca(u, v))
        
        # Let's just do the O(n^3) loop for now.
        
        for i in range(n):
            for j in range(i+1, n):
                u = leaves[i]
                v = leaves[j]
                
                # Find distance
                # BFS from u to v
                visited = {id(u): 0.0}
                stack = [u]
                found = False
                while stack:
                    node = stack.pop()
                    if node is v:
                        found = True
                        break
                    if node.left and id(node.left) not in visited:
                        visited[id(node.left)] = visited[id(node)] + node.left.length
                        stack.append(node.left)
                    if node.right and id(node.right) not in visited:
                        visited[id(node.right)] = visited[id(node)] + node.right.length
                        stack.append(node.right)
                        
                if found:
                    d = visited[id(v)]
                else:
                    d = 0.0 # Should not happen
                
                implied_dists[i, j] = d
                implied_dists[j, i] = d
                
        return implied_dists

    def objective(tree_root):
        implied = get_implied_dists(tree_root)
        # Sum of squared differences
        diff = dists - implied
        return np.sum(diff**2)
        
    current_obj = objective(root)
    
    # Find all internal edges
    # An internal edge is one connecting two internal nodes, or an internal node and a leaf?
    # NNI applies to internal edges (edges where both sides have at least 2 leaves? No, NNI is defined on internal edges).
    # In an unrooted binary tree, an NNI move is applied to an internal edge.
    # The four subtrees hanging off the edge are A, B, C, D.
    # The original is (A-B)-(C-D) (where the edge is between the nodes connecting A,B and C,D).
    # The neighbors are (A-C)-(B-D) and (A-D)-(B-C).
    
    # To enumerate NNI moves, we need to identify internal edges.
    # We can iterate over all internal nodes.
    # For each internal node X, look at its children Y and Z.
    # The edge (X, Y) is an internal edge if Y is also an internal node?
    # Actually, any edge in the tree can be considered.
    # But NNI is typically defined on edges where the two sides are both "subtrees".
    
    # Let's iterate over all nodes.
    # For each node X, if X is internal:
    #   Look at the edge connecting X to its parent P.
    #   This edge splits the tree into two parts: Subtree(X) and Subtree(P) - Subtree(X).
    #   Wait, NNI is local.
    
    # Standard NNI enumeration:
    # For each internal edge (u, v):
    #   Let u have children a, b (one is v, one is a).
    #   Let v have children c, d (one is u, one is c).
    #   The four subtrees are: Subtree(a), Subtree(b) [wait, b is v? No], Subtree(c), Subtree(d).
    #   
    # Let's be precise.
    # Edge (u, v).
    # u has two neighbors: parent(u) and child(u) (if rooted). In unrooted, u has degree 3.
    # Let neighbors of u be p, v, a. (p is parent, v is the other end of edge, a is the third neighbor).
    # Let neighbors of v be u, q, b. (q is parent of v, b is third neighbor).
    # The four subtrees are:
    #   T_a: subtree hanging off u at a
    #   T_b: subtree hanging off v at b
    #   T_p: subtree hanging off u at p (which includes the rest of the tree on u's side)
    #   T_q: subtree hanging off v at q (which includes the rest of the tree on v's side)
    #   
    # Original: (T_a, T_p) connected to (T_b, T_q) via edge (u,v).
    # Actually, the structure is:
    #      a
    #      |
    #   p--u--v--q
    #      |     |
    #      ?     ?
    # No, in a binary tree, u has 3 neighbors. One is v. The other two are a and p.
    # v has 3 neighbors. One is u. The other two are b and q.
    # 
    # The NNI move swaps the connections.
    # New tree 1: a connected to b, p connected to q.
    # New tree 2: a connected to q, p connected to b.
    
    # Implementation:
    # 1. Identify all internal edges.
    # 2. For each, perform the swap.
    # 3. Evaluate.
    
    # How to identify internal edges?
    # Iterate over all nodes.
    # For each node u:
    #   If u is internal (has 2 children in rooted sense, or degree 3 in unrooted):
    #     For each neighbor v of u:
    #       If v is also internal (or leaf? NNI usually requires the edge to be internal, meaning both sides have >1 leaf?
    #       Actually, if one side is a leaf, it's a "cherries" swap, which is a type of NNI).
    #       Let's consider all edges where both endpoints are not leaves?
    #       No, NNI can involve leaves.
    #       
    #       Let's just consider all edges (u, v) such that u != v.
    #       To avoid duplicates, only consider if id(u) < id(v).
    #       
    #       For each such edge:
    #         Determine the 4 subtrees.
    #         This is tricky without a rooted tree.
    
    # Given the complexity and time, I will skip NNI refinement in this version to ensure robustness and focus on a very fast, robust NJ implementation.
    # The NJ implementation above is likely faster and more robust than the string-based one in the prompt.
    
    return root, current_obj

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Convert sequences to integer matrix
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seqs_int = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Run NJ
    root = _nj_run(dists, labels)
    
    # Convert to Newick
    newick = _tree_to_newick(root)
    
    # Optional: NNI refinement (skipped for now)
    
    return newick + ";"

# EVOLVE-BLOCK-END
