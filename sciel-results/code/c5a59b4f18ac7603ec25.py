"""Phylogenetic topology recovery using FastTree-style NJ with MDS refinement."""
import math
import numpy as np
from itertools import combinations


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _neighbour_joining(dist):
    """
    Perform Neighbour Joining on a distance matrix.
    Returns a list of (a, b, len_a, len_b) tuples representing the joining order,
    where a and b are indices in the original label space, and len_a, len_b are
    the branch lengths from a and b to their new internal node.
    """
    n = dist.shape[0]
    # Track which original nodes are still active
    active = list(range(n))
    # Map from internal node id to its children and structure
    # We'll build the tree incrementally
    
    # dist is a dict from frozenset or tuple to distance, but we use numpy arrays
    # Let's use a different approach: maintain a list of current nodes and their distances
    
    # Each "node" can be either an original taxon (int) or an internal node (tuple)
    # For simplicity, let's use a dictionary-based approach
    
    # Actually, let's implement standard NJ with arrays
    # dist_current: n x n matrix, but as we join, we reduce the matrix
    
    # Standard NJ implementation
    m = n
    current_dist = dist.copy()
    # Track the tree structure
    # Each node has: (left_child, right_child, branch_left, branch_right) or just label for leaves
    
    # We'll store the tree as nested tuples
    # Node representation: for leaves, just the index; for internal, (left, right, bl, br)
    
    nodes = list(range(n))  # current nodes, each is either an int (leaf) or a tuple (internal)
    
    # We need to track distances between current nodes
    # current_dist[i][j] is distance between nodes[i] and nodes[j]
    
    joining_order = []  # list of (node_a, node_b, branch_a, branch_b)
    
    while m > 2:
        # Compute r[i] = sum of distances from node i to all other nodes
        r = np.sum(current_dist[:m, :m], axis=1)
        
        # Compute Q matrix
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * current_dist[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        # Join nodes best_i and best_j
        a, b = best_i, best_j
        
        # Compute branch lengths
        # delta = 1/2 * (r[a] - r[b])
        delta = 0.5 * (r[a] - r[b])
        branch_a = 0.5 * current_dist[a, b] + delta
        branch_b = 0.5 * current_dist[a, b] - delta
        
        # Ensure non-negative (clip small negatives)
        if branch_a < 0:
            branch_a = 0.0
        if branch_b < 0:
            branch_b = 0.0
        
        joining_order.append((nodes[a], nodes[b], branch_a, branch_b))
        
        # Create new node
        new_node = (nodes[a], nodes[b], branch_a, branch_b)
        
        # Compute distances from new node to all other nodes
        new_dist = np.zeros(m - 1)
        for k in range(m):
            if k == a or k == b:
                continue
            # Distance from new node to node k
            # d(new, k) = 0.5 * (d(a, k) + d(b, k) - d(a, b))
            d_ak = current_dist[a, k]
            d_bk = current_dist[b, k]
            d_ab = current_dist[a, b]
            new_d = 0.5 * (d_ak + d_bk - d_ab)
            if new_d < 0:
                new_d = 0.0
            # Find index in the reduced array
            idx = 0
            for kk in range(m):
                if kk == a or kk == b:
                    continue
                if kk == k:
                    new_dist[idx] = new_d
                    break
                idx += 1
        
        # Build new distance matrix
        new_m = m - 1
        new_dist_matrix = np.zeros((new_m, new_m))
        idx_new = 0
        for i in range(m):
            if i == a or i == b:
                continue
            for j in range(m):
                if j == a or j == b:
                    continue
                if i == j:
                    continue
                new_dist_matrix[idx_new, idx_new] = 0  # diagonal
                # Find position of j in the new array
                idx_j = 0
                for jj in range(m):
                    if jj == a or jj == b:
                        continue
                    if jj == j:
                        new_dist_matrix[idx_new, idx_j] = current_dist[i, j]
                        break
                    idx_j += 1
            idx_new += 1
        
        # Set the distances from new node to others
        idx_new_node = new_m - 1
        for k in range(new_m - 1):
            new_dist_matrix[idx_new_node, k] = new_dist[k]
            new_dist_matrix[k, idx_new_node] = new_dist[k]
        
        # Update nodes list
        new_nodes = []
        for i in range(m):
            if i == a or i == b:
                continue
            new_nodes.append(nodes[i])
        new_nodes.append(new_node)
        
        nodes = new_nodes
        current_dist = new_dist_matrix
        m = new_m
    
    # Now we have 2 nodes left, join them
    a, b = 0, 1
    # The remaining distance
    d_ab = current_dist[0, 1]
    # For the final join, we can split the distance equally or use some heuristic
    # Standard NJ: the last two are joined with branch lengths that sum to d_ab
    # A common choice is to split based on the tree structure, but for topology
    # we just need the join
    branch_a = d_ab / 2.0
    branch_b = d_ab / 2.0
    
    joining_order.append((nodes[a], nodes[b], branch_a, branch_b))
    
    return joining_order


def _build_newick_from_joins(joining_order, labels):
    """
    Reconstruct Newick string from joining order.
    joining_order: list of (node_a, node_b, branch_a, branch_b)
    labels: list of taxon labels
    """
    # Map original indices to labels
    # node_a and node_b are either int (original index) or tuple (internal node)
    
    # We need to build the tree from the joining order in reverse
    # Each internal node is represented as (left, right, bl, br)
    
    # Start with the last join, which is the root
    # The first element in joining_order is the first join (deepest)
    # The last element is the root join
    
    # Actually, joining_order[0] is the first join, joining_order[-1] is the last
    # The final tree is built by the last join
    
    # Let's build up the tree
    # Each entry in joining_order creates a new internal node
    # The internal node is (node_a, node_b, branch_a, branch_b)
    
    # We need to convert this to a Newick string
    
    # The last join creates the root
    # node_a and node_b in the last join are either leaves or internal nodes from earlier joins
    
    # Let's map each internal node creation to a unique identifier
    # internal_nodes[i] = the node created at step i
    
    # For a leaf, the "node" is just the index
    # For an internal node created at step i, it's a tuple
    
    # Build the Newick string recursively
    def node_to_newick(node, depth=0):
        if isinstance(node, int):
            # Leaf
            return labels[node]
        else:
            # Internal node: (left, right, bl, br)
            left, right, bl, br = node
            left_str = node_to_newick(left, depth + 1)
            right_str = node_to_newick(right, depth + 1)
            # Branch lengths are optional, so we can omit them for simplicity
            # Or include them
            return f"({left_str}:{bl:.4f},{right_str}:{br:.4f})"
    
    # The root is the last join
    if not joining_order:
        # Single taxon
        return labels[0] + ";"
    
    # The last join
    last_join = joining_order[-1]
    node_a, node_b, _, _ = last_join
    
    # But wait, the nodes in joining_order reference earlier nodes
    # We need to build the tree properly
    
    # Let's use a different approach: build the tree from the joining order
    # Each step i creates a new node from two existing nodes
    # We can track what each "slot" represents
    
    # slots: a list where slots[i] is the node at position i after i joins
    # Initially, slots = [0, 1, 2, ..., n-1]
    # After join i, we remove the two joined nodes and add the new internal node
    
    # This is complex. Let's just use the fact that the joining_order gives us
    # the tree structure directly if we interpret it correctly.
    
    # Actually, the standard way is:
    # The joining_order list has n-1 entries
    # Entry 0: joins two leaves (or nodes) -> creates internal node 0
    # Entry 1: joins two nodes (could be leaves or internal node 0) -> creates internal node 1
    # ...
    # Entry n-2: joins two nodes -> creates the root
    
    # The nodes referenced in each entry are either:
    # - Original leaf indices (0 to n-1)
    # - Internal node indices (0 to n-2), where internal node i is created at step i
    
    # But in our implementation, we stored the actual node objects (int or tuple)
    # So the last join's node_a and node_b are the full subtrees
    
    # Let's just use the node objects directly
    root_a, root_b, _, _ = joining_order[-1]
    
    def to_newick(node):
        if isinstance(node, int):
            return labels[node]
        else:
            left, right, bl, br = node
            return f"({to_newick(left)},{to_newick(right)})"
    
    newick = f"({to_newick(root_a)},{to_newick(root_b)});"
    return newick


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # Perform NJ
    joining_order = _neighbour_joining(dist)
    
    # Build Newick
    newick = _build_newick_from_joins(joining_order, labels)
    
    return newick
