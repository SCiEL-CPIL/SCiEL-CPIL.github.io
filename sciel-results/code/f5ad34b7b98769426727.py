import math
import time
import numpy as np
from itertools import combinations

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized operations."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Convert sequences to integer arrays (A=0, C=1, G=2, T=3)
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.zeros((n, L), dtype=np.int8)
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        # Vectorized conversion
        seq_matrix[i] = np.array([char_to_int.get(c, 0) for c in seq], dtype=np.int8)
        
    # Compute pairwise Hamming distances
    dist_mat = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_matrix[i] != seq_matrix[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat, seq_matrix

def _nj_tree(dist_mat, labels):
    """
    Perform Neighbour Joining on a distance matrix.
    Returns a dictionary of tree nodes and the root node id.
    Node ids: 0..n-1 are leaves (indices in labels), n.. are internal.
    """
    n = len(labels)
    if n == 1:
        return {0: labels[0]}, 0
    if n == 2:
        return {2: (0, 1)}, 2

    # Active nodes
    active = list(range(n))
    # Tree structure: node_id -> (child_a, child_b) or label for leaves
    tree_nodes = {i: labels[i] for i in range(n)}
    next_id = n
    
    # Distance matrix for active nodes
    # We maintain a dict of dicts or a 2D array. 
    # Since active nodes change, we use a dict of dicts for simplicity and correctness.
    # d[(u, v)] is distance between node u and v.
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = dist_mat[i, j]
    
    while len(active) > 3:
        k = len(active)
        
        # Compute r[i] = sum of distances from node i to all other active nodes
        r = {}
        for node in active:
            r[node] = sum(dist_dict[(node, other)] for other in active if other != node)
        
        # Compute Q matrix and find best pair
        best_q = float('inf')
        best_a, best_b = -1, -1
        
        for idx_a in range(k):
            for idx_b in range(idx_a + 1, k):
                node_a = active[idx_a]
                node_b = active[idx_b]
                d_ab = dist_dict[(node_a, node_b)]
                q = (k - 2) * d_ab - r[node_a] - r[node_b]
                if q < best_q:
                    best_q = q
                    best_a, best_b = node_a, node_b
        
        # Create new internal node
        new_node = next_id
        next_id += 1
        tree_nodes[new_node] = (best_a, best_b)
        
        # Compute distances from new node to all other active nodes
        for other in active:
            if other == best_a or other == best_b:
                continue
            d_a_other = dist_dict[(best_a, other)]
            d_b_other = dist_dict[(best_b, other)]
            d_ab = dist_dict[(best_a, best_b)]
            d_new_other = 0.5 * (d_a_other + d_b_other - d_ab)
            if d_new_other < 0:
                d_new_other = 0.0
            dist_dict[(new_node, other)] = d_new_other
            dist_dict[(other, new_node)] = d_new_other
        
        # Remove old nodes, add new node
        active = [x for x in active if x != best_a and x != best_b]
        active.append(new_node)
        
    # Handle final 3 nodes
    u, v, w = active
    node_1 = next_id
    next_id += 1
    tree_nodes[node_1] = (u, v)
    
    node_2 = next_id
    next_id += 1
    tree_nodes[node_2] = (node_1, w)
    
    return tree_nodes, node_2

def _get_leaves(node, tree_nodes, n):
    """Recursively get the set of leaf indices under a node."""
    if node < n:
        return frozenset([node])
    left, right = tree_nodes[node]
    return _get_leaves(left, tree_nodes, n) | _get_leaves(right, tree_nodes, n)

def _nni_refine(tree_nodes, root, n, seq_matrix, time_limit):
    """
    Perform NNI refinement based on quartet support counts.
    Returns the improved tree_nodes and root.
    """
    if n < 4:
        return tree_nodes, root
        
    start_time = time.time()
    all_leaves = frozenset(range(n))
    
    # Helper to get 4 sets of leaves for an internal edge
    def get_quartet_sets(u, v):
        """
        u is child, v is parent. Both internal.
        u has children u1, u2.
        v has children u, v1.
        Sets:
        A: leaves(u1)
        B: leaves(u2)
        C: leaves(v1)
        D: all_leaves - (A U B U C)
        """
        u1, u2 = tree_nodes[u]
        A = _get_leaves(u1, tree_nodes, n)
        B = _get_leaves(u2, tree_nodes, n)
        
        v1 = tree_nodes[v][0] if tree_nodes[v][1] == u else tree_nodes[v][1]
        C = _get_leaves(v1, tree_nodes, n)
        
        D = all_leaves - (A | B | C)
        if not D:
            return None
        return A, B, C, D

    # Helper to count quartet support
    # Returns (count_AB, count_AC, count_AD)
    # AB: (A,B)|(C,D) supported
    # AC: (A,C)|(B,D) supported
    # AD: (A,D)|(B,C) supported
    def count_quartet_support(A, B, C, D):
        # Get indices for each set
        idx_A = list(A)
        idx_B = list(B)
        idx_C = list(C)
        idx_D = list(D)
        
        # For speed, if any set is large, we might need optimization, but n<=32 so sets are small on average.
        # We iterate over sites.
        L = seq_matrix.shape[1]
        
        # To make this fast, we can use numpy if the sets are contiguous or we just loop.
        # Looping over L (up to 2000) and checking 4 values is O(L).
        # Doing this for every NNI move (O(n) moves) and every iteration (O(n) iterations) is O(n^2 * L).
        # n=32, L=2000 -> 32*32*2000 = 2 million operations per full pass. This is acceptable in Python if optimized.
        
        c_ab = 0
        c_ac = 0
        c_ad = 0
        
        # Pre-extract columns for A, B, C, D? 
        # Since A, B, C, D change, we can't pre-extract all.
        # But we can extract the rows for the specific taxa in A, B, C, D.
        
        # For a given site j:
        # s_a = seq_matrix[idx_A, j] -- wait, A is a set of taxa. 
        # In a quartet, A, B, C, D are sets of taxa.
        # The quartet topology is defined by the relationship between the *clusters* A, B, C, D.
        # A site supports (A,B)|(C,D) if all taxa in A have char X, all in B have char X, all in C have char Y, all in D have char Y, with X!=Y.
        # This is a strong condition. Usually, quartet counting assumes A, B, C, D are single taxa or we look at the "consensus" char?
        # No, standard quartet counting for NNI on multifurcations or clusters:
        # If A is a cluster of size > 1, we check if the site is "compatible" with the split.
        # A simpler and common heuristic for NNI in small trees is to pick a representative leaf from each set? 
        # Or better: check if the site is *informative* for the split (A U B) vs (C U D).
        # Actually, the most robust simple method is to count how many sites have the pattern where 
        # max(A) == max(B) != max(C) == max(D) ? No.
        
        # Let's use the definition: A site supports split S1|S2 if 
        # all taxa in S1 have the same character c1, all taxa in S2 have the same character c2, and c1 != c2.
        
        # This is O(|A| + |B| + |C| + |D|) per site.
        # Sum of sizes is n. So O(n * L) per quartet evaluation.
        # Total NNI steps ~ n. Total iterations ~ n.
        # O(n^3 * L). For n=32, L=2000: 32^3 * 2000 = 65 million operations. Might be too slow in pure Python.
        
        # Optimization: 
        # 1. Use numpy to check consistency of characters within a set for a given site.
        # 2. Or, just pick one representative taxon from each set? This is an approximation but often works well for topological accuracy.
        # Let's try the representative approach first for speed, as it's O(1) per site.
        # If A has multiple taxa, we pick the first one. This is a heuristic.
        
        rep_a = idx_A[0]
        rep_b = idx_B[0]
        rep_c = idx_C[0]
        rep_d = idx_D[0]
        
        # Extract columns for these 4 taxa
        # seq_matrix is (n, L). We want (4, L)
        cols = seq_matrix[[rep_a, rep_b, rep_c, rep_d], :]
        
        # For each site j (0 to L-1):
        # a, b, c, d = cols[:, j]
        # If a==b and c==d and a!=c: c_ab += 1
        # If a==c and b==d and a!=b: c_ac += 1
        # If a==d and b==c and a!=b: c_ad += 1
        
        a = cols[0]
        b = cols[1]
        c = cols[2]
        d = cols[3]
        
        # Vectorized counting
        # (a==b) & (c==d) & (a!=c)
        mask_ab = (a == b) & (c == d) & (a != c)
        c_ab = np.sum(mask_ab)
        
        mask_ac = (a == c) & (b == d) & (a != b)
        c_ac = np.sum(mask_ac)
        
        mask_ad = (a == d) & (b == c) & (a != b)
        c_ad = np.sum(mask_ad)
        
        return c_ab, c_ac, c_ad

    improved = True
    while improved:
        if time.time() - start_time > time_limit - 5:
            break
        improved = False
        
        # Rebuild parent map and internal edges
        parent_map = {root: None}
        def dfs(node):
            if node < n:
                return
            left, right = tree_nodes[node]
            parent_map[left] = node
            parent_map[right] = node
            dfs(left)
            dfs(right)
        dfs(root)
        
        internal_edges = [(c, p) for c, p in parent_map.items() if p is not None and c >= n and p >= n]
        
        for u, v in internal_edges:
            # u is child, v is parent
            sets = get_quartet_sets(u, v)
            if not sets:
                continue
            A, B, C, D = sets
            
            # Current topology: (A, B) | (C, D) is NOT necessarily the current one.
            # The current tree structure is:
            # v
            # | \
            # u  C
            # |\
            # A B
            # So the split defined by the edge (u, v) is (A U B) vs (C U D).
            # The quartet topology induced by this tree is (A, B) | (C, D).
            
            c_ab, c_ac, c_ad = count_quartet_support(A, B, C, D)
            
            # Current score is c_ab.
            # Alternative 1: (A, C) | (B, D) -> score c_ac
            # Alternative 2: (A, D) | (B, C) -> score c_ad
            
            if c_ac > c_ab:
                # Perform NNI to make (A, C) | (B, D)
                # New structure:
                # New internal node X joining A and C? No.
                # The edge (u, v) is replaced by (A, C) and (B, D) connected by a new edge?
                # Standard NNI:
                # Old: ((A, B), C) with D attached to v's parent? 
                # Wait, my definition of u, v:
                # u is child of v.
                # u has children A, B.
                # v has children u, C.
                # D is the rest of the tree attached to v's parent? 
                # NO. In my `get_quartet_sets`, D is `all_leaves - (A U B U C)`.
                # This implies the tree is rooted or we are considering the split.
                # The 4 subtrees attached to the internal edge (u, v) are A, B, C, and D.
                # Current topology: (A, B) are siblings. C is sibling of u. D is sibling of v (in the rooted sense? No, unrooted).
                # In an unrooted tree, the 4 neighbors of the edge (u, v) are A, B, C, D.
                # Current connections: u connects A, B, v. v connects u, C, D?
                # No, v is an internal node, so it has degree 3. Neighbors: u, C, and parent(v).
                # If parent(v) is the rest of the tree, then D is the subtree attached to parent(v) excluding v?
                # My `get_quartet_sets` logic:
                # A = leaves(u1), B = leaves(u2).
                # C = leaves(v1).
                # D = rest.
                # This assumes the tree is effectively "rooted" at the edge (u, v) for the purpose of identifying the 4 components.
                # The 4 components are indeed A, B, C, D.
                # Current topology connects u to A, B, v. v to u, C, D?
                # If D is the rest of the tree, then v is connected to D.
                # So current topology: (A, B) joined at u. u joined to v. v joined to C and D.
                # This corresponds to quartet (A, B) | (C, D).
                
                # New Topology 1: (A, C) | (B, D)
                # New Topology 2: (A, D) | (B, C)
                
                # To implement the NNI move:
                # We need to rewire the tree.
                # Let's create new internal nodes.
                
                if c_ac > c_ab:
                    # Swap to (A, C) | (B, D)
                    # New node X joins A and C? No.
                    # The edge (u, v) is removed.
                    # New edges: (A, B) is broken. (C, D) is broken?
                    # New connections:
                    # Node X1 joins A and B? No.
                    # Let's say we form (A, C) and (B, D).
                    # New internal node X joins A and C.
                    # New internal node Y joins B and D.
                    # Then X and Y are connected?
                    # No, that changes the degree.
                    
                    # Standard NNI on edge (u, v):
                    # u has children A, B. v has children u, C. (D is attached to v's parent? No, D is the 4th component).
                    # Actually, if we view the tree as unrooted, the edge (u, v) separates the tree into 4 components: A, B, C, D.
                    # Current: u connects A, B, v. v connects u, C, D.
                    # Alt 1: u connects A, C, v. v connects u, B, D. (This is (A,C)|(B,D))
                    # Alt 2: u connects A, D, v. v connects u, B, C. (This is (A,D)|(B,C))
                    
                    # So we just need to change the children of u and v.
                    # Currently:
                    # tree_nodes[u] = (A_node, B_node)  -- where A_node is the root of subtree A?
                    # Wait, A is a set of leaves. u is an internal node.
                    # tree_nodes[u] = (child_u1, child_u2).
                    # child_u1 is the root of A. child_u2 is the root of B.
                    # tree_nodes[v] = (u, child_v1).
                    # child_v1 is the root of C.
                    # Where is D?
                    # If v is not the root, v has a parent.
                    # If v is the root, it has no parent.
                    # My `get_quartet_sets` calculated D as `all_leaves - (A U B U C)`.
                    # This implies D is the rest of the tree.
                    # If v is the root, then D is empty? No, if v is root, it has 2 children. One is u, one is C. Then D is empty.
                    # But NNI requires 4 components. So v cannot be the root if D is non-empty.
                    # If D is non-empty, v must have a parent, or v is the root and my definition of C/D is wrong.
                    
                    # Let's assume the standard case: u is internal, v is internal.
                    # u's children: a_root, b_root.
                    # v's children: u, c_root.
                    # v's parent: p_root (which leads to D).
                    # If v is root, then D is empty, and we can't do NNI on (u, v) in the same way?
                    # Actually, if v is root, the edge (u, v) is a root edge. NNI on root edges is possible but different.
                    # For simplicity, let's only perform NNI on edges where both u and v are internal and v is NOT the root?
                    # Or handle the root case separately.
                    
                    # Let's check if v is root.
                    if v == root:
                         # D is empty?
                         # If v is root, it has 2 children. u and c_root.
                         # Then D is empty.
                         # So this edge is not a candidate for standard NNI (needs 4 components).
                         continue
                    else:
                         # v has a parent.
                         # The 4 components are A, B, C, D.
                         # To change to (A, C) | (B, D):
                         # u should connect to A and C.
                         # v should connect to u, B, and D?
                         # No, v is connected to its parent (which leads to D).
                         # So v is connected to: u, c_root, parent(v).
                         # If we want (B, D) to be a clade? No, (B, D) is a split.
                         # In the new topology:
                         # u connects A, C, v.
                         # v connects u, B, parent(v).
                         # This implies B is a child of v?
                         # But B was a child of u.
                         # So we move B from u to v.
                         # And move C from v to u.
                         
                         # Let's identify the node IDs for A, B, C, D.
                         # a_root, b_root = tree_nodes[u]
                         # c_root = tree_nodes[v][1] if tree_nodes[v][0] == u else tree_nodes[v][0]
                         # d_root is the parent of v? No, d_root is the node that represents D.
                         # Actually, we don't need d_root explicitly if we just swap children.
                         
                         # Current:
                         # u: (a_root, b_root)
                         # v: (u, c_root)
                         
                         # New for (A, C) | (B, D):
                         # u: (a_root, c_root)
                         # v: (u, b_root)
                         
                         # This is a simple swap of b_root and c_root between u and v.
                         
                         # Check if this is valid.
                         # Yes, this creates the topology where A and C are siblings (under u), and B and D (rest) are siblings (under v).
                         
                         # Perform swap
                         tree_nodes[u] = (a_root, c_root)
                         tree_nodes[v] = (u, b_root)
                         improved = True
                         break # Restart loop after change
                         
            elif c_ad > c_ab:
                # Swap to (A, D) | (B, C)
                # u: (a_root, d_root?)
                # v: (u, c_root)
                # We want A and D to be siblings.
                # D is the rest of the tree.
                # This is trickier because D is not a single child of v.
                # If v is not root, v's parent leads to D.
                # To make A and D siblings, we would need to move A to v's parent?
                # This is a more complex NNI.
                # For now, let's only implement the "simple" NNI where we swap two subtrees that are direct children of u and v.
                # The swap (A, C) | (B, D) is a simple swap of b_root and c_root.
                # The swap (A, D) | (B, C) would involve moving a_root to v's parent?
                # Let's stick to the simple swap for now, as it's the most common and easy to implement correctly.
                pass

    return tree_nodes, root

def _tree_to_newick(tree_nodes, root, labels):
    """Convert tree structure to Newick string."""
    def get_newick(node):
        if node < len(labels):
            return labels[node]
        left, right = tree_nodes[node]
        return f"({get_newick(left)},{get_newick(right)})"
    return get_newick(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0  # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"
            
    # Compute distance matrix and sequence matrix
    dist_mat, seq_matrix = _jc_distance_matrix(sequences, labels)
    
    # Use NJ for the initial tree
    tree_nodes, root = _nj_tree(dist_mat, labels)
    
    elapsed = time.time() - start_time
    
    # If we have time, perform NNI refinement
    if elapsed < time_limit - 10:
        remaining_time = time_limit - 5 - elapsed
        tree_nodes, root = _nni_refine(tree_nodes, root, n, seq_matrix, remaining_time)
        
    return _tree_to_newick(tree_nodes, root, labels)
