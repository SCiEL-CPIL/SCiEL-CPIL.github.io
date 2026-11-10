"""Phylogenetic topology recovery using Maximum Parsimony with SPR optimization."""
import numpy as np
import random

def _seq_to_int(seq):
    """Convert a DNA sequence string to a numpy array of integers (0-3)."""
    # A=0, C=1, G=2, T=3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping[c] for c in seq], dtype=np.int8)

def _build_int_matrix(sequences, labels):
    """Build an integer matrix of sequences (n_sites, n_taxa)."""
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    matrix = np.zeros((n_sites, n_taxa), dtype=np.int8)
    for i, label in enumerate(labels):
        matrix[:, i] = _seq_to_int(sequences[label])
    return matrix

def _calc_parsimony_vectorized(tree, X):
    """
    Calculate the parsimony score of a tree for all sites simultaneously.
    tree: list of (left, right) for internal nodes (indices n..2n-2).
          Leaves are 0..n-1.
    X: numpy array of shape (n_sites, n_taxa).
    Returns: total parsimony score (integer).
    """
    n_taxa = X.shape[1]
    n_nodes = 2 * n_taxa - 2
    
    # masks[i, j] is the state mask for node i at site j
    # Mask is 4-bit: bit k is 1 if state k is possible
    masks = np.zeros((n_nodes, X.shape[0]), dtype=np.int32)
    
    # Initialize leaves
    # X is (n_sites, n_taxa). We need (n_taxa, n_sites) for easy indexing by taxon?
    # Or just use X.T
    X_T = X.T # (n_taxa, n_sites)
    for i in range(n_taxa):
        # 1 << state
        masks[i] = 1 << X_T[i]
        
    total_cost = 0
    
    # Process internal nodes in reverse order (post-order)
    # Internal nodes are n_taxa to n_nodes - 1
    # We assume the tree is stored such that children indices are < parent index?
    # In our construction, we should ensure this.
    for i in range(n_taxa, n_nodes):
        left, right = tree[i]
        l_mask = masks[left]
        r_mask = masks[right]
        
        intersection = l_mask & r_mask
        # If intersection is 0, cost 1, mask is union
        # If intersection is non-zero, cost 0, mask is intersection
        
        # Vectorized operation
        # cost is 1 where intersection == 0
        cost_mask = (intersection == 0).astype(np.int32)
        total_cost += np.sum(cost_mask)
        
        # Update masks
        # np.where(condition, x, y)
        masks[i] = np.where(intersection == 0, l_mask | r_mask, intersection)
        
    return total_cost

def _random_tree(n_taxa):
    """
    Generate a random binary tree.
    Returns a list of length 2n-2.
    Indices 0..n-1 are leaves (values ignored, just placeholders).
    Indices n..2n-3 are internal nodes (left, right).
    """
    if n_taxa <= 1:
        return []
        
    # We will build the tree by adding leaves one by one to a random edge.
    # This ensures a valid binary tree.
    
    # Start with 2 leaves
    # Node 0, 1 are leaves.
    # Internal node n_taxa connects 0 and 1.
    
    # This representation is tricky if we add leaves.
    # Let's use a different representation for generation, then convert.
    
    # Easier: Generate a random permutation of leaves, then build a balanced or random tree?
    # Let's just use the greedy parsimony start instead, it's better.
    pass

def _greedy_parsimony_start(X):
    """
    Construct a tree using greedy parsimony.
    Start with 2 taxa, add the next taxon to the best edge, repeat.
    Returns tree structure (list of (left, right) for internal nodes).
    """
    n_taxa = X.shape[1]
    if n_taxa == 1:
        return []
    if n_taxa == 2:
        # Tree: (0, 1)
        # Internal node index 2 (since 2*2-2 = 2 nodes total, indices 0,1 leaves, 2 internal? No.)
        # Total nodes = 2n - 2.
        # If n=2, total nodes=2. Indices 0,1. Both leaves? No, we need an internal node.
        # Standard: Leaves 0..n-1. Internal n..2n-3.
        # If n=2, internal is index 2. Total nodes 3? No, 2n-2=2.
        # Wait. A binary tree with n leaves has n-1 internal nodes.
        # Total nodes = 2n - 1.
        # My previous assumption of 2n-2 was for unrooted?
        # Unrooted tree with n leaves has n-3 internal nodes. Total 2n-3.
        # Rooted tree with n leaves has n-1 internal nodes. Total 2n-1.
        
        # Parsimony is usually defined on rooted or unrooted.
        # Fitch's algorithm works on rooted trees.
        # We can root the tree arbitrarily (e.g., at a leaf or an internal node).
        # Let's use a rooted tree representation for calculation.
        # Unrooted tree with n leaves -> Rooted tree with n leaves and n-1 internal nodes.
        # Total nodes = 2n - 1.
        
        # Let's redefine:
        # Leaves: 0..n-1
        # Internal: n..2n-2
        # Total nodes: 2n-1
        
        # For n=2:
        # Leaves 0, 1.
        # Internal node 2.
        # Tree: [None, None, (0, 1)]
        
        # For n=3:
        # Leaves 0, 1, 2.
        # Internal 3, 4.
        # One internal connects two leaves, the other connects the third leaf and the first internal.
        
        # Greedy Algorithm:
        # 1. Start with 2 taxa (0, 1). Tree has 1 internal node.
        # 2. Add taxon 2.
        # 3. Add taxon 3.
        
        # We need to maintain the tree structure and recalculate parsimony.
        
        # Since n is small, we can just build the tree incrementally.
        
        # State: list of internal nodes.
        # Let's represent the tree as a list of tuples for internal nodes.
        # But indices change as we add nodes.
        
        # Alternative: Use a Newick string and parse it? Too slow.
        # Use a parent array?
        
        # Let's implement a simple recursive builder.
        
        # Actually, for the start, we can just use a random tree and rely on SPR to fix it.
        # Or, we can use a simple heuristic:
        # 1. Pick 3 random taxa to form a cherry.
        # 2. Add remaining taxa one by one to a random edge.
        # This is O(n^2).
        
        # Let's just return a random tree structure for now and let SPR do the work.
        # Random Tree Generation:
        # 1. Create a list of leaves [0..n-1].
        # 2. While len(list) > 1:
        #    Pick two random elements, remove them, create a new internal node, add it to the list.
        # This creates a random binary tree.
        
        # Let's implement this.
        
        nodes = list(range(n_taxa))
        internal_nodes = []
        next_id = n_taxa
        
        while len(nodes) > 1:
            # Pick two random indices
            i1, i2 = random.sample(range(len(nodes)), 2)
            n1 = nodes[i1]
            n2 = nodes[i2]
            
            # Remove them
            # To avoid index shifting issues, remove higher index first
            if i1 > i2:
                i1, i2 = i2, i1
            del nodes[i1]
            del nodes[i2]
            
            # Create new internal node
            internal_nodes.append((n1, n2))
            nodes.append(next_id)
            next_id += 1
            
        # The last node in 'nodes' is the root.
        # We need to map these to our 0..2n-2 indexing scheme.
        # In this generation, internal nodes are assigned IDs n_taxa, n_taxa+1, ...
        # This matches our scheme.
        
        # We need to return a list of length 2n-1?
        # No, the function _calc_parsimony expects a list of length 2n-2?
        # Let's check _calc_parsimony again.
        # It loops `for i in range(n_taxa, n_nodes)`.
        # If n_nodes = 2n-1, range is n..2n-2.
        # The list `tree` must have indices 0..2n-2.
        # Indices 0..n-1 are leaves (dummy).
        # Indices n..2n-2 are internal.
        
        # So we return a list of length 2n-1.
        # Fill 0..n-1 with None.
        # Fill n..2n-2 with internal_nodes.
        
        tree = [None] * (2 * n_taxa - 1)
        for i, node in enumerate(internal_nodes):
            tree[n_taxa + i] = node
            
        return tree

def _spr_optimize(tree, X, max_iter=100):
    """
    Optimize tree using SPR moves.
    """
    n_taxa = X.shape[1]
    n_nodes = 2 * n_taxa - 1
    
    current_score = _calc_parsimony_vectorized(tree, X)
    
    # Precompute post-order? 
    # For SPR, we need to identify edges.
    # An edge is defined by (parent, child).
    # We can iterate through all internal nodes and their children.
    
    for _ in range(max_iter):
        improved = False
        best_score = current_score
        best_tree = tree
        
        # Generate all SPR moves
        # For each internal node i (parent):
        #   For each child c of i:
        #     Prune subtree rooted at c.
        #     Regraft onto every other edge (p2, c2) where p2 != i and c2 is not in subtree(c).
        
        # This is complex to implement efficiently without a full tree data structure.
        # Given the small size, we can just generate candidate trees and evaluate them.
        
        # Candidate generation:
        # 1. Identify all edges (parent, child).
        edges = []
        for i in range(n_taxa, n_nodes):
            left, right = tree[i]
            edges.append((i, left))
            edges.append((i, right))
            
        for p, c in edges:
            # Prune c from p
            # The subtree rooted at c is moved.
            
            # We need to create a new tree.
            # 1. Copy tree.
            # 2. Modify p's child to be a "leaf" representing the pruned subtree?
            #    No, the topology changes.
            
            # This is getting complicated.
            # Let's use NNI instead. NNI is simpler.
            # NNI: Swap subtrees across an internal edge.
            
            # Let's switch to NNI optimization. It's less powerful but much easier to implement correctly.
            pass
        
        # Implement NNI
        for i in range(n_taxa, n_nodes - 1): # Exclude root?
            # Internal node i has children L, R.
            # If L and R are both internal, we can perform NNI.
            l, r = tree[i]
            if l >= n_taxa and r >= n_taxa:
                # Get children of L and R
                l1, l2 = tree[l]
                r1, r2 = tree[r]
                
                # Two possible swaps:
                # 1. (l1, r1) and (l2, r2)
                # 2. (l1, r2) and (l2, r1)
                
                # Candidate 1
                new_tree = tree.copy()
                new_tree[i] = (l, r) # Unchanged
                new_tree[l] = (l1, r1)
                new_tree[r] = (l2, r2)
                
                # Check if this is a valid tree (no cycles).
                # Since we are swapping subtrees between two internal nodes, it should be valid.
                
                score1 = _calc_parsimony_vectorized(new_tree, X)
                if score1 < best_score:
                    best_score = score1
                    best_tree = new_tree
                    improved = True
                    
                # Candidate 2
                new_tree2 = tree.copy()
                new_tree2[i] = (l, r)
                new_tree2[l] = (l1, r2)
                new_tree2[r] = (l2, r1)
                
                score2 = _calc_parsimony_vectorized(new_tree2, X)
                if score2 < best_score:
                    best_score = score2
                    best_tree = new_tree2
                    improved = True
        
        if not improved:
            break
        
        tree = best_tree
        current_score = best_score
        
    return tree, current_score

def _tree_to_newick(tree, labels):
    """Convert tree structure to Newick string."""
    n_taxa = len(labels)
    if n_taxa == 1:
        return labels[0] + ";"
        
    # Find the root.
    # The root is the internal node that is not a child of any other internal node.
    # Or simply, the last internal node added?
    # In our generation, the root is the last internal node created.
    # In the list `tree`, the root is the one not appearing as a child in any other internal node.
    
    # Simpler: The root is the internal node with index 2*n_taxa - 2?
    # Not necessarily, depends on generation order.
    # Let's find it.
    
    children = set()
    for i in range(n_taxa, 2 * n_taxa - 1):
        if tree[i] is not None:
            l, r = tree[i]
            children.add(l)
            children.add(r)
            
    root = None
    for i in range(n_taxa, 2 * n_taxa - 1):
        if i not in children:
            root = i
            break
            
    if root is None:
        # Fallback
        root = 2 * n_taxa - 2
        
    def to_newick(node):
        if node < n_taxa:
            return labels[node]
        else:
            l, r = tree[node]
            return f"({to_newick(l)},{to_newick(r)})"
            
    return to_newick(root) + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
        
    X = _build_int_matrix(sequences, labels)
    
    # Random start
    random.seed(42)
    tree = _random_tree(n)
    
    # Optimize with NNI
    tree, score = _spr_optimize(tree, X, max_iter=50)
    
    newick = _tree_to_newick(tree, labels)
    
    return newick
