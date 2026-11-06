# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = None
_DEADLINE_FRACTION = 0.85

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _calc_likelihood(tree, arr, branch_lengths, labels):
    """Calculate log-likelihood of the tree given the sequences under JC model."""
    n_taxa = len(labels)
    L = arr.shape[1]
    
    # Build a map from label to index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Parse the Newick tree to get the structure
    # We need to reconstruct the tree structure from the Newick string
    # For efficiency, we'll use a simpler approach: compute likelihood based on the splits
    
    # Actually, for NNI refinement, we need the tree structure with internal nodes
    # Let's build a proper tree representation
    
    # Parse Newick
    def parse_newick(newick):
        """Parse Newick string and return tree structure."""
        # Remove trailing semicolon
        newick = newick.strip()
        if newick.endswith(';'):
            newick = newick[:-1]
        
        # We'll use a recursive parser
        # Returns: list of (node_id, [children], is_leaf, label)
        nodes = []
        next_id = [0]
        
        def parse(s, pos):
            """Parse starting at position pos, return (node_id, new_pos)."""
            if s[pos] == '(':
                pos += 1
                children = []
                while True:
                    child_id, pos = parse(s, pos)
                    children.append(child_id)
                    if s[pos] == ',':
                        pos += 1
                    elif s[pos] == ')':
                        pos += 1
                        break
                    else:
                        break
                # Check for branch length after closing paren
                if pos < len(s) and s[pos] == ':':
                    pos += 1
                    # Parse branch length
                    start = pos
                    while pos < len(s) and s[pos] not in ',)':
                        pos += 1
                    bl = float(s[start:pos])
                else:
                    bl = 0.0
                node_id = next_id[0]
                next_id[0] += 1
                nodes.append((node_id, children, False, None, bl))
                return node_id, pos
            else:
                # Leaf
                start = pos
                while pos < len(s) and s[pos] not in ':,):':
                    pos += 1
                label = s[start:pos]
                if pos < len(s) and s[pos] == ':':
                    pos += 1
                    start = pos
                    while pos < len(s) and s[pos] not in ',)':
                        pos += 1
                    bl = float(s[start:pos])
                else:
                    bl = 0.0
                node_id = next_id[0]
                next_id[0] += 1
                nodes.append((node_id, [], True, label, bl))
                return node_id, pos
        
        root_id, _ = parse(newick, 0)
        return nodes, root_id
    
    nodes, root_id = parse_newick(tree)
    
    # Build adjacency
    n_nodes = len(nodes)
    children_map = {}
    parent_map = {}
    for node_id, children, is_leaf, label, bl in nodes:
        children_map[node_id] = children
        for c in children:
            parent_map[c] = node_id
    
    # Post-order traversal for Felsenstein algorithm
    order = []
    def dfs(node_id):
        for c in children_map[node_id]:
            dfs(c)
        order.append(node_id)
    dfs(root_id)
    
    # Felsenstein's pruning algorithm
    # For JC model, the transition probabilities are:
    # P(same) = 0.25 + 0.75 * exp(-4/3 * d)
    # P(diff) = 0.25 * (1 - exp(-4/3 * d))
    
    # For each site, compute the likelihood
    log_likelihood = 0.0
    
    for site in range(L):
        # Forward pass: compute likelihoods at each node
        # P(node | data below node)
        # For JC, we can work with 4 states, but due to symmetry, we can simplify
        
        # Actually, for JC with 4 equally likely states, the likelihood calculation
        # can be done more efficiently. Let's use the standard approach.
        
        # Initialize leaf likelihoods
        node_likelihoods = {}
        for node_id in order:
            node_id, children, is_leaf, label, bl = nodes[node_id]
            if is_leaf:
                taxon_idx = label_to_idx[label]
                obs = arr[taxon_idx, site]
                # Likelihood of observing this base at this leaf
                # Under JC, all bases are equally likely, so P(obs) = 0.25
                # But we need the conditional likelihood given the parent state
                # For a leaf with observation obs:
                # P(obs | parent_state) = 0.25 if obs == parent_state (after considering branch)
                # Actually, we need to be more careful.
                
                # Let's use the standard Felsenstein algorithm.
                # For a leaf, the likelihood vector is:
                # P(s | data) where s is the state at this node
                # If this node is a leaf with observed state obs:
                # P(obs | state=s) = 1 if s == obs, 0 otherwise
                # But we need to account for the branch from parent to this node.
                
                # For simplicity and efficiency with JC model, let's use a different approach.
                
                # Actually, let's just compute the likelihood using the distance-based approach
                # For JC, the likelihood can be computed from the pairwise distances
                # But that's not straightforward for a tree.
                
                # Let me implement the proper Felsenstein algorithm.
                
                # For a leaf, the vector of likelihoods (probability of data below given state at this node)
                # is: [1, 0, 0, 0] if obs == 0, etc.
                # But we need to propagate up the tree.
                
                # Let's store the likelihood vector for each node
                # Each vector has 4 elements (one for each possible state)
                
                # For a leaf with observed state obs:
                # The likelihood of the data below (just this leaf) given state s at this node
                # is: 1 if s == obs, 0 otherwise
                # But we need to multiply by the transition probability along the branch to the parent.
                
                # Let's do this properly.
                
                # Initialize: for each leaf, likelihood vector = indicator of observed state
                vec = np.zeros(4)
                vec[obs] = 1.0
                node_likelihoods[node_id] = vec
            else:
                # Internal node: combine children
                # First, for each child, propagate the likelihood up the branch
                # Then, multiply the vectors (element-wise) for all children
                # Finally, the likelihood at this node is the product of child contributions
                
                # For each child, we need to compute:
                # P(state at this node = s | data below child) = sum over child state c of
                # P(c -> s | branch_length) * P(state at child = c | data below child)
                
                # Due to JC symmetry, we can simplify this.
                
                # For now, let's just return 0 as a placeholder and use a different approach
                # for NNI refinement. We'll use the splits-based RF distance instead.
                node_likelihoods[node_id] = None
        
        # This is getting complex. Let's use a simpler approach for NNI refinement:
        # Use the Robinson-Foulds distance to guide the search, and just try
        # a few NNI moves and keep the one that minimizes RF distance to the NJ tree.
        
        # Actually, let's just return a dummy value for now
        return 0.0
    
    return log_likelihood

def _get_splits_from_newick(newick, n_taxa):
    """Extract non-trivial splits from a Newick string."""
    # Parse the Newick string to get the tree structure
    # For simplicity, we'll use a different approach: use the splits from NJ
    
    # Actually, let's just return an empty set for now and use a different strategy
    return set()

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

def _rf_distance(s1, s2):
    return len(s1.symmetric_difference(s2))

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    # For now, just return the NJ tree. We could add bootstrap consensus later.
    return tree_full
# EVOLVE-BLOCK-END
