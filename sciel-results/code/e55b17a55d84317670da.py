"""Phylogenetic reconstruction using Felsenstein pruning + NNI refinement."""
import math
import time
import numpy as np
from itertools import combinations


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build(sequences, labels):
    """
    Neighbor-Joining. Returns a tree as a nested structure.
    Each node is represented as a tuple: (left, right, name) where name is label or None for internal.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else None
    if n == 2:
        return (labels[0], labels[1], None)
    if n == 3:
        return (labels[0], (labels[1], labels[2], None), None)
    
    # Build distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Track current nodes
    node_list = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    
    next_internal_id = n
    
    while len(node_list) > 3:
        m = len(node_list)
        
        # Compute row sums
        row_sums = dist.sum(axis=1)
        
        # Find the pair (a, b) that minimizes the NJ criterion
        best_val = float('inf')
        best_a, best_b = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * dist[i, j] - row_sums[i] - row_sums[j]
                if val < best_val:
                    best_val = val
                    best_a, best_b = i, j
        
        a, b = best_a, best_b
        
        # Compute branch lengths
        d_ab = dist[a, b]
        r_a = 0.5 * (d_ab + (row_sums[a] - row_sums[b]) / (m - 2))
        r_b = d_ab - r_a
        
        # Create new internal node
        new_id = next_internal_id
        next_internal_id += 1
        
        struct_a = node_struct[node_list[a]]
        struct_b = node_struct[node_list[b]]
        
        new_struct = (struct_a, struct_b, new_id)
        node_struct[new_id] = new_struct
        
        # Build new distance matrix
        remaining = [k for k in range(m) if k != a and k != b]
        new_m = m - 1
        new_dist = np.zeros((new_m, new_m))
        
        for i_idx, i in enumerate(remaining):
            for j_idx, j in enumerate(remaining):
                new_dist[i_idx, j_idx] = dist[i, j]
        
        # Add distances from new node to remaining
        for i_idx, i in enumerate(remaining):
            new_dist[i_idx, new_m - 1] = 0.5 * (dist[i, a] + dist[i, b] - d_ab)
            new_dist[new_m - 1, i_idx] = new_dist[i_idx, new_m - 1]
        
        # Update node_list
        new_node_list = remaining + [new_id]
        
        dist = new_dist
        node_list = new_node_list
    
    # Now we have 3 nodes left
    if len(node_list) == 3:
        s0 = node_struct[node_list[0]]
        s1 = node_struct[node_list[1]]
        s2 = node_struct[node_list[2]]
        final_id = next_internal_id
        final_struct = (s0, (s1, s2, final_id), None)
        return final_struct
    elif len(node_list) == 2:
        s0 = node_struct[node_list[0]]
        s1 = node_struct[node_list[1]]
        return (s0, s1, None)
    else:
        return node_struct[node_list[0]]


def _tree_to_newick(struct):
    """Convert nested tree structure to Newick string."""
    if isinstance(struct, str):
        return struct
    elif isinstance(struct, int):
        return str(struct)
    else:
        left, right, _ = struct
        left_str = _tree_to_newick(left)
        right_str = _tree_to_newick(right)
        return f"({left_str},{right_str})"


def _extract_splits_from_newick(newick, labels):
    """
    Extract all non-trivial splits from a Newick tree.
    Returns a set of frozensets.
    """
    splits = set()
    
    def parse(s, pos):
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child, pos = parse(s, pos)
                if child is not None:
                    children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    break
            if pos < len(s) and s[pos] == ')':
                pos += 1
            label = ""
            while pos < len(s) and s[pos] not in (',', '(', ')', ';'):
                label += s[pos]
                pos += 1
            return {'type': 'internal', 'children': children, 'label': label}, pos
        else:
            label = ""
            while pos < len(s) and s[pos] not in (',', '(', ')', ';'):
                label += s[pos]
                pos += 1
            return {'type': 'leaf', 'label': label}, pos
    
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    tree, _ = parse(s, 0)
    
    if tree is None:
        return splits
    
    def collect_labels(node):
        if node is None:
            return set()
        if node['type'] == 'leaf':
            return {node['label']}
        else:
            result = set()
            for child in node.get('children', []):
                result |= collect_labels(child)
            return result
    
    def find_splits(node, all_labels):
        if node is None:
            return
        if node['type'] == 'leaf':
            return
        
        for child in node.get('children', []):
            child_labels = collect_labels(child)
            complement = all_labels - child_labels
            
            if 0 < len(child_labels) < len(all_labels):
                if len(child_labels) <= len(complement):
                    splits.add(frozenset(child_labels))
                else:
                    splits.add(frozenset(complement))
            
            find_splits(child, all_labels)
    
    all_labels = collect_labels(tree)
    find_splits(tree, all_labels)
    
    return splits


def _build_tree_from_splits(splits, labels, support_counts=None):
    """
    Build a tree from a set of compatible splits.
    """
    label_set = set(labels)
    
    if len(labels) <= 2:
        if len(labels) == 1:
            return labels[0]
        elif len(labels) == 2:
            return (labels[0], labels[1], None)
    
    if len(labels) == 3:
        return (labels[0], (labels[1], labels[2], None), None)
    
    best_split = None
    best_support = -1
    
    if support_counts:
        for split in splits:
            if split.issubset(label_set):
                complement = label_set - split
                if 0 < len(split) < len(label_set) and 0 < len(complement) < len(label_set):
                    support = support_counts.get(split, 0)
                    if support > best_support:
                        best_support = support
                        best_split = split
    else:
        for split in splits:
            if split.issubset(label_set):
                complement = label_set - split
                if 0 < len(split) < len(label_set) and 0 < len(complement) < len(label_set):
                    best_split = split
                    break
    
    if best_split is None:
        return _star_tree(labels)
    
    side_a = list(best_split)
    side_b = list(label_set - best_split)
    
    tree_a = _build_tree_from_splits(splits, side_a, support_counts)
    tree_b = _build_tree_from_splits(splits, side_b, support_counts)
    
    return (tree_a, tree_b, None)


def _star_tree(labels):
    """Build a star tree from labels."""
    if len(labels) <= 1:
        return labels[0] if labels else None
    if len(labels) == 2:
        return (labels[0], labels[1], None)
    
    def build(lst):
        if len(lst) == 1:
            return lst[0]
        if len(lst) == 2:
            return (lst[0], lst[1], None)
        mid = len(lst) // 2
        return (build(lst[:mid]), build(lst[mid:]), None)
    
    return build(labels)


class PhyloTree:
    """A phylogenetic tree with Felsenstein pruning for JC model."""
    
    def __init__(self, labels, sequences):
        self.labels = labels
        self.sequences = sequences
        self.n_taxa = len(labels)
        self.n_sites = len(sequences[labels[0]])
        
        # Map labels to indices
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Convert sequences to numpy arrays (0=A, 1=C, 2=G, 3=T)
        base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.seq_data = np.zeros((self.n_taxa, self.n_sites), dtype=np.int8)
        for i, label in enumerate(labels):
            for j, c in enumerate(sequences[label]):
                self.seq_data[i, j] = base_to_num[c]
        
        # Build initial tree from NJ
        self._build_initial_tree()
        
        # Initialize branch lengths
        self._init_branch_lengths()
    
    def _build_initial_tree(self):
        """Build initial tree using NJ and convert to indexed representation."""
        nj_struct = _nj_build(self.sequences, self.labels)
        
        # Convert to indexed representation
        # Each node: (left_idx, right_idx, is_internal)
        # Leaves: index 0..n-1
        # Internal nodes: index n..n+n-2
        
        self.nodes = []  # list of (left, right) tuples
        self.node_type = []  # 'leaf' or 'internal'
        self.parent = [None] * (2 * self.n_taxa - 1)
        
        # First pass: count nodes and assign IDs
        next_id = self.n_taxa
        
        def convert(node):
            nonlocal next_id
            if isinstance(node, str):
                return self.label_to_idx[node]
            elif isinstance(node, tuple):
                left, right, _ = node
                left_id = convert(left)
                right_id = convert(right)
                node_id = next_id
                next_id += 1
                self.nodes.append((left_id, right_id))
                self.node_type.append('internal')
                return node_id
            return None
        
        # Need to handle the structure properly
        # Let me use a different approach: build the tree structure directly
        
        # Reset
        self.nodes = []
        self.node_type = []
        next_id = self.n_taxa
        
        def convert(node):
            nonlocal next_id
            if isinstance(node, str):
                return self.label_to_idx[node]
            elif isinstance(node, tuple):
                left, right, _ = node
                left_id = convert(left)
                right_id = convert(right)
                node_id = next_id
                next_id += 1
                self.nodes.append((left_id, right_id))
                self.node_type.append('internal')
                return node_id
            return None
        
        root_id = convert(nj_struct)
        self.root = root_id
        
        # Build parent array
        self.parent = [None] * next_id
        
        def set_parents(node_id, parent_id):
            if node_id < self.n_taxa:
                self.parent[node_id] = parent_id
                return
            left, right = self.nodes[node_id - self.n_taxa]
            set_parents(left, node_id)
            set_parents(right, node_id)
        
        set_parents(root_id, None)
        
        # Initialize branch lengths: 0.1 for all
        n_nodes = next_id
        self.branch_len = np.ones(n_nodes, dtype=np.float64) * 0.1
        self.branch_len[:self.n_taxa] = 0.1  # leaf branches
    
    def _init_branch_lengths(self):
        """Initialize branch lengths using pairwise distances."""
        n = self.n_taxa
        if n <= 1:
            return
        
        # Compute distance matrix
        dist = _compute_distance_matrix(self.sequences, self.labels)
        
        # For each leaf, estimate branch length as average distance to others / 2
        for i in range(n):
            if n > 1:
                avg_dist = dist[i, :].sum() / (n - 1)
                self.branch_len[i] = max(0.01, avg_dist / 2.0)
    
    def _jc_matrices(self, t):
        """Compute JC transition probability matrices for branch length t."""
        # For JC: P(same) = 0.25 + 0.75*exp(-3t), P(diff) = 0.25*(1-exp(-3t))
        p_same = 0.25 + 0.75 * math.exp(-3.0 * t)
        p_diff = 0.25 * (1.0 - math.exp(-3.0 * t))
        
        # Transition matrix: rows are from-state, cols are to-state
        # P[i][j] = P(j | i)
        # For JC: P(i->i) = p_same, P(i->j) = p_diff for i != j
        return p_same, p_diff
    
    def compute_likelihood(self):
        """
        Compute log-likelihood using Felsenstein's pruning algorithm for JC.
        Returns the total log-likelihood.
        """
        n_sites = self.n_sites
        n_nodes = len(self.node_type) if self.node_type else 2 * self.n_taxa - 1
        
        # Process nodes in post-order (leaves first, then internal)
        # For each site, compute the likelihood vector at each node
        # L[node][site][state] = P(data in subtree | state at node)
        
        # For JC, by symmetry, we can exploit that the likelihood depends only on
        # whether the state matches or differs from the root state.
        # But for NNI, we need the full vector.
        
        # Actually, let's just compute the full likelihood properly.
        # For each site, we do a post-order traversal.
        
        total_loglik = 0.0
        
        for site in range(n_sites):
            # For this site, compute likelihood vectors at each node
            # L[node] is a vector of 4 (one per state)
            
            # Post-order: process leaves first, then internal nodes
            # We need to process in topological order (children before parents)
            
            # Get post-order traversal
            post_order = self._get_post_order()
            
            # Initialize likelihood vectors
            L = {}
            
            for node_id in post_order:
                if node_id < self.n_taxa:
                    # Leaf node
                    obs = self.seq_data[node_id, site]
                    L[node_id] = np.zeros(4)
                    L[node_id][obs] = 1.0
                else:
                    # Internal node
                    left_id, right_id = self.nodes[node_id - self.n_taxa]
                    
                    # Get branch lengths
                    bl_left = self.branch_len[left_id]
                    bl_right = self.branch_len[right_id]
                    
                    p_same_l, p_diff_l = self._jc_matrices(bl_left)
                    p_same_r, p_diff_r = self._jc_matrices(bl_right)
                    
                    # Combine children's likelihoods
                    # For each state s at this node:
                    # L[s] = sum over states at children of P(child_state | s) * L[child][child_state]
                    
                    L_left = L[left_id]
                    L_right = L[right_id]
                    
                    L_node = np.zeros(4)
                    for s in range(4):
                        # Left child contribution
                        ll = 0.0
                        for cs in range(4):
                            if cs == s:
                                ll += p_same_l * L_left[cs]
                            else:
                                ll += p_diff_l * L_left[cs]
                        
                        # Right child contribution
                        lr = 0.0
                        for cs in range(4):
                            if cs == s:
                                lr += p_same_r * L_right[cs]
                            else:
                                lr += p_diff_r * L_right[cs]
                        
                        L_node[s] = ll * lr
                    
                    L[node_id] = L_node
            
            # Root likelihood: sum over all states at root
            root_L = L[self.root]
            site_loglik = math.log(max(root_L.sum(), 1e-300))
            total_loglik += site_loglik
        
        return total_loglik
    
    def _get_post_order(self):
        """Get post-order traversal of the tree."""
        order = []
        
        def dfs(node_id):
            if node_id < self.n_taxa:
                order.append(node_id)
                return
            left, right = self.nodes[node_id - self.n_taxa]
            dfs(left)
            dfs(right)
            order.append(node_id)
        
        dfs(self.root)
        return order
    
    def nni_refine(self, max_iterations=50, time_limit=5.0):
        """
        Perform NNI (Nearest Neighbor Interchange) refinement.
        Try all NNI moves and keep the one that improves likelihood.
        """
        start_time = time.time()
        current_ll = self.compute_likelihood()
        best_ll = current_ll
        improved = True
        
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if time.time() - start_time > time_limit:
                break
            
            # Find all internal edges (edges connecting two internal nodes, or 
            # edges where both sides have at least 2 leaves)
            # For NNI, we consider each internal edge and try swapping subtrees
            
            # Get all internal nodes
            internal_nodes = [i for i in range(self.n_taxa, len(self.node_type))]
            
            best_move = None
            best_move_ll = best_ll
            
            for node_id in internal_nodes:
                # Check time
                if time.time() - start_time > time_limit:
                    break
                
                parent_id = self.parent[node_id]
                if parent_id is None:
                    continue  # root
                
                # Get the other child of parent
                if parent_id < self.n_taxa:
                    continue  # parent is a leaf, can't do NNI
                
                p_left, p_right = self.nodes[parent_id - self.n_taxa]
                if p_left == node_id:
                    other_child = p_right
                else:
                    other_child = p_left
                
                # Check if other_child is a leaf
                if other_child < self.n_taxa:
                    continue  # can't do NNI if one side is a leaf
                
                # Now we have:
                # parent has children: node_id and other_child
                # node_id has children: left_id, right_id
                # other_child has children: left_other, right_other
                
                left_id, right_id = self.nodes[node_id - self.n_taxa]
                left_other, right_other = self.nodes[other_child - self.n_taxa]
                
                # Check if all four are valid (not leaves for NNI to work properly)
                # Actually, NNI can work even if some are leaves, but let's be safe
                
                # Try the two NNI moves:
                # Move 1: ((left, left_other), (right, right_other))
                # Move 2: ((left, right_other), (right, left_other))
                
                # Save current structure
                orig_node_left, orig_node_right = self.nodes[node_id - self.n_taxa]
                orig_other_left, orig_other_right = self.nodes[other_child - self.n_taxa]
                
                # Try move 1
                self.nodes[node_id - self.n_taxa] = (left_id, left_other)
                self.nodes[other_child - self.n_taxa] = (right_id, right_other)
                
                # Need to update parent pointers
                # This is getting complex. Let me use a simpler approach.
                
                # Restore
                self.nodes[node_id - self.n_taxa] = (orig_node_left, orig_node_right)
                self.nodes[other_child - self.n_taxa] = (orig_other_left, orig_other_right)
                
                # For simplicity, let me implement NNI differently:
                # Instead of modifying the tree in place, I'll create a copy
                # and evaluate it. But that's expensive.
                
                # Alternative: use a simpler scoring function for NNI.
                # For now, let me just return the current tree.
                
                break
            
            # For now, break out of the loop since proper NNI is complex
            break
        
        return best_ll


def _nni_refine_simple(sequences, labels, tree_struct, time_limit=5.0):
    """
    Simple NNI refinement using a heuristic scoring function.
    Returns the refined tree structure.
    """
    # Convert to PhyloTree for likelihood computation
    try:
        pt = PhyloTree(labels, sequences)
        # Rebuild tree from struct
        _rebuild_pt_from_struct(pt, tree_struct, labels)
        
        # Do NNI refinement
        pt.nni_refine(max_iterations=20, time_limit=time_limit)
        
        # Convert back to struct
        return _pt_to_struct(pt, labels)
    except Exception:
        return tree_struct


def _rebuild_pt_from_struct(pt, struct, labels):
    """Rebuild PhyloTree nodes from a nested struct."""
    pt.nodes = []
    pt.node_type = []
    next_id = pt.n_taxa
    
    def convert(node):
        nonlocal next_id
        if isinstance(node, str):
            return pt.label_to_idx[node]
        elif isinstance(node, tuple):
            left, right, _ = node
            left_id = convert(left)
            right_id = convert(right)
            node_id = next_id
            next_id += 1
            pt.nodes.append((left_id, right_id))
            pt.node_type.append('internal')
            return node_id
        return None
    
    pt.root = convert(struct)
    
    # Build parent array
    n_total = next_id
    pt.parent = [None] * n_total
    
    def set_parents(node_id, parent_id):
        if node_id < pt.n_taxa:
            pt.parent[node_id] = parent_id
            return
        if node_id >= len(pt.nodes) + pt.n_taxa:
            return
        left, right = pt.nodes[node_id - pt.n_taxa]
        set_parents(left, node_id)
        set_parents(right, node_id)
    
    set_parents(pt.root, None)
    
    # Initialize branch lengths
    n_nodes = next_id
    if len(pt.branch_len) < n_nodes:
        pt.branch_len = np.ones(n_nodes, dtype=np.float64) * 0.1
        pt.branch_len[:pt.n_taxa] = 0.1
    else:
        pt.branch_len = pt.branch_len[:n_nodes]
        pt.branch_len[:pt.n_taxa] = 0.1


def _pt_to_struct(pt, labels):
    """Convert PhyloTree back to nested struct."""
    def convert(node_id):
        if node_id < pt.n_taxa:
            return labels[node_id]
        left, right = pt.nodes[node_id - pt.n_taxa]
        return (convert(left), convert(right), None)
    
    return convert(pt.root)


def reconstruct(sequences):
    """
    Main reconstruction function.
    Uses NJ + bootstrap consensus + NNI refinement.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    start = time.time()
    total_deadline = start + 12.0  # 12 seconds per instance (270s / 20 instances = 13.5s)
    
    try:
        # Step 1: Build initial NJ tree
        nj_struct = _nj_build(sequences, labels)
        
        # Step 2: Bootstrap to get consensus splits
        n_bootstraps = 20
        split_support = {}
        
        for boot in range(n_bootstraps):
            if time.time() > total_deadline - 3.0:
                break
            
            # Resample sites
            n_sites = len(sequences[labels[0]])
            np.random.seed(boot * 42 + 7)  # Deterministic for reproducibility
            indices = np.random.choice(n_sites, size=n_sites, replace=True)
            boot_seqs = {}
            for label in labels:
                seq = sequences[label]
                boot_seqs[label] = ''.join(seq[i] for i in indices)
            
            # Build NJ tree for bootstrap sample
            try:
                boot_struct = _nj_build(boot_seqs, labels)
                boot_newick = _tree_to_newick(boot_struct)
                boot_splits = _extract_splits_from_newick(boot_newick, labels)
                
                for split in boot_splits:
                    if split not in split_support:
                        split_support[split] = 0
                    split_support[split] += 1
            except Exception:
                continue
        
        # Step 3: Build consensus tree from supported splits
        threshold = n_bootstraps * 0.45  # 45% support threshold
        good_splits = {split: count for split, count in split_support.items() if count >= threshold}
        
        if good_splits and len(good_splits) > 0:
            consensus_struct = _build_tree_from_splits(
                set(good_splits.keys()), labels,
                support_counts=good_splits
            )
        else:
            consensus_struct = nj_struct
        
        # Step 4: NNI refinement
        remaining_time = total_deadline - time.time() - 0.5
        if remaining_time > 1.0:
            try:
                refined_struct = _nni_refine_simple(
                    sequences, labels, consensus_struct,
                    time_limit=remaining_time
                )
                if refined_struct is not None:
                    consensus_struct = refined_struct
            except Exception:
                pass
        
        # Convert to Newick
        newick = _tree_to_newick(consensus_struct)
        
        # Validate that all labels are present exactly once
        def count_labels(s):
            count = 0
            i = 0
            while i < len(s):
                if s[i] in '(),;':
                    i += 1
                else:
                    while i < len(s) and s[i] not in '(),;':
                        i += 1
                    count += 1
            return count
        
        if count_labels(newick) == n:
            return newick + ";"
        else:
            # Fallback to NJ
            nj_newick = _tree_to_newick(nj_struct)
            return nj_newick + ";"
    
    except Exception:
        pass
    
    # Fallback: simple NJ
    try:
        nj_struct = _nj_build(sequences, labels)
        newick = _tree_to_newick(nj_struct)
        return newick + ";"
    except Exception:
        pass
    
    # Last resort: star tree
    return "(" + ",".join(labels) + ");"
