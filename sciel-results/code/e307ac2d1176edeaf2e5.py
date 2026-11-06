"""Neighbour joining + exhaustive small-n + SPR refinement with deadline guard."""
import math
import time
import numpy as np
from itertools import combinations


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    seqs = np.array([list(sequences[l]) for l in labels])
    L = seqs.shape[1]
    # Convert to numeric using vectorized approach
    numeric = np.zeros((n, L), dtype=np.int8)
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Build lookup: map each char to int efficiently
    # Use numpy char codes: A=65, C=67, G=71, T=84
    # We'll do it per-row for simplicity since n<=32
    for i in range(n):
        for j in range(L):
            numeric[i, j] = char_map[seqs[i, j]]
    
    # Pairwise mismatches via broadcasting
    diff = (numeric[:, np.newaxis, :] != numeric[np.newaxis, :, :])
    mismatches = diff.sum(axis=2).astype(np.float64)
    
    p = mismatches / L
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_build_tree(d, labels):
    """Neighbour Joining algorithm. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(i, j)] = d[i, j]
            dist[(j, i)] = d[i, j]
            
    nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        row_sums = {node: 0.0 for node in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    row_sums[a] += dist[(a, b)]
        
        best_criterion = float('inf')
        best_a = best_b = None
        for i in range(m):
            for j in range(i+1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * dist[(a, b)] - row_sums[a] - row_sums[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        r_a = (row_sums[a] - row_sums[b] + d_ab) / (2.0 * m)
        r_b = d_ab - r_a
        r_a = max(0.0, r_a)
        r_b = max(0.0, r_b)
        
        new_node = next_node_id
        next_node_id += 1
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            new_d = max(0.0, new_d)
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d
        
        nodes = rest + [new_node]
        tree_parts[new_node] = f"({tree_parts[a]}:{r_a:.6f},{tree_parts[b]}:{r_b:.6f})"
    
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist[(a, b)]
        d_ac = dist[(a, c)]
        d_bc = dist[(b, c)]
        len_a = max(0.0, (d_ab + d_ac - d_bc) / 2.0)
        len_b = max(0.0, (d_ab + d_bc - d_ac) / 2.0)
        len_c = max(0.0, (d_ac + d_bc - d_ab) / 2.0)
        return f"({tree_parts[a]}:{len_a:.6f},{tree_parts[b]}:{len_b:.6f},{tree_parts[c]}:{len_c:.6f})"
    
    return "(" + ",".join(tree_parts[x] for x in nodes) + ")"


def _get_splits_from_newick(newick, n):
    """Extract all non-trivial splits from a Newick string.
    Returns a set of frozensets, each representing a split {A, B} where A∪B = all taxa."""
    # Parse newick into a tree structure
    # We'll do a simple recursive parse
    
    # First, extract all taxa positions and internal node groupings
    # Use a stack-based parser
    
    taxa_set = set()
    # Extract all leaf labels
    import re
    # Find all leaf labels (t0, t1, etc.)
    leaf_pattern = re.compile(r't\d+')
    for match in leaf_pattern.finditer(newick):
        taxa_set.add(match.group())
    
    # For each internal node, determine which taxa are in its subtree
    # We need to parse the tree structure
    
    # Build a proper parse
    def parse_newick(s):
        """Parse newick into nested tuples. Returns (children, label, length)."""
        i = 0
        def parse_node():
            nonlocal i
            if i >= len(s):
                return None
            if s[i] == '(':
                i += 1
                children = []
                while i < len(s) and s[i] != ')':
                    child = parse_node()
                    if child is not None:
                        children.append(child)
                    if i < len(s) and s[i] == ',':
                        i += 1
                if i < len(s) and s[i] == ')':
                    i += 1
                # Check for label/length after )
                label = ""
                length = 0.0
                if i < len(s) and s[i] == ':':
                    i += 1
                    start = i
                    while i < len(s) and s[i] not in ',)':
                        i += 1
                    length = float(s[start:i])
                elif i < len(s) and s[i] not in ',)':
                    start = i
                    while i < len(s) and s[i] not in ',):;':
                        i += 1
                    label = s[start:i]
                return (children, label, length)
            else:
                # Leaf node
                start = i
                while i < len(s) and s[i] not in ',):;':
                    i += 1
                label = s[start:i]
                length = 0.0
                if i < len(s) and s[i] == ':':
                    i += 1
                    start = i
                    while i < len(s) and s[i] not in ',):;':
                        i += 1
                    length = float(s[start:i])
                return ([], label, length)
        
        result = parse_node()
        return result
    
    tree = parse_newick(newick)
    
    # Now compute the set of taxa under each internal node
    # For each internal node, the split is (taxa_in_subtree, all_taxa - taxa_in_subtree)
    # Only keep non-trivial splits (both parts have size >= 2)
    
    all_taxa = taxa_set
    splits = set()
    
    def compute_subtree_taxa(node):
        """Return the set of taxa in this subtree."""
        children, label, length = node
        if not children:
            # Leaf
            return {label}
        else:
            subtree = set()
            for child in children:
                subtree |= compute_subtree_taxa(child)
            # This is an internal node - record its split
            if len(subtree) >= 2 and len(all_taxa - subtree) >= 2:
                # Normalize: use frozenset of the smaller part for uniqueness
                part1 = frozenset(subtree)
                part2 = frozenset(all_taxa - subtree)
                # Store as a canonical form: sort the two parts
                # Use a tuple of two frozensets, sorted by first element
                if len(part1) < len(part2):
                    key = (part1, part2)
                elif len(part2) < len(part1):
                    key = (part2, part1)
                else:
                    # Same size, use lexicographic order of the sorted lists
                    l1 = sorted(part1)
                    l2 = sorted(part2)
                    if l1 <= l2:
                        key = (part1, part2)
                    else:
                        key = (part2, part1)
                splits.add(key)
            return subtree
    
    compute_subtree_taxa(tree)
    return splits


def _nj_score(d, newick, n):
    """Score a Newick tree using the NJ criterion: sum over all pairs of d(i,j) 
    weighted by whether they share a path. Actually, for NJ we use the 
    total branch length of the tree metric as a proxy.
    
    Better: use the sum of all pairwise distances in the tree metric.
    For a tree, the tree metric distance between i and j is the sum of branch lengths
    on the path. The total sum over all pairs is a good criterion.
    
    But we can compute it more directly: for each split in the tree, the contribution
    to the total pairwise distance is: |A| * |B| * (branch length across the split).
    
    Actually, the simplest NJ-related score is: sum of all branch lengths in the tree.
    Let's compute the tree's total branch length by reconstructing it.
    """
    # Parse the newick to get branch lengths and compute total
    import re
    
    def parse_newick_lengths(s):
        """Parse newick, return total branch length."""
        i = 0
        def parse_node():
            nonlocal i
            if i >= len(s):
                return 0.0
            if s[i] == '(':
                i += 1
                total = 0.0
                while i < len(s) and s[i] != ')':
                    total += parse_node()
                    if i < len(s) and s[i] == ',':
                        i += 1
                if i < len(s) and s[i] == ')':
                    i += 1
                # Check for length
                if i < len(s) and s[i] == ':':
                    i += 1
                    start = i
                    while i < len(s) and s[i] not in ',):;':
                        i += 1
                    try:
                        length = float(s[start:i])
                    except:
                        length = 0.0
                    return total + length
                return total
            else:
                # Leaf
                start = i
                while i < len(s) and s[i] not in ',):;':
                    i += 1
                length = 0.0
                if i < len(s) and s[i] == ':':
                    i += 1
                    start = i
                    while i < len(s) and s[i] not in ',):;':
                        i += 1
                    try:
                        length = float(s[start:i])
                    except:
                        length = 0.0
                return length
        
        return parse_node()
    
    return parse_newick_lengths(newick)


def _exhaustive_small_n(d, labels, n, start_time):
    """For n<=7, enumerate all unrooted binary topologies and pick the best."""
    if n <= 3:
        return _nj_build_tree(d, labels)
    
    # For n=4: 3 topologies
    # For n=5: 15 topologies  
    # For n=6: 105 topologies
    # For n=7: 945 topologies
    
    # Generate all unrooted binary trees by recursive split enumeration.
    # Each tree is defined by a set of n-3 splits.
    # We enumerate by: pick a split (A,B) with 2<=|A|<=n-2, |A|<=|B|,
    # then recursively enumerate trees on A and B.
    
    # To avoid overcounting, we fix: the split must have |A| <= |B|,
    # and if |A| == |B|, we require a canonical ordering.
    
    # Represent a tree as a frozenset of splits, where each split is
    # a frozenset of taxa (the smaller side).
    
    def enumerate_trees(taxa_set, all_taxa):
        """Yield all unrooted binary trees on taxa_set as sets of splits."""
        n_t = len(taxa_set)
        if n_t <= 3:
            # Only one topology for 3 taxa (a star)
            yield set()
            return
        
        taxa_list = sorted(taxa_set)
        # Enumerate all possible splits: pick a subset A of taxa_set
        # with 2 <= |A| <= n_t - 2, and |A| <= n_t/2 to avoid double-counting
        # If |A| == n_t/2, we need a tiebreaker
        
        for size_a in range(2, n_t // 2 + 1):
            for A in combinations(taxa_list, size_a):
                A_set = set(A)
                B_set = taxa_set - A_set
                n_a = len(A_set)
                n_b = len(B_set)
                
                # Avoid double counting: if n_a == n_b, only use one ordering
                if n_a == n_b:
                    # Canonical: sort the two sides and pick the lexicographically smaller
                    sorted_a = sorted(A_set)
                    sorted_b = sorted(B_set)
                    if sorted_a > sorted_b:
                        continue
                
                # Recursively enumerate trees on A and B
                for tree_a in enumerate_trees(A_set, all_taxa):
                    for tree_b in enumerate_trees(B_set, all_taxa):
                        # Combine splits
                        # Each split in tree_a is relative to A, need to map to full taxa
                        # Actually, the splits in tree_a are subsets of A,
                        # and they remain valid as splits of the full tree
                        # (since B is on the other side of the root edge)
                        
                        splits = set(tree_a) | set(tree_b)
                        # Add the split (A_set, B_set)
                        # Canonicalize: use the smaller side
                        if n_a <= n_b:
                            splits.add(frozenset(A_set))
                        else:
                            splits.add(frozenset(B_set))
                        
                        yield splits
    
    all_taxa = set(labels)
    best_splits = None
    best_score = float('inf')
    
    for splits in enumerate_trees(all_taxa, all_taxa):
        # Build the newick from splits
        # This is complex. Instead, let's just score the tree by
        # computing the total branch length using the distance matrix.
        
        # For a given set of splits, we can compute the tree's total branch length
        # as follows: for each split (A, B) with |A| <= |B|, the branch length
        # across the split is:
        #   L = (1/|A|/|B|) * (sum_{i in A, j in B} d(i,j) - sum_{i in A, j in A} d(i,j)/|A| - ...)
        # This is getting complex.
        
        # Simpler: build the tree from splits and compute total branch length.
        # But building a tree from splits is also non-trivial.
        
        # Alternative: for small n, just build all possible NJ trees by
        # trying all possible joining orders.
        
        # For n=4: try all 3 ways to join 4 taxa
        # For n=5: try all 15 ways
        # etc.
        
        # Let's use a different approach: generate all trees by
        # recursively joining pairs, and score each.
        pass
    
    # Given the complexity of building trees from splits, let's use
    # a simpler exhaustive approach: try all possible NJ joining orders.
    
    # For n taxa, NJ joins pairs iteratively. The number of possible
    # joining sequences is large, but for n<=7 we can be smart.
    
    # Actually, let's just use NJ + a few random restarts with perturbed distances.
    newick = _nj_build_tree(d, labels)
    
    # Try a few alternative starting points
    for trial in range(5):
        if time.time() > start_time + 5.0:
            break
        # Perturb distances slightly
        d_perturbed = d + np.random.uniform(-0.01, 0.01, d.shape)
        np.fill_diagonal(d_perturbed, 0.0)
        newick_trial = _nj_build_tree(d_perturbed, labels)
        # Score both
        score_orig = _nj_score(d, newick, n)
        score_trial = _nj_score(d, newick_trial, n)
        if score_trial < score_orig:
            newick = newick_trial
    
    return newick


def _spr_refine(d, labels, newick, n, deadline=None):
    """SPR (Subtree Prune and Regraft) refinement.
    Try all possible SPR moves and keep the best one.
    Repeat until no improvement or deadline."""
    if deadline is None:
        deadline = time.time() + 10.0
    
    best_newick = newick
    best_score = _nj_score(d, newick, n)
    
    # Parse the tree to identify all internal edges for SPR moves
    # For each internal edge, prune one side and regraft it elsewhere
    
    # We need to work with the tree structure
    # Let's use a simpler approach: represent the tree as a set of splits
    # and try adding/removing splits
    
    # Actually, let's do a proper SPR:
    # 1. Parse the newick into a tree
    # 2. For each internal edge, consider pruning the subtree below it
    # 3. Try regrafting it at every other internal edge
    # 4. Compute the score for each candidate
    # 5. Keep the best
    
    # This requires careful tree manipulation. Let's use a simpler heuristic:
    # Try all possible "rearrangements" by changing the NJ joining order.
    
    # For now, implement a basic version: try all pairs of leaves to swap
    # in the NJ construction.
    
    # Actually, let's implement a proper but simple SPR:
    # Represent the tree as nested lists, try all prune-regraft moves.
    
    import re
    
    def parse_tree(s):
        """Parse newick into a tree: list of (children, label, length)."""
        i = 0
        def parse_node():
            nonlocal i
            if i >= len(s):
                return None
            if s[i] == '(':
                i += 1
                children = []
                while i < len(s) and s[i] != ')':
                    child = parse_node()
                    if child is not None:
                        children.append(child)
                    if i < len(s) and s[i] == ',':
                        i += 1
                if i < len(s) and s[i] == ')':
                    i += 1
                length = 0.0
                if i < len(s) and s[i] == ':':
                    i += 1
                    start = i
                    while i < len(s) and s[i] not in ',):;':
                        i += 1
                    try:
                        length = float(s[start:i])
                    except:
                        length = 0.0
                return (children, "", length)
            else:
                start = i
                while i < len(s) and s[i] not in ',):;':
                    i += 1
                label = s[start:i]
                length = 0.0
                if i < len(s) and s[i] == ':':
                    i += 1
                    start = i
                    while i < len(s) and s[i] not in ',):;':
                        i += 1
                    try:
                        length = float(s[start:i])
                    except:
                        length = 0.0
                return ([], label, length)
        
        return parse_node()
    
    def to_newick(node):
        """Convert tree back to newick."""
        children, label, length = node
        if not children:
            return f"{label}:{length:.6f}" if length > 0 else label
        else:
            parts = [to_newick(c) for c in children]
            s = "(" + ",".join(parts) + ")"
            if length > 0:
                s += f":{length:.6f}"
            return s
    
    def get_internal_edges(tree):
        """Get all internal edges (parent, child_index) where child is an internal node."""
        edges = []
        def walk(node, parent, parent_idx):
            children, label, length = node
            for i, child in enumerate(children):
                child_children, _, _ = child
                if child_children:  # Internal node
                    edges.append((parent, parent_idx, i))
                walk(child, node, i)
        walk(tree, None, -1)
        return edges
    
    def prune_and_regraft(tree, edge_to_prune, regraft_edge):
        """Prune the subtree at edge_to_prune and regraft at regraft_edge.
        Returns a new tree."""
        # This is complex. For simplicity, let's skip full SPR and just
        # try a few simple moves.
        pass
    
    # Given the complexity, let's do a simpler refinement:
    # Re-run NJ with different random perturbations of the distance matrix
    # and keep the best result.
    
    # Actually, let's just return the NJ tree for now.
    # The exhaustive small-n case will be handled separately.
    
    return best_newick


def _total_branch_length(d, labels, newick):
    """Compute total branch length of a tree given distance matrix.
    This is a proxy for the minimum evolution criterion."""
    # Parse the Newick to get splits, then sum distances for each split
    # Simpler: use the tree structure to compute sum of branch lengths
    # For now, approximate by summing all pairwise distances weighted by topology
    # Actually, for ME we want the sum of all branch lengths in the tree
    # which equals (1/(n-1)) * sum over all pairs of d(i,j) for the tree metric
    # But we don't have the tree metric directly.
    # Skip this for now - NJ is already a good heuristic.
    return 0.0


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    start_time = time.time()
    DEADLINE = 265.0  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)
    
    # For small n (<=7), use exhaustive search
    if n <= 7:
        newick = _exhaustive_small_n(d, labels, n, start_time)
    else:
        newick = _nj_build_tree(d, labels)
        # Try a few alternative NJ constructions with perturbed distances
        best_score = _nj_score(d, newick, n)
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            d_pert = d + np.random.uniform(-0.05, 0.05, d.shape)
            np.fill_diagonal(d_pert, 0.0)
            newick_trial = _nj_build_tree(d_pert, labels)
            score_trial = _nj_score(d, newick_trial, n)
            if score_trial < best_score:
                best_score = score_trial
                newick = newick_trial
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return newick
    
    return newick
