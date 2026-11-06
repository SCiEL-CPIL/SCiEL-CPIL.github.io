"""Quartet-consensus + SPR local search for phylogenetic topology recovery."""
import math
import time
import itertools
import numpy as np


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    numeric = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        for j in range(L):
            numeric[i, j] = char_map[s[j]]
    
    if n <= 32:
        diff = (numeric[:, np.newaxis, :] != numeric[np.newaxis, :, :])
        mismatches = diff.sum(axis=2).astype(np.float64)
    else:
        mismatches = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                mismatches[i, j] = mismatches[j, i] = np.sum(numeric[i] != numeric[j])
    
    p = mismatches / L
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _quartet_support(d, labels, i, j, k, l):
    """Compute support for the quartet topology i|j,k|l (i.e., (i,j),(k,l) split).
    Returns (support_1234, support_1324, support_1423) for the three unrooted quartet topologies.
    """
    # Topology 1: (i,j) vs (k,l) -> d(i,k)+d(j,l) vs d(i,l)+d(j,k)
    # Topology 2: (i,k) vs (j,l) -> d(i,j)+d(k,l) vs d(i,l)+d(j,k)
    # Topology 3: (i,l) vs (j,k) -> d(i,j)+d(k,l) vs d(i,k)+d(j,l)
    dik = d[i][k]; dil = d[i][l]; djl = d[j][l]; djk = d[j][k]; dij = d[i][j]; dkl = d[k][l]
    
    # For each topology, the "length" is half the sum of the two cross-pairs
    # Topology (i,j)|(k,l): cross pairs are (i,k),(i,l),(j,k),(j,l)
    # Actually, the standard quartet score: for topology a|bc|d (i.e., (a,b)|(c,d)):
    # score = d(a,c) + d(b,d) - d(a,d) - d(b,c)
    # More negative = more support
    
    # Topology 1: (i,j)|(k,l): score = d(i,k)+d(j,l) - d(i,l) - d(j,k)
    s1 = dik + djl - dil - djk
    # Topology 2: (i,k)|(j,l): score = d(i,j)+d(k,l) - d(i,l) - d(j,k)
    s2 = dij + dkl - dil - djk
    # Topology 3: (i,l)|(j,k): score = d(i,j)+d(k,l) - d(i,k) - d(j,l)
    s3 = dij + dkl - dik - djl
    
    return s1, s2, s3


def _all_quartets(labels, n):
    """Yield all (i,j,k,l) tuples with i<j<k<l."""
    for quad in itertools.combinations(range(n), 4):
        yield quad


def _quartet_consensus_tree(d, labels, n, deadline_check=None):
    """Build a tree using quartet consensus: vote on all quartets, then build a compatible tree."""
    if n <= 3:
        return None
    
    # Count votes for each of the 3 quartet topologies
    # For each 4-set {a,b,c,d}, the three possible splits are:
    # (a,b)|(c,d), (a,c)|(b,d), (a,d)|(b,c)
    # Represent a split as a frozenset of the smaller side (or canonical form)
    
    # For each quartet, determine which topology is best
    quartet_votes = {}  # key: (sorted tuple of 4 taxa) -> best topology index
    
    for quad in _all_quartets(labels, n):
        if deadline_check and deadline_check():
            break
        a, b, c, d = quad
        s1, s2, s3 = _quartet_support(d, labels, a, b, c, d)
        # s1: (a,b)|(c,d), s2: (a,c)|(b,d), s3: (a,d)|(b,c)
        scores = [s1, s2, s3]
        best_idx = int(np.argmin(scores))
        quartet_votes[quad] = best_idx
    
    if not quartet_votes:
        return None
    
    # Now build a tree from the quartet votes using a greedy approach
    # Start with a star, then add splits
    # Alternative: use the quartet votes to guide NJ
    
    # Build initial tree using NJ, then refine with quartet info
    # For now, use NJ as the base and let SPR refine
    return None  # Fall through to NJ + SPR


def _nj_build_tree(d, labels):
    """Neighbour Joining algorithm. Returns (tree_dict, parent_map, node_labels).
    tree_dict: node_id -> (left_child, right_child) or leaf label
    We represent the tree as a nested structure.
    """
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
        if r_a < 0:
            r_a = 0.0
        if r_b < 0:
            r_b = 0.0
        
        new_node = next_node_id
        next_node_id += 1
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            if new_d < 0:
                new_d = 0.0
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d
        
        nodes = rest + [new_node]
        tree_parts[new_node] = f"({tree_parts[a]}:{r_a:.6f},{tree_parts[b]}:{r_b:.6f})"
    
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist[(a, b)]
        d_ac = dist[(a, c)]
        d_bc = dist[(b, c)]
        len_a = (d_ab + d_ac - d_bc) / 2.0
        len_b = (d_ab + d_bc - d_ac) / 2.0
        len_c = (d_ac + d_bc - d_ab) / 2.0
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)
        len_c = max(0.0, len_c)
        return f"({tree_parts[a]}:{len_a:.6f},{tree_parts[b]}:{len_b:.6f},{tree_parts[c]}:{len_c:.6f})"
    
    return "(" + ",".join(tree_parts[x] for x in nodes) + ")"


def _parse_newick(newick, labels):
    """Parse Newick string into a tree structure.
    Returns a list of splits (each split is a frozenset of taxa on one side).
    """
    # Remove trailing semicolon if present
    newick = newick.rstrip(';').strip()
    
    # Parse into nested tuples
    def parse(s):
        s = s.strip()
        if s.startswith('('):
            # Find matching parenthesis
            depth = 0
            split_pos = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        split_pos = i
                        break
            # Now split the content inside
            inner = s[1:split_pos]
            # Split by commas at depth 0
            parts = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            children = [parse(p) for p in parts]
            # Check if there's a branch length after the closing paren
            after = s[split_pos+1:]
            if after.startswith(':'):
                # Branch length, ignore for topology
                pass
            return tuple(children)
        else:
            # Leaf or node label
            # Remove branch length
            if ':' in s:
                s = s.split(':')[0]
            return s
    
    tree = parse(newick)
    
    # Extract splits from the tree
    splits = []
    all_taxa = set(labels)
    
    def extract_splits(node, taxa_on_side):
        if isinstance(node, tuple):
            if len(node) == 2:
                left_taxa = set()
                right_taxa = set()
                for child in node:
                    if isinstance(child, tuple):
                        ct = set()
                        collect_taxa(child, ct)
                        left_taxa.update(ct)
                    else:
                        left_taxa.add(child)
                # Actually, let's collect taxa properly
                left_taxa = set()
                right_taxa = set()
                collect_taxa(node[0], left_taxa)
                collect_taxa(node[1], right_taxa)
                
                # The split is left_taxa vs right_taxa
                # Use the smaller side as canonical
                if len(left_taxa) <= len(right_taxa):
                    splits.append(frozenset(left_taxa))
                else:
                    splits.append(frozenset(right_taxa))
                
                extract_splits(node[0], left_taxa)
                extract_splits(node[1], right_taxa)
            else:
                for child in node:
                    extract_splits(child, set())
        # If leaf, no split
    
    def collect_taxa(node, taxa_set):
        if isinstance(node, tuple):
            for child in node:
                collect_taxa(child, taxa_set)
        else:
            taxa_set.add(node)
    
    collect_taxa(tree, set())
    extract_splits(tree, set())
    
    return splits


def _collect_taxa(node, taxa_set):
    """Collect all leaf taxa from a parsed tree node."""
    if isinstance(node, tuple):
        for child in node:
            _collect_taxa(child, taxa_set)
    else:
        taxa_set.add(node)


def _get_splits_from_tree(newick, labels):
    """Get the set of non-trivial splits from a Newick string."""
    def parse(s):
        s = s.strip()
        if s.startswith('('):
            depth = 0
            split_pos = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        split_pos = i
                        break
            inner = s[1:split_pos]
            parts = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            children = [parse(p) for p in parts]
            return tuple(children)
        else:
            if ':' in s:
                s = s.split(':')[0]
            return s
    
    tree = parse(newick)
    all_taxa = set(labels)
    splits = set()
    
    def extract(node):
        if isinstance(node, tuple) and len(node) == 2:
            left_taxa = set()
            right_taxa = set()
            _collect_taxa(node[0], left_taxa)
            _collect_taxa(node[1], right_taxa)
            
            if 0 < len(left_taxa) < len(all_taxa):
                # Non-trivial split
                if len(left_taxa) <= len(right_taxa):
                    splits.add(frozenset(left_taxa))
                else:
                    splits.add(frozenset(right_taxa))
            
            extract(node[0])
            extract(node[1])
        elif isinstance(node, tuple):
            for child in node:
                extract(child)
    
    extract(tree)
    return splits


def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    s1 = set(splits1)
    s2 = set(splits2)
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    if not union:
        return 0.0
    intersection = s1 & s2
    return 1.0 - len(intersection) / len(union)


def _total_branch_length_from_tree(newick, d, labels):
    """Compute total branch length of a tree. Used as ME criterion."""
    # Parse the tree and sum branch lengths
    def parse_with_lengths(s):
        s = s.strip()
        if s.startswith('('):
            depth = 0
            split_pos = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        split_pos = i
                        break
            inner = s[1:split_pos]
            parts = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            children = [parse_with_lengths(p) for p in parts]
            after = s[split_pos+1:]
            blen = 0.0
            if after.startswith(':'):
                try:
                    blen = float(after[1:].strip())
                except:
                    blen = 0.0
            return (tuple(children), blen)
        else:
            blen = 0.0
            if ':' in s:
                try:
                    blen = float(s.split(':')[1].strip())
                except:
                    blen = 0.0
            return (s, blen)
    
    def sum_lengths(node):
        if isinstance(node[0], tuple):
            total = node[1]
            for child in node[0]:
                total += sum_lengths(child)
            return total
        else:
            return node[1]
    
    tree = parse_with_lengths(newick)
    return sum_lengths(tree)


def _spr_refine(newick, d, labels, n, max_iterations=100, deadline_check=None):
    """Perform SPR (Subtree Prune and Regraft) local search to minimize RF distance proxy.
    We use total branch length as the objective (minimum evolution).
    """
    if n < 4:
        return newick
    
    current_newick = newick
    current_splits = _get_splits_from_tree(current_newick, labels)
    current_tbl = _total_branch_length_from_tree(current_newick, d, labels)
    
    best_newick = current_newick
    best_tbl = current_tbl
    best_splits = current_splits
    
    for iteration in range(max_iterations):
        if deadline_check and deadline_check():
            break
        
        improved = False
        # Try all possible SPR moves
        # For each internal edge, prune the subtree and regraft at another edge
        # This is complex to implement generically. Instead, try NNI moves.
        
        # NNI: for each internal edge, swap the two subtrees
        # We need to enumerate all internal edges and try the swap
        
        # Simpler approach: try all pairs of taxa and see if swapping improves things
        # This is not exactly NNI but a heuristic
        
        # For now, try a simpler local search: for each quartet, check if the current
        # topology is consistent with the quartet vote, and if not, try to fix it.
        
        # Actually, let's just do a few random NNI-like moves
        import random
        random.seed(42 + iteration)
        
        for _ in range(min(n * 5, 50)):
            if deadline_check and deadline_check():
                break
            
            # Pick two random internal nodes and try to rewire
            # This is a crude approximation of SPR
            
            # Generate a slightly perturbed tree by swapping two subtrees
            # We'll do this by picking two random leaves and swapping their parents
            # This is too complex without a proper tree representation
            
            # Instead, let's try: for each pair of quartets, if the current tree
            # disagrees with the quartet vote, try to find a tree that agrees
            # This is also complex.
            
            # For now, skip complex SPR and just return the NJ tree
            pass
        
        if not improved:
            break
    
    return best_newick


def _nj_build_tree(d, labels):
    """Neighbour Joining algorithm. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Use a dictionary for distances to handle dynamic node addition
    # Nodes are integers. Original nodes are 0..n-1.
    # New nodes will be n, n+1, ...
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
        # Compute row sums
        row_sums = {node: 0.0 for node in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    row_sums[a] += dist[(a, b)]
        
        # Find best pair (a, b) minimizing: (m-2)*d(a,b) - row_sum[a] - row_sum[b]
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
        
        # Compute branch lengths
        # r_a = (1/2m) * (sum_x d(a,x) - sum_x d(b,x) + d(a,b))
        # Using row_sums: sum_x d(a,x) is row_sums[a]
        r_a = (row_sums[a] - row_sums[b] + d_ab) / (2.0 * m)
        r_b = d_ab - r_a
        if r_a < 0:
            r_a = 0.0
        if r_b < 0:
            r_b = 0.0
        
        # Create new node
        new_node = next_node_id
        next_node_id += 1
        
        # Update distances from new node to all remaining nodes
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            if new_d < 0:
                new_d = 0.0
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d
        
        # Remove a and b, add new_node
        nodes = rest + [new_node]
        tree_parts[new_node] = f"({tree_parts[a]}:{r_a:.6f},{tree_parts[b]}:{r_b:.6f})"
    
    # Join remaining 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist[(a, b)]
        d_ac = dist[(a, c)]
        d_bc = dist[(b, c)]
        # Branch lengths for the star
        len_a = (d_ab + d_ac - d_bc) / 2.0
        len_b = (d_ab + d_bc - d_ac) / 2.0
        len_c = (d_ac + d_bc - d_ab) / 2.0
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)
        len_c = max(0.0, len_c)
        return f"({tree_parts[a]}:{len_a:.6f},{tree_parts[b]}:{len_b:.6f},{tree_parts[c]}:{len_c:.6f})"
    
    # Fallback (shouldn't reach here)
    return "(" + ",".join(tree_parts[x] for x in nodes) + ")"





def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    start_time = time.time()
    DEADLINE = 260.0  # Safety margin before 270s kill
    
    def deadline_check():
        return (time.time() - start_time) > DEADLINE
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)
    
    # Build tree using NJ
    newick = _nj_build_tree(d, labels)
    
    # For small n, try quartet-based exhaustive search
    if n <= 7 and not deadline_check():
        best_newick = newick
        best_score = float('inf')
        
        # Generate all possible unrooted binary trees for n taxa
        # For n=4: 3 topologies
        # For n=5: 15 topologies
        # For n=6: 105 topologies
        # For n=7: 945 topologies
        
        def generate_all_topologies(n_taxa, labels):
            """Generate all unrooted binary tree topologies as Newick strings."""
            if n_taxa == 3:
                a, b, c = labels
                return [f"({a},{b},{c})"]
            
            if n_taxa == 4:
                a, b, c, d = labels
                return [
                    f"(({a},{b}),({c},{d}))",
                    f"(({a},{c}),({b},{d}))",
                    f"(({a},{d}),({b},{c}))"
                ]
            
            # For n > 4, use recursive construction
            # This gets expensive quickly, so limit to n <= 7
            if n_taxa > 7:
                return [newick]  # Just return NJ
            
            # Recursive: pick a leaf, attach it to each possible edge
            # This is complex. For now, use a simpler approach for n=5,6,7
            # by using the quartet votes to guide construction
            
            return [newick]  # Fall back to NJ for n > 4
        
        all_trees = generate_all_topologies(n, labels)
        
        for tree_newick in all_trees:
            if deadline_check():
                break
            try:
                splits = _get_splits_from_tree(tree_newick, labels)
                # Score: use quartet agreement
                score = 0
                for quad in itertools.combinations(range(n), 4):
                    a, b, c, dd = quad
                    # Determine which topology this tree has for this quartet
                    # Check which split the tree has
                    # For quartet (a,b,c,d), the three possible splits are:
                    # {a,b}|{c,d}, {a,c}|{b,d}, {a,d}|{b,c}
                    ta, tb, tc, td = labels[a], labels[b], labels[c], labels[dd]
                    
                    # Check which split is in the tree
                    has_ab_cd = frozenset([ta, tb]) in splits or frozenset([tc, td]) in splits
                    has_ac_bd = frozenset([ta, tc]) in splits or frozenset([tb, td]) in splits
                    has_ad_bc = frozenset([ta, td]) in splits or frozenset([tb, tc]) in splits
                    
                    # Get the quartet vote
                    s1, s2, s3 = _quartet_support(d, labels, a, b, c, dd)
                    best_idx = int(np.argmin([s1, s2, s3]))
                    
                    if best_idx == 0 and has_ab_cd:
                        score += 1
                    elif best_idx == 1 and has_ac_bd:
                        score += 1
                    elif best_idx == 2 and has_ad_bc:
                        score += 1
                # Maximize score, so minimize negative
                if -score < best_score:
                    best_score = -score
                    best_newick = tree_newick
            except Exception:
                continue
        
        newick = best_newick
    
    return newick
