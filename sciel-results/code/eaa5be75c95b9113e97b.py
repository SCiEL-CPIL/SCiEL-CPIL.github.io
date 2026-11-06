"""Phylogenetic topology recovery: NJ with distance-based quartet polishing."""
import math
import time
import itertools

# EVOLVE-BLOCK-START

def _jc_correct(p):
    """Jukes-Cantor correction for pairwise distance."""
    if p < 0 or p > 1:
        p = max(0.0, min(1.0, p))
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    # Build binary representation for fast comparison
    # Encode A,C,G,T as bits
    base_to_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # For each position, count mismatches between all pairs
    # Use a more efficient approach: for each pair, count differences
    # For small n, direct comparison is fine
    
    dist = [[0.0] * n for _ in range(n)]
    
    # Pre-encode sequences
    encoded = []
    for s in seqs:
        enc = [base_to_bits[c] for c in s]
        encoded.append(enc)
    
    for i in range(n):
        for j in range(i + 1, n):
            diffs = 0
            ei = encoded[i]
            ej = encoded[j]
            for k in range(length):
                if ei[k] != ej[k]:
                    diffs += 1
            p = diffs / length
            d = _jc_correct(p)
            dist[i][j] = d
            dist[j][i] = d
    
    return dist


def _nj_build(dist, labels):
    """Build tree using Neighbor Joining. Returns Newick string and node mapping."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    
    # dist is n x n
    # trees[i] is the Newick subtree for node i
    trees = list(labels)
    active = list(range(n))
    current_dist = [row[:] for row in dist]
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += current_dist[active[i]][active[j]]
        
        # Find best pair using Q-matrix
        best_q = float('inf')
        best_i = 0
        best_j = 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * current_dist[active[i]][active[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        ai = active[best_i]
        aj = active[best_j]
        
        # Create new node
        new_tree = f"({trees[ai]},{trees[aj]})"
        
        # Compute distances from new node to all other active nodes
        new_active = []
        for idx in range(m):
            if idx != best_i and idx != best_j:
                new_active.append(active[idx])
        
        # Add new node to the working set
        new_idx = len(active)  # position in current_dist for new node
        # Extend current_dist
        current_dist.append([0.0] * (len(current_dist) + 1))
        for row in current_dist:
            row.append(0.0)
        
        for k in new_active:
            # d(new, k) = 0.5 * (d(ai,k) + d(aj,k) - d(ai,aj))
            d_new_k = 0.5 * (current_dist[ai][k] + current_dist[aj][k] - current_dist[ai][aj])
            current_dist[new_idx][k] = d_new_k
            current_dist[k][new_idx] = d_new_k
        
        # Update trees and active
        trees.append(new_tree)
        active = new_active + [new_idx]
        next_id += 1
    
    # Combine remaining 3 nodes
    if len(active) == 3:
        result = f"({trees[active[0]]},{trees[active[1]]},{trees[active[2]]})"
    else:
        result = trees[active[0]]
    
    return result


def _extract_splits(newick):
    """Extract all non-trivial splits from a Newick tree string."""
    # Parse the Newick tree
    # We need to handle both rooted and unrooted representations
    # For unrooted trees with n taxa, we have n-3 non-trivial splits
    
    # Simple parser: find all internal edges
    # A split is defined by the set of leaves on one side of an internal edge
    
    def parse_newick(s):
        """Parse Newick into a tree structure. Returns list of (children, label) where children is list of nodes."""
        # Simple recursive parser
        def parse(pos):
            if pos >= len(s) or s[pos] == ')':
                return None, pos
            if s[pos] == '(':
                children = []
                pos += 1
                while pos < len(s) and s[pos] != ')':
                    child, pos = parse(pos)
                    if child is not None:
                        children.append(child)
                    # Skip comma
                    if pos < len(s) and s[pos] == ',':
                        pos += 1
                pos += 1  # skip ')'
                # Skip branch length
                while pos < len(s) and s[pos] not in (',', ')', ';', ' '):
                    pos += 1
                return children, pos
            else:
                # Leaf
                label = ""
                while pos < len(s) and s[pos] not in (',', ')', ';', ' ', '(', ':'):
                    label += s[pos]
                    pos += 1
                # Skip branch length
                if pos < len(s) and s[pos] == ':':
                    pos += 1
                    while pos < len(s) and s[pos] not in (',', ')', ';', ' '):
                        pos += 1
                return label, pos
        
        node, pos = parse(0)
        return node
    
    def get_leaves(node):
        """Get all leaf labels in a subtree."""
        if isinstance(node, str):
            return {node}
        leaves = set()
        for child in node:
            leaves |= get_leaves(child)
        return leaves
    
    def collect_splits(node, all_leaves, splits):
        """Collect all non-trivial splits."""
        if isinstance(node, str):
            return
        leaves_here = get_leaves(node)
        # For each child, the split is child_leaves vs rest
        for child in node:
            child_leaves = get_leaves(child)
            if 0 < len(child_leaves) < len(all_leaves):
                # Canonical form: always put the smaller set first (or lexicographically)
                a = frozenset(child_leaves)
                b = frozenset(all_leaves - child_leaves)
                if len(a) > len(b):
                    a, b = b, a
                splits.add((a, b))
            collect_splits(child, all_leaves, splits)
    
    tree = parse_newick(newick)
    all_leaves = get_leaves(tree)
    splits = set()
    collect_splits(tree, all_leaves, splits)
    return splits


def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance."""
    common = splits1 & splits2
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    return 2.0 * (len(splits1) + len(splits2) - 2 * len(common)) / total


def _quartet_test(dist, i, j, k, l):
    """
    Determine the best quartet topology for leaves i,j,k,l given distances.
    Returns (a,b,c,d) representing the split (a,b)|(c,d).
    The three possible unrooted quartet topologies are:
      (i,j)|(k,l), (i,k)|(j,l), (i,l)|(j,k)
    """
    dij = dist[i][j]
    dik = dist[i][k]
    dil = dist[i][l]
    djk = dist[j][k]
    djl = dist[j][l]
    dkl = dist[k][l]
    
    # Sum of opposite edges for each topology
    # (i,j)|(k,l): dij + dkl
    # (i,k)|(j,l): dik + djl
    # (i,l)|(j,k): dil + djk
    s1 = dij + dkl
    s2 = dik + djl
    s3 = dil + djk
    
    if s1 <= s2 and s1 <= s3:
        return (i, j, k, l)  # split (i,j)|(k,l)
    elif s2 <= s1 and s2 <= s3:
        return (i, k, j, l)  # split (i,k)|(j,l)
    else:
        return (i, l, j, k)  # split (i,l)|(j,k)


def _quartet_polish(newick, dist, labels, max_iter=50):
    """
    Polish the tree using quartet-based local search.
    Try to improve the tree by changing quartet topologies.
    """
    current_splits = _extract_splits(newick)
    
    if not current_splits:
        return newick
    
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    best_newick = newick
    best_splits = current_splits
    best_score = 1.0  # worst possible RF
    
    # We'll do a simple local search: try changing one quartet at a time
    # For efficiency, only do a few iterations
    
    for iteration in range(max_iter):
        improved = False
        
        # Try all quartets
        for i, j, k, l in itertools.combinations(range(n), 4):
            # Determine current topology from splits
            # Check which quartet split is present
            lab_i = labels[i]
            lab_j = labels[j]
            lab_k = labels[k]
            lab_l = labels[l]
            
            # The three possible splits for this quartet
            split1 = (frozenset({lab_i, lab_j}), frozenset({lab_k, lab_l}))
            split2 = (frozenset({lab_i, lab_k}), frozenset({lab_j, lab_l}))
            split3 = (frozenset({lab_i, lab_l}), frozenset({lab_j, lab_k}))
            
            # Check which one is in the current tree
            present = None
            for split in [split1, split2, split3]:
                # Canonical form
                a, b = split
                if len(a) > len(b):
                    a, b = b, a
                if (a, b) in current_splits:
                    present = split
                    break
            
            if present is None:
                continue
            
            # Determine the best topology from distances
            best_quad = _quartet_test(dist, i, j, k, l)
            
            # Map back to split
            bi, bj, bk, bl = best_quad
            lab_bi = labels[bi]
            lab_bj = labels[bj]
            lab_bk = labels[bk]
            lab_bl = labels[bl]
            best_split = (frozenset({lab_bi, lab_bj}), frozenset({lab_bk, lab_bl}))
            
            if best_split == present:
                continue
            
            # Try this change: remove present, add best_split
            # This is a simplification - in reality, changing a quartet split
            # requires modifying the tree structure
            # For now, we'll just track the best set of splits
            
            new_splits = set(current_splits)
            # Remove the canonical form of present
            a, b = present
            if len(a) > len(b):
                a, b = b, a
            new_splits.discard((a, b))
            
            # Add the canonical form of best_split
            a2, b2 = best_split
            if len(a2) > len(b2):
                a2, b2 = b2, a2
            new_splits.add((a2, b2))
            
            # This is a rough heuristic - we can't easily compute the RF
            # without knowing if the new set of splits is realizable
            # For simplicity, skip this approach and use a different strategy
            
        break  # Skip the quartet polish for now due to complexity
    
    return best_newick


def _build_from_splits(splits, labels):
    """
    Build a tree from a set of splits.
    This is a simplified approach: use the splits to guide NJ.
    """
    # For now, just return a balanced tree as fallback
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Build a balanced binary tree
    def balanced_tree(labs):
        if len(labs) == 1:
            return labs[0]
        if len(labs) == 2:
            return f"({labs[0]},{labs[1]})"
        mid = len(labs) // 2
        left = balanced_tree(labs[:mid])
        right = balanced_tree(labs[mid:])
        return f"({left},{right})"
    
    return balanced_tree(labels)


def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree from sequences.
    Uses Neighbor Joining with optional quartet polishing.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Build tree with NJ
    newick = _nj_build(dist, labels)
    
    # Check time
    if time.time() > deadline - 30.0:
        return newick
    
    # Try quartet polishing (simplified)
    # For now, return the NJ tree
    # A more sophisticated version would do local search on quartets
    
    return newick

# EVOLVE-BLOCK-END
