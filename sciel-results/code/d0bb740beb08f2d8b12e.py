import math
import time
import numpy as np

_DEADLINE = 255.0  # Safety margin before 270s kill
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy for speed."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    
    conv = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_seqs = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lb in enumerate(labels):
        seq = sequences[lb]
        for j, c in enumerate(seq):
            num_seqs[i, j] = conv[c]
    
    L = len(sequences[labels[0]])
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(num_seqs[i] != num_seqs[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_build(D, labels):
    """Run Neighbor Joining, return Newick string."""
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    max_nodes = 2 * n
    dist = np.zeros((max_nodes, max_nodes))
    dist[:n, :n] = D
    
    active = list(range(n))
    tree_strs = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        total = np.zeros(m)
        for k, i in enumerate(active):
            total[k] = sum(dist[i, j] for j in active if j != i)
        
        best_criterion = float('inf')
        best_pair = None
        for ki in range(m):
            i = active[ki]
            for kj in range(ki + 1, m):
                j = active[kj]
                criterion = (m - 2) * dist[i, j] - total[ki] - total[kj]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        i, j = best_pair
        joined = f"({tree_strs[i]},{tree_strs[j]})"
        new_idx = max(active) + 1
        
        if new_idx >= max_nodes:
            break
            
        rest = [x for x in active if x not in (i, j)]
        
        for x in rest:
            d_new = 0.5 * (dist[i, x] + dist[j, x] - dist[i, j])
            dist[new_idx, x] = d_new
            dist[x, new_idx] = d_new
        
        active = rest + [new_idx]
        tree_strs[new_idx] = joined
    
    parts = [tree_strs[x] for x in active]
    return "(" + ",".join(parts) + ");"


def _extract_splits(newick, all_taxa):
    """Extract unrooted splits (frozensets) from a Newick string."""
    s = newick.rstrip(';').strip()
    splits = set()
    
    def _split_top_level(s):
        parts = []
        depth = 0
        current = []
        for c in s:
            if c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(c)
        if current:
            parts.append(''.join(current))
        return parts
    
    def _get_taxa(s):
        """Get all taxa in a subtree string."""
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            taxa = set()
            for part in _split_top_level(inner):
                taxa |= _get_taxa(part.strip())
            return taxa
        else:
            # Leaf - strip branch length
            if ':' in s:
                s = s.split(':')[0].strip()
            return {s} if s in all_taxa else set()
    
    def _collect_splits(s, parent_taxa=None):
        s = s.strip()
        if not s:
            return
        
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            parts = _split_top_level(inner)
            subtree_taxa = set()
            for part in parts:
                subtree_taxa |= _get_taxa(part.strip())
            
            if 0 < len(subtree_taxa) < len(all_taxa):
                complement = all_taxa - subtree_taxa
                # Normalize: use the smaller side
                if len(subtree_taxa) > len(complement):
                    subtree_taxa = complement
                splits.add(frozenset(subtree_taxa))
            
            # Recurse into each part
            for part in parts:
                _collect_splits(part.strip(), all_taxa)
        else:
            # Leaf, no split
            pass
    
    _collect_splits(s)
    return splits


def _consensus_tree(splits_counts, all_taxa, threshold=0.5):
    """Build a consensus tree from split frequencies using greedy clustering."""
    n = len(all_taxa)
    if n <= 3:
        if n == 3:
            return f"({','.join(sorted(all_taxa))});"
        elif n == 2:
            return f"({','.join(sorted(all_taxa))});"
        else:
            return f"({','.join(sorted(all_taxa))});"
    
    # Filter splits by threshold
    good_splits = set()
    for split, count in splits_counts.items():
        if count >= threshold:
            good_splits.add(split)
    
    if not good_splits:
        # Fallback: star tree
        return "(" + ",".join(sorted(all_taxa)) + ");"
    
    # Build consensus using a greedy approach
    # Start with all taxa as separate clusters, merge based on splits
    # Actually, let's use a simpler approach: build the tree from splits
    
    # For each split, we know which taxa go together
    # We can build a tree by finding a set of non-conflicting splits that form a valid tree
    
    # Simple approach: use the most frequent splits to build NJ-like tree
    # Or: return the tree from the most common NJ run
    
    # Better: build a tree from the split set using a recursive method
    # For each split, it partitions the taxa. We can build a tree where each split
    # corresponds to an internal node.
    
    # Let's use a different approach: find the best single tree that maximizes
    # agreement with the split frequencies. Since we have NJ trees, let's just
    # pick the one with the highest number of matching splits.
    
    return None  # Signal to use a different approach


def _nj_from_splits(splits, labels):
    """Build a Newick tree from a set of splits using a greedy method."""
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({','.join(labels)});"
        elif n == 2:
            return f"({','.join(labels)});"
        else:
            return f"({','.join(labels)});"
    
    # Use a simple approach: for each split, it defines a bipartition
    # Build a tree by recursively splitting
    
    all_taxa = set(labels)
    
    def _build_subtree(taxa_set):
        if len(taxa_set) == 1:
            return list(taxa_set)[0]
        if len(taxa_set) == 2:
            a, b = sorted(taxa_set)
            return f"({a},{b})"
        
        # Find a split that partitions this taxa_set
        best_split = None
        best_size = 0
        for split in splits:
            if not split.issubset(all_taxa):
                continue
            # Check if split is relevant to this subtree
            side_a = split & taxa_set
            side_b = taxa_set - split
            if 1 < len(side_a) < len(taxa_set) and 1 < len(side_b) < len(taxa_set):
                # Valid split for this subtree
                if len(side_a) < len(taxa_set) - len(side_a):
                    size = len(side_a)
                else:
                    size = len(taxa_set) - len(side_a)
                if size > best_size:
                    best_size = size
                    best_split = (side_a, side_b)
        
        if best_split is None:
            # No valid split found, make a star
            return "(" + ",".join(sorted(taxa_set)) + ")"
        
        side_a, side_b = best_split
        sub_a = _build_subtree(side_a)
        sub_b = _build_subtree(side_b)
        return f"({sub_a},{sub_b})"
    
    result = _build_subtree(all_taxa)
    return result + ";"


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 3:
            return f"({','.join(labels)});"
        elif n == 2:
            return f"({','.join(labels)});"
        else:
            return f"({','.join(labels)});"
    
    # Compute full distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run NJ on full data
    newick_full = _nj_build(D, labels)
    splits_full = _extract_splits(newick_full, set(labels))
    
    # Check if we have time for bootstrap
    if _time_left() < 10.0:
        return newick_full
    
    # Bootstrap: resample sites with replacement
    L = len(sequences[labels[0]])
    rng = np.random.RandomState(42)
    
    # Determine number of bootstrap replicates based on time and n
    # For small n, we can do more; for large n, fewer
    if n <= 10:
        max_boot = 200
    elif n <= 16:
        max_boot = 100
    elif n <= 24:
        max_boot = 50
    else:
        max_boot = 30
    
    boot_splits = {}  # frozenset -> count
    all_boot_trees = []
    
    # Pre-convert sequences to numeric for fast resampling
    conv = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_seqs = np.zeros((n, L), dtype=np.int8)
    for i, lb in enumerate(labels):
        seq = sequences[lb]
        for j, c in enumerate(seq):
            num_seqs[i, j] = conv[c]
    
    for b in range(max_boot):
        if _time_left() < 5.0:
            break
        
        # Resample sites
        boot_indices = rng.randint(0, L, size=L)
        boot_seqs = num_seqs[:, boot_indices]
        
        # Compute bootstrap distance matrix
        D_boot = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mismatches = np.sum(boot_seqs[i] != boot_seqs[j])
                p = mismatches / L
                if p >= 0.7499:
                    d = 3.0
                else:
                    d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                D_boot[i, j] = d
                D_boot[j, i] = d
        
        # Run NJ
        newick_boot = _nj_build(D_boot, labels)
        all_boot_trees.append(newick_boot)
        splits_boot = _extract_splits(newick_boot, set(labels))
        
        for split in splits_boot:
            boot_splits[split] = boot_splits.get(split, 0) + 1
    
    # If we didn't do any bootstrap, return the full NJ tree
    if not all_boot_trees:
        return newick_full
    
    # Find the best tree: the one that maximizes the number of splits
    # that appear in at least 50% of bootstrap trees
    threshold = 0.5 * len(all_boot_trees)
    good_splits = set()
    for split, count in boot_splits.items():
        if count >= threshold:
            good_splits.add(split)
    
    # Score each bootstrap tree by how many of its splits are in good_splits
    best_tree = newick_full
    best_score = -1
    
    for tree in all_boot_trees:
        tree_splits = _extract_splits(tree, set(labels))
        score = sum(1 for s in tree_splits if s in good_splits)
        if score > best_score:
            best_score = score
            best_tree = tree
    
    # Also consider the full NJ tree
    full_splits = _extract_splits(newick_full, set(labels))
    full_score = sum(1 for s in full_splits if s in good_splits)
    if full_score > best_score:
        best_tree = newick_full
    
    return best_tree
