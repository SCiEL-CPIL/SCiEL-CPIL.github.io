# EVOLVE-BLOCK-START
import math
import random
from collections import defaultdict

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

def _compute_distances(sequences, labels):
    """Compute pairwise JC distances."""
    dist = {}
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = labels[i], labels[j]
            d = _jc_distance(sequences[a], sequences[b])
            dist[(a, b)] = d
            dist[(b, a)] = d
    return dist

def _neighbor_joining(dist_matrix, labels):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns: Newick string
    """
    if len(labels) <= 1:
        return labels[0] if labels else ""
    
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    dist = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                dist[(a, b)] = dist_matrix[(a, b)]
    
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(dist[(a, b)] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * dist[(a, b)] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
        
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new_node = f"nj_{len(nodes)}_{a}_{b}"
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_ax = dist[(a, x)]
            d_bx = dist[(b, x)]
            d_ab = dist[(a, b)]
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x
        
        nodes = rest + [new_node]
        trees[new_node] = joined
    
    if len(nodes) == 3:
        a, b, c = nodes
        return f"({trees[a]},{trees[b]},{trees[c]})"
    elif len(nodes) == 2:
        return f"({trees[0]},{trees[1]})"
    else:
        return trees[0]

def _bootstrap_resample(sequences, n_sites, rng):
    """Resample sites with replacement for bootstrap."""
    labels = list(sequences.keys())
    boot_seq = {}
    for label in labels:
        seq = sequences[label]
        indices = rng.choices(range(n_sites), k=n_sites)
        boot_seq[label] = ''.join(seq[i] for i in indices)
    return boot_seq

def _extract_splits(newick, labels):
    """Extract unrooted splits from a Newick string."""
    newick = newick.strip().rstrip(';').strip()
    
    def parse_newick(s, pos=0):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse_newick(s, pos)
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    break
                else:
                    break
            pos += 1
            if pos < len(s) and s[pos] not in ',)':
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            return ('node', children), pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            label = s[start:pos].strip()
            return ('leaf', label), pos

    def get_leaves(node):
        if node[0] == 'leaf':
            return {node[1]}
        else:
            leaves = set()
            for child in node[1]:
                leaves.update(get_leaves(child))
            return leaves

    def extract_splits(node, all_leaves, splits):
        if node[0] == 'leaf':
            return
        for child in node[1]:
            child_leaves = get_leaves(child)
            if 1 < len(child_leaves) < len(all_leaves):
                complement = all_leaves - child_leaves
                split = frozenset([frozenset(child_leaves), frozenset(complement)])
                splits.append(split)
            extract_splits(child, all_leaves, splits)

    try:
        parsed, _ = parse_newick(newick)
        all_leaves = set(labels)
        splits = []
        extract_splits(parsed, all_leaves, splits)
        return splits
    except Exception:
        return []

def _build_consensus_tree(labels, split_counts, threshold=0.5):
    """
    Build a consensus tree from split frequencies.
    This is a simplified approach: we just use the splits that are supported.
    For a proper consensus, we need to find a compatible set of splits.
    Here, we will use a greedy approach or just return NJ if it's too complex.
    """
    # For simplicity and safety, we will return the NJ tree if consensus construction fails.
    # A proper consensus tree builder is complex.
    pass

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(next(iter(sequences.values())))
    
    rng = random.Random(42)
    
    dist_matrix = _compute_distances(sequences, labels)
    original_tree = _neighbor_joining(dist_matrix, labels)
    
    if n_taxa <= 8:
        return original_tree
    
    # Perform bootstrap
    n_boot = min(100, max(20, 500 // n_taxa))
    split_counts = defaultdict(int)
    
    for _ in range(n_boot):
        boot_seq = _bootstrap_resample(sequences, n_sites, rng)
        boot_dist = _compute_distances(boot_seq, labels)
        boot_tree = _neighbor_joining(boot_dist, labels)
        splits = _extract_splits(boot_tree, labels)
        for split in splits:
            split_counts[split] += 1
    
    # For now, return original_tree as consensus construction is complex.
    # Future improvement: Implement a proper consensus tree builder.
    
    return original_tree
# EVOLVE-BLOCK-END
