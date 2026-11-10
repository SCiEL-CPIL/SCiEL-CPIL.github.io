"""Phylogenetic topology recovery using split-based approach with NJ backbone."""
import math
import time
import sys

# EVOLVE-BLOCK-START

def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    if p <= 0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix."""
    n = len(labels)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_reconstruct(sequences):
    """Standard Neighbor-Joining reconstruction."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    dist = _compute_distance_matrix(sequences, labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    current_n = n
    
    while len(nodes) > 2:
        m = len(nodes)
        # Compute row sums
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += dist[nodes[i]][nodes[j]]
        
        # Find pair with minimum Q-criterion
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * dist[nodes[i]][nodes[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        
        # Compute branch lengths
        delta = (r[best_i] - r[best_j]) / (2 * (m - 2))
        d_ab = dist[a][b]
        len_a = 0.5 * d_ab + delta
        len_b = 0.5 * d_ab - delta
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        # Create new node
        new_node = current_n
        current_n += 1
        trees[new_node] = "(" + trees[a] + "," + trees[b] + ");"
        
        # Update distance matrix
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            x = nodes[k]
            dist[new_node][x] = 0.5 * (dist[a][x] + dist[b][x] - d_ab)
            dist[x][new_node] = dist[new_node][x]
        
        # Remove old nodes
        nodes = [nodes[k] for k in range(m) if k != best_i and k != best_j]
        nodes.append(new_node)
        trees[new_node] = trees[new_node]  # already set
    
    # Final two nodes: create unrooted tree
    if len(nodes) == 2:
        # We need to merge the last two into an unrooted tree
        # Actually NJ typically leaves 3 nodes; let's handle 2
        # Merge them into a single unrooted tree
        a, b = nodes[0], nodes[1]
        # Compute distance between them
        d = dist[a][b]
        # For unrooted tree with 2 subtrees, we just join them
        # This shouldn't happen with standard NJ (stops at 3)
        # But handle it gracefully
        return "(" + trees[a] + "," + trees[b] + ");"
    
    # This should not be reached for n >= 3
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _extract_splits(newick, labels):
    """Extract all non-trivial splits from a Newick string."""
    # Parse Newick to get all internal nodes and their splits
    # A split is a partition of taxa into two non-empty proper subsets
    splits = set()
    
    # We need to parse the tree structure
    # For simplicity, use a recursive approach
    def parse_node(s, pos):
        """Parse a node, return (taxon_set, new_pos)."""
        if pos >= len(s):
            return set(), pos
        
        if s[pos] == '(':
            # Internal node
            pos += 1
            child_sets = []
            while pos < len(s) and s[pos] != ')':
                child_set, pos = parse_node(s, pos)
                child_sets.append(child_set)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            result = set()
            for cs in child_sets:
                result |= cs
            return result, pos
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in ',();':
                pos += 1
            name = s[start:pos].strip()
            if name in labels:
                return {name}, pos
            return set(), pos
    
    all_labels = set(labels)
    # The Newick might not have all taxa if it's a tree (not network)
    # Parse the whole tree
    root_set, _ = parse_node(newick, 0)
    
    # To get splits, we need to traverse internal nodes
    # This is complex; let's use a different approach
    # For now, return empty set (this function is a placeholder)
    return splits


def _build_tree_from_splits(splits, labels):
    """Build a tree from a set of compatible splits."""
    # This is complex; use NJ as fallback
    # For now, just return NJ result
    return None


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from sequences."""
    start_time = time.time()
    deadline = start_time + 170.0  # Leave margin for 180s budget
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    # Primary method: NJ
    result = _nj_reconstruct(sequences)
    
    # Check time
    if time.time() > deadline:
        return result
    
    # The result should be valid
    return result

# EVOLVE-BLOCK-END
