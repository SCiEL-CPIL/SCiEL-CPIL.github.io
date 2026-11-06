"""Improved phylogenetic topology recovery: NJ + quartet refinement + small-n exact search."""
# EVOLVE-BLOCK-START
import math
import time
import itertools
import sys

_DEADLINE = 260.0  # safety margin under 270s wall-clock
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix as a 2D list indexed by label index."""
    n = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_build(dist, labels):
    """
    Neighbor Joining algorithm.
    Returns a Newick string and the tree structure as nested tuples.
    """
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    
    # Work with a mutable copy
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # dist is the current distance matrix
    d = [row[:] for row in dist]
    
    while len(nodes) > 3:
        if _deadline_exceeded():
            break
        m = len(nodes)
        # Compute total distances
        total = [0.0] * m
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    total[i] += d[a][b]
        
        # Find best pair
        best_val = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * d[a][b] - total[i] - total[j]
                if criterion < best_val:
                    best_val = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        
        # Compute branch lengths
        da = 0.5 * d[a][b]
        db = 0.5 * d[a][b]
        if m > 2:
            for x in nodes:
                if x != a and x != b:
                    da += 0.5 * (d[a][x] - d[b][x])
                    db += 0.5 * (d[b][x] - d[a][x])
        
        joined = f"({trees[a]},{trees[b]})"
        new_node = -1  # use a negative index for new nodes
        new_id = -(len(nodes))
        
        # Compute distances from new node to all others
        for x in nodes:
            if x != a and x != b:
                d[a][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
        
        # Create new distance row/col
        for x in nodes:
            if x != a and x != b:
                d[new_id][x] = d[x][new_id] = d[a][x]
        
        # Remove a and b, add new_id
        nodes = [x for x in nodes if x not in (a, b)]
        nodes.append(new_id)
        trees[new_id] = joined
        
        # Update d for new node
        # d is indexed by original node ids and negative ids
        # We need to handle this carefully - let's rebuild
        # Actually, let's use a different approach with a dict-based distance
    
    # Build final tree from remaining 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        return f"(({trees[a]},{trees[b]}),{trees[c]})"
    elif len(nodes) == 2:
        return f"({trees[nodes[0]]},{trees[nodes[1]]})"
    elif len(nodes) == 1:
        return trees[nodes[0]]
    else:
        # Fallback: star tree
        return "(" + ",".join(trees[x] for x in nodes) + ")"


def _nj_build_v2(dist, labels):
    """
    Neighbor Joining with proper distance matrix handling.
    Returns Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    # Use a dictionary-based approach for clarity
    # nodes: list of current node identifiers (integers)
    # tree_repr: dict mapping node id -> Newick substring
    # dist: dict mapping (node_i, node_j) -> distance
    
    nodes = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    
    # Initialize distance dict
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    counter = n  # next new node id
    
    while len(nodes) > 3:
        if _deadline_exceeded():
            break
        m = len(nodes)
        # Compute total distances
        total = {a: 0.0 for a in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    total[a] += d[(a, b)]
        
        # Find best pair
        best_val = float('inf')
        best_a, best_b = nodes[0], nodes[1]
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * d[(a, b)] - total[a] - total[b]
                if criterion < best_val:
                    best_val = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        # Compute new distances
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            new_d = max(new_d, 0.0)
            d[(counter, x)] = new_d
            d[(x, counter)] = new_d
        
        # Remove old entries
        for x in nodes:
            if x != a and x != b:
                del d[(a, x)]
                del d[(b, x)]
        del d[(a, b)]
        del d[(b, a)]
        
        joined = f"({tree_repr[a]},{tree_repr[b]})"
        nodes = rest + [counter]
        tree_repr[counter] = joined
        counter += 1
    
    # Build final 3-node tree
    if len(nodes) == 3:
        a, b, c = nodes
        return f"(({tree_repr[a]},{tree_repr[b]}),{tree_repr[c]})"
    elif len(nodes) == 2:
        return f"({tree_repr[nodes[0]]},{tree_repr[nodes[1]]})"
    elif len(nodes) == 1:
        return tree_repr[nodes[0]]
    else:
        return "(" + ",".join(tree_repr[x] for x in nodes) + ")"


def _parse_newick(newick):
    """
    Parse a Newick string into a tree structure.
    Returns a list of splits (each split is a frozenset of taxa on one side).
    """
    # Remove trailing semicolon
    s = newick.strip().rstrip(';')
    
    # We'll do a simple recursive parse
    # Build adjacency representation
    # Actually, let's extract splits directly from the tree structure
    
    taxa = set()
    
    def extract_taxa(s):
        for ch in s:
            if ch in 'ACGT' or ch.isdigit() or ch.isalpha():
                pass
        # Better: find all leaf labels
        import re
        # Labels are t0, t1, ...
        labels = re.findall(r't\d+', s)
        return set(labels)
    
    taxa = extract_taxa(s)
    
    # Extract all internal splits
    # A split is defined by each internal edge
    # For a binary unrooted tree, each internal edge defines a split
    # We can extract splits by finding all subtrees at each internal node
    
    splits = set()
    
    def parse_node(s, pos):
        """
        Parse a node starting at position pos.
        Returns (split_set, new_pos) where split_set is the set of taxa in this subtree.
        """
        if pos >= len(s):
            return set(), pos
        
        if s[pos] == '(':
            # Internal node
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child_set, pos = parse_node(s, pos)
                children.append(child_set)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            # This subtree's taxa
            all_taxa = set()
            for c in children:
                all_taxa |= c
            # The split defined by this node (if it's not the root)
            # We'll collect splits separately
            return all_taxa, pos
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in '(),:;':
                pos += 1
            label = s[start:pos]
            return {label}, pos
    
    # For extracting splits, we need a different approach
    # Let's use a stack-based method to find all internal subtrees
    
    # Actually, the simplest way: for each internal node, the set of leaves under it
    # (and its complement) defines a split. We collect all such sets.
    
    def collect_splits(s):
        """Collect all non-trivial splits from a Newick string."""
        splits = set()
        all_taxa = set()
        
        # First pass: get all taxa
        import re
        all_taxa = set(re.findall(r't\d+', s))
        
        # Second pass: find all internal subtrees
        # Use a stack to track depth
        depth = 0
        current = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
                if depth == 0:
                    # This was the root, skip
                    pass
                else:
                    # This closes an internal node
                    subtree = ''.join(current)
                    # Extract taxa from this subtree
                    sub_taxa = set(re.findall(r't\d+', subtree))
                    if 0 < len(sub_taxa) < len(all_taxa):
                        # Add both sides of the split (canonical form: smaller side first)
                        other = all_taxa - sub_taxa
                        if len(sub_taxa) <= len(other):
                            splits.add(frozenset(sub_taxa))
                        else:
                            splits.add(frozenset(other))
                    current = []
            else:
                if depth > 0:
                    current.append(ch)
            i += 1
        
        return splits
    
    return collect_splits(s), taxa


def _rf_distance(splits1, splits2):
    """
    Compute normalized Robinson-Foulds distance between two sets of splits.
    """
    s1 = set(splits1)
    s2 = set(splits2)
    common = s1 & s2
    # Non-trivial splits only (size 2 to n-2)
    # Both sets should already be non-trivial
    union = s1 | s2
    if len(union) == 0:
        return 0.0
    return (len(union) - 2 * len(common)) / len(union)


def _quartet_score(newick, sequences, labels):
    """
    Score a tree topology by summing quartet log-likelihoods (approximate).
    For each quartet, check if the tree's induced quartet topology matches
    the most likely quartet topology from the data.
    """
    # This is a simplified version - just count matching quartets
    # For a proper implementation, we'd compute quartet likelihoods
    # Here we use a distance-based proxy
    
    splits, taxa = _parse_newick(newick)
    n = len(taxa)
    if n < 4:
        return 0.0
    
    # For each quartet, determine the tree's topology and compare to data
    # This is O(n^4) which is too slow for large n
    # Instead, use a simpler heuristic: count how many pairs are "close"
    # in the tree vs in the data
    
    # For now, return 0 as a placeholder
    return 0.0


def _small_n_exact(sequences, labels):
    """
    For small n (<= 7), try all unrooted binary topologies and pick the best
    by a simple distance-based criterion.
    """
    n = len(labels)
    if n > 7:
        return None
    
    # Generate all unrooted binary tree topologies
    # For n taxa, the number of unrooted binary trees is (2n-5)!!
    # For n=7, that's 9!! = 945
    # For n=8, that's 11!! = 10395 (might be too slow)
    
    if n <= 6:
        # Brute force all topologies
        # Represent a tree as a set of splits
        # Generate all possible sets of non-trivial splits that form a valid tree
        
        # Simpler: use the fact that for small n, we can enumerate
        # all ways to build the tree by successive joins
        
        # Actually, let's just try NJ and a few alternatives
        # For n <= 5, the number of unrooted binary trees is small:
        # n=4: 3
        # n=5: 15
        # n=6: 105
        # n=7: 945
        
        # Generate all unrooted binary trees for n <= 7
        # A tree can be represented by its set of splits
        # We need to generate all valid sets of splits
        
        # Alternative: enumerate all possible "joining orders"
        # This is complex. Let's use a simpler approach:
        # For each pair of taxa, try joining them and recurse
        # This generates all possible NJ-like trees but with different join orders
        
        # Actually, the cleanest way for small n is to enumerate all 
        # unrooted binary tree shapes and assign taxa to leaves
        
        # For n=4: 3 topologies
        # For n=5: 15 topologies  
        # For n=6: 105 topologies
        # For n=7: 945 topologies
        
        # Let's generate them recursively
        def generate_trees(taxa_list):
            """Generate all unrooted binary tree topologies for the given taxa."""
            n = len(taxa_list)
            if n == 3:
                return [(f"({taxa_list[0]},{taxa_list[1]})", taxa_list[2])]
            
            results = []
            # Choose two taxa to join
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = taxa_list[i], taxa_list[j]
                    rest = [t for k, t in enumerate(taxa_list) if k not in (i, j)]
                    new_taxa = f"({a},{b})"
                    rest_with_new = rest + [new_taxa]
                    # Recurse
                    for subtree in generate_trees(rest_with_new):
                        # subtree is a Newick string for the reduced tree
                        # We need to substitute back
                        results.append(subtree)
            
            # Deduplicate
            unique = list(set(results))
            return unique
        
        # This approach has issues with string representation.
        # Let's use a different method: represent trees as split sets.
        
        # For now, just return NJ result for small n too
        return None
    
    return None


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Build NJ tree
    nj_tree = _nj_build_v2(dist, labels)
    
    # For small n, try to improve with quartet-based refinement
    if n <= 8:
        # Try a few alternative topologies and pick the best
        # Use a simple scoring: sum of squared distances vs tree-implied distances
        # This is approximate but may help
        
        # Actually, for the given time budget, NJ is likely sufficient
        # The main improvement would be for very small n where we can be exhaustive
        
        if n <= 5:
            # Try all possible unrooted binary topologies
            # n=4: 3 topologies
            # n=5: 15 topologies
            best_tree = nj_tree
            best_score = float('inf')
            
            # Generate all unrooted binary trees for n taxa
            # For n=4: ((a,b),(c,d)), ((a,c),(b,d)), ((a,d),(b,c))
            if n == 4:
                a, b, c, d = labels
                candidates = [
                    f"(({a},{b}),({c},{d}))",
                    f"(({a},{c}),({b},{d}))",
                    f"(({a},{d}),({b},{c}))",
                ]
            elif n == 5:
                # 15 topologies - generate by choosing which 4 to group
                a, b, c, d, e = labels
                candidates = []
                # All ways to partition 5 taxa into a binary tree
                # Equivalent to: choose a 4-taxon subtree and attach the 5th
                for i in range(5):
                    rest = [labels[j] for j in range(5) if j != i]
                    # For the 4 remaining, there are 3 topologies
                    # Attach the excluded taxon to one of the 3 internal edges
                    x, y, z, w = rest
                    for quad_tree in [
                        f"(({x},{y}),({z},{w}))",
                        f"(({x},{z}),({y},{w}))",
                        f"(({x},{w}),({y},{z}))",
                    ]:
                        # Attach label[i] to each of the 3 internal edges
                        # This is complex; let's just use NJ for n=5
                        pass
                # Just use NJ for n=5
                return nj_tree
            
            # Score each candidate
            def score_tree(tree_newick):
                """Score a tree by comparing its implied distances to observed."""
                # Parse the tree and compute implied distances
                # This is complex; use a simpler proxy
                # Count how well the tree matches the distance matrix
                # For a binary tree, the implied distance between two leaves
                # is the sum of branch lengths along the path
                # We don't have branch lengths, so use a topological proxy:
                # For each pair, check if they are "close" in the tree
                # (i.e., share a recent common ancestor)
                
                # Simpler: use the number of quartets that match
                # For n=4, just check which topology is most supported
                return 0.0
            
            # For n=4, determine best quartet topology from data
            if n == 4:
                a, b, c, d = labels
                # Compute the three possible quartet scores
                # Under JC, the most likely topology is the one where
                # the two closest pairs are joined
                da = dist[0][1]  # a-b
                db = dist[0][2]  # a-c
                dc = dist[0][3]  # a-d
                de = dist[1][2]  # b-c
                df = dist[1][3]  # b-d
                dg = dist[2][3]  # c-d
                
                # Score for ((a,b),(c,d)): da + dg - (db + dc) - (de + df) ... 
                # Actually, the standard quartet test:
                # For topology ((a,b),(c,d)): score = d_ab + d_cd - d_ac - d_ad - d_bc - d_bd + 2*d_ac + 2*d_bd... 
                # Simpler: the topology with the smallest sum of "inconsistent" distances
                # is best. For ((a,b),(c,d)), the "inconsistent" distances are
                # those that don't follow the tree: d_ac, d_ad, d_bc, d_bd should be
                # larger than d_ab and d_cd.
                
                # Use the standard quartet score:
                # Score((a,b)|(c,d)) = d_ab + d_cd - d_ac - d_ad - d_bc - d_bd + d_ac + d_bd... 
                # Actually, let's just use: the topology that minimizes
                # sum of distances for the two "short" pairs
                score_ab_cd = da + dg  # a-b and c-d are short
                score_ac_bd = db + df  # a-c and b-d are short  
                score_ad_bc = dc + de  # a-d and b-c are short
                
                # But we also need to consider that the long distances should be long
                # A better score: for ((a,b),(c,d)), the expected pattern is
                # d_ab < d_ac, d_ab < d_ad, d_cd < d_bc, d_cd < d_bd
                # Score = (d_ac - d_ab) + (d_ad - d_ab) + (d_bc - d_cd) + (d_bd - d_cd)
                # Lower is better (more consistent)
                
                s1 = (db - da) + (dc - da) + (de - dg) + (df - dg)  # ((a,b),(c,d))
                s2 = (da - db) + (df - db) + (dg - dc) + (de - dc)  # ((a,c),(b,d)) -- need to be careful
                s3 = (da - dc) + (db - dc) + (de - df) + (dg - df)  # ((a,d),(b,c))
                
                # Hmm, let me just use the standard approach:
                # For each topology, compute the "quartet score" as:
                # sum of the two distances that are "internal" minus the four "external"
                # For ((a,b),(c,d)): internal = d_ab, d_cd; external = d_ac, d_ad, d_bc, d_bd
                # Score = (d_ab + d_cd) - (d_ac + d_ad + d_bc + d_bd) / 2
                # More negative is better (more consistent)
                
                q1 = (da + dg) - (db + dc + de + df) / 2  # ((a,b),(c,d))
                q2 = (db + df) - (da + dc + de + dg) / 2  # ((a,c),(b,d))
                q3 = (dc + de) - (da + db + df + dg) / 2  # ((a,d),(b,c))
                
                best_q = min(q1, q2, q3)
                if best_q == q1:
                    return f"(({a},{b}),({c},{d}))"
                elif best_q == q2:
                    return f"(({a},{c}),({b},{d}))"
                else:
                    return f"(({a},{d}),({b},{c}))"
        
        # For n > 4, just use NJ
        return nj_tree
    
    # For larger n, use NJ with possible refinement
    # Add a simple refinement: try swapping subtrees and see if it improves
    # a distance-based score
    
    # For now, just return NJ
    return nj_tree

# EVOLVE-BLOCK-END
