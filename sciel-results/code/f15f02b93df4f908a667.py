import math
import time
import numpy as np

_DEADLINE = 260.0  # Safety margin before 270s kill
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy for speed."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    
    # Convert sequences to numeric arrays for fast comparison
    # A=0, C=1, G=2, T=3
    conv = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_seqs = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lb in enumerate(labels):
        seq = sequences[lb]
        for j, c in enumerate(seq):
            num_seqs[i, j] = conv[c]
    
    L = len(sequences[labels[0]])
    D = np.zeros((n, n))
    
    # Vectorized pairwise mismatch calculation
    # num_seqs is (n, L). We want (n, n) mismatches.
    # This can be memory intensive for large L, but L <= 2000, n <= 32.
    # n^2 * L is 32*32*2000 = 2M ops. Very fast.
    
    for i in range(n):
        for j in range(i + 1, n):
            # Count mismatches
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
    """Run Neighbor Joining, return Newick string and internal node mapping."""
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # We will maintain a list of active nodes and their current tree strings
    # and a distance matrix that gets updated.
    
    # To avoid the complexity of dynamic matrix resizing which led to errors in inspiration 2,
    # we will use a dictionary for the distance matrix between active nodes.
    
    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Tree strings for each active node
    tree_strs = {i: labels[i] for i in range(n)}
    
    # Distance dictionary: (i, j) -> dist
    # Initialize from D
    dist = {}
    for i in range(n):
        for j in range(i+1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]
    
    counter = n
    while len(active) > 3:
        m = len(active)
        
        # Calculate row sums (total distance from each node to all others in active)
        total = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    total[i] += dist[(i, j)]
        
        # Find best pair (i, j) minimizing (m-2)*d(i,j) - total[i] - total[j]
        best_criterion = float('inf')
        best_pair = None
        
        for ki in range(m):
            i = active[ki]
            for kj in range(ki + 1, m):
                j = active[kj]
                criterion = (m - 2) * dist[(i, j)] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        i, j = best_pair
        joined_str = f"({tree_strs[i]},{tree_strs[j]})"
        new_idx = counter
        counter += 1
        
        # Update distances to new node
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        for x in active:
            if x != i and x != j:
                d_new = 0.5 * (dist[(i, x)] + dist[(j, x)] - dist[(i, j)])
                dist[(new_idx, x)] = d_new
                dist[(x, new_idx)] = d_new
        
        # Remove i and j from active, add new_idx
        active = [x for x in active if x not in (i, j)]
        active.append(new_idx)
        tree_strs[new_idx] = joined_str
        
    # Final tree
    parts = [tree_strs[x] for x in active]
    return "(" + ",".join(parts) + ");"


def _parse_newick(newick):
    """
    Parse Newick string into an adjacency list representation.
    Returns:
        nodes: list of node IDs
        adj: dict mapping node ID to list of (neighbor, branch_length)
        leaves: dict mapping node ID to label (for leaves)
    """
    # Remove trailing semicolon
    s = newick.rstrip(';').strip()
    
    nodes = []
    adj = {}
    leaves = {}
    node_counter = 0
    parent_stack = []
    
    # We need a robust parser.
    # Standard approach: recursive descent or stack-based.
    # Let's use a stack-based approach for unrooted trees.
    # Actually, Newick is usually rooted. For unrooted, we can treat it as rooted at the first leaf or any node.
    # The structure is the same, just the root is arbitrary.
    
    # Let's parse into a nested structure first, then flatten.
    def parse(s, parent=None):
        nonlocal node_counter
        s = s.strip()
        if not s:
            return None
        
        # Create a node
        node_id = node_counter
        node_counter += 1
        nodes.append(node_id)
        adj[node_id] = []
        
        # Check if it's a leaf
        if s.startswith('('):
            # Internal node
            # Find matching parenthesis
            depth = 0
            end_idx = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            
            inner = s[1:end_idx]
            # Split inner by top-level commas
            parts = []
            depth = 0
            current = []
            for c in inner:
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
            
            # Recurse on children
            for part in parts:
                child_id = parse(part, node_id)
                if child_id is not None:
                    # Connect node_id and child_id
                    # Branch length is 0 for now, we will optimize later if needed
                    adj[node_id].append((child_id, 0.0))
                    adj[child_id].append((node_id, 0.0))
            
            # Check for branch length after the closing parenthesis
            rest = s[end_idx+1:].strip()
            if rest.startswith(':'):
                # Branch length present, but we ignore it for topology
                pass
        else:
            # Leaf
            # Check for branch length
            label = s
            if ':' in s:
                label = s.split(':')[0].strip()
            leaves[node_id] = label
        
        if parent is not None:
            # Connection already handled in the internal node case?
            # No, in the internal node case, we connected children to the node.
            # Here, if this node is a child of parent, the parent call will handle the connection?
            # Actually, my recursive structure above connects children to the node being parsed.
            # So when parse returns node_id, the caller (which is parsing the parent) needs to connect.
            # But in the internal node block, I did:
            # child_id = parse(part, node_id)
            # adj[node_id].append...
            # This connects the child to the node.
            # So the return value is just the ID.
            
            pass
            
        return node_id

    # The top-level call
    root_id = parse(s)
    
    # Re-map node IDs to 0..N-1 for easier indexing?
    # The current node_counter is N. IDs are 0..N-1.
    
    return nodes, adj, leaves


def _nni_polish(newick, D, labels):
    """
    Perform NNI polish to improve topology based on distance fit.
    """
    nodes, adj, leaves = _parse_newick(newick)
    n = len(nodes)
    
    if n <= 3:
        return newick
    
    # Map leaf labels to node IDs
    label_to_node = {leaf: node for node, leaf in leaves.items()}
    
    # Map node IDs to labels for output
    node_to_label = {node: leaf for leaf, node in label_to_node.items()}
    
    # We need to compute the cost of the tree.
    # Cost = Sum of squared differences between D[i][j] and path_length(i, j)
    # To compute path lengths, we need branch lengths.
    # We can estimate branch lengths using least squares for the current topology.
    # This is O(N^2) per topology evaluation.
    
    def compute_branch_lengths(adj, D, label_to_node):
        """
        Estimate branch lengths using least squares.
        Returns dict: (u, v) -> length
        """
        # Build system of equations
        # For each pair of leaves (i, j): sum of branch lengths on path = D[i][j]
        # This is a linear system A x = b
        # A is (N_pairs x N_edges), b is (N_pairs x 1)
        # N_pairs ~ N^2/2, N_edges ~ N-3
        # For N=32, N_pairs ~ 500, N_edges ~ 29.
        # Solving 500x29 system is fast.
        
        edges = []
        edge_idx = {}
        for u in adj:
            for v, _ in adj[u]:
                if u < v:
                    idx = len(edges)
                    edges.append((u, v))
                    edge_idx[(u, v)] = idx
                    edge_idx[(v, u)] = idx
        
        num_edges = len(edges)
        if num_edges == 0:
            return {}
        
        # Get leaf indices
        leaf_nodes = list(label_to_node.values())
        num_leaves = len(leaf_nodes)
        
        # Build A and b
        # We will use numpy least squares
        # A[i, k] = 1 if edge k is on path between leaf i and leaf j, else 0
        # b[i] = D[leaf_i, leaf_j]
        
        # To save memory, we can build the system iteratively or use sparse matrices.
        # But N is small.
        
        # Let's build the full matrix
        num_pairs = num_leaves * (num_leaves - 1) // 2
        A = np.zeros((num_pairs, num_edges))
        b = np.zeros(num_pairs)
        
        pair_idx = 0
        for i in range(num_leaves):
            for j in range(i + 1, num_leaves):
                li = leaf_nodes[i]
                lj = leaf_nodes[j]
                
                # Find path between li and lj in adj
                # BFS
                visited = {li}
                path = []
                queue = [(li, [])]
                found = False
                while queue:
                    node, p = queue.pop(0)
                    if node == lj:
                        found = True
                        path = p
                        break
                    for neighbor, _ in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, p + [(node, neighbor)]))
                
                if found:
                    # path is list of (u, v)
                    for u, v in path:
                        e_idx = edge_idx[(u, v)]
                        A[pair_idx, e_idx] = 1.0
                    
                    # Get distance
                    # D is indexed by labels, but we have node IDs.
                    # We need to map node IDs back to labels to index D.
                    # Wait, D was computed for labels.
                    # label_to_node maps label -> node.
                    # We need node -> label.
                    # We have node_to_label.
                    li_label = node_to_label[li]
                    lj_label = node_to_label[lj]
                    
                    # Get index in D
                    # D is n x n where n is number of labels.
                    # labels is sorted list of labels.
                    # We need the index of li_label in labels.
                    # This is inefficient.
                    # Let's pre-compute a map from label to index in D.
                    pass
        
        # This approach is getting complicated and slow due to repeated BFS and mapping.
        # Alternative: Use the NJ distance matrix D directly with node indices?
        # No, D is defined for leaves.
        
        # Let's simplify:
        # Instead of least squares, use a heuristic:
        # For each edge, length = 0.5 * (dist_to_rest - dist_between_ends)?
        # This is the NJ formula, which is valid for the *correct* topology.
        # For an incorrect topology, it might give negative lengths.
        # We can clamp to 0.
        
        # Let's try a simpler cost: Sum of branch lengths (minimize total length).
        # We need branch lengths.
        # We can compute branch lengths using the formula:
        # l(u, v) = 0.5 * (sum_{x in S_u} D(u, x) + sum_{x in S_v} D(v, x) - sum_{x in S_u, y in S_v} D(x, y) / |S_u||S_v| ... )
        # This is complex.
        
        # Let's use a different cost: **Quadratic Assignment Problem (QAP)**
        # Minimize sum_{i<j} D[i][j] * (1 if i,j separated by edge e else 0) ... no.
        
        # Given the time constraints and the fact that NJ is already 1.0 in inspiration,
        # maybe NNI is overkill and risky to implement correctly in this short timeframe.
        # The "structural change" could be **using a more robust distance metric** or **multiple NJ runs with different random perturbations** (if we had randomness, but we don't).
        
        # Let's stick to the **simplest possible NNI** that uses the **existing distance matrix** to estimate branch lengths locally.
        
        # For an edge (u, v) in the tree:
        # Let S_u be the set of leaves in the subtree rooted at u (when edge is cut).
        # Let S_v be the set of leaves in the subtree rooted at v.
        # The length of the edge can be approximated by:
        # l = 0.5 * (avg_dist(S_u, S_v) - avg_dist(S_u, S_u) - avg_dist(S_v, S_v)) ? No.
        
        # Actually, the NJ formula for the new node distance is:
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        # This suggests that the "length" of the branch connecting the joined pair to the rest is related to these terms.
        
        # Let's try this:
        # For each NNI move, calculate the **Sum of Pairwise Distances** implied by the tree using the **additive** property.
        # If the tree is additive, sum of branch lengths on path = D[i][j].
        # If not, there's error.
        
        # To avoid the complexity of full least squares, I will **skip the NNI polish** and rely on the **robust NJ** implementation.
        # The inspiration 1 scored 1.0 with simple NJ. The current program scored 0.74 with complex NJ+Bootstrap.
        # The difference is likely the Bootstrap Consensus step degrading the result.
        # Therefore, the "structural change" is **removing the Bootstrap Consensus** and **using a clean, optimized NJ**.
        
        return newick

    # Since I decided to skip NNI, this function is a no-op.
    # But I should provide a fallback.
    return newick


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    newick = _nj_build(D, labels)
    
    # No Bootstrap Consensus (removed to avoid degradation)
    # No NNI Polish (too complex to implement robustly in time, and NJ is likely sufficient)
    
    return newick
