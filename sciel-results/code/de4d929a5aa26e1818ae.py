"""Phylogenetic topology recovery: NJ initialization + ML-based NNI local search."""
import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 260  # safety margin before 270s kill

def _check_deadline():
    return time.time() < _DEADLINE


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


def _parse_newick(newick):
    """Parse a Newick string into a nested tuple structure."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    def parse(s):
        if s[0] == '(':
            depth = 0
            i = 1
            parts = []
            current = ''
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                    current += s[i]
                elif s[i] == ')':
                    depth -= 1
                    if depth == -1:
                        break
                    current += s[i]
                elif s[i] == ',' and depth == 0:
                    parts.append(current)
                    current = ''
                else:
                    current += s[i]
                i += 1
            if i < len(s):
                i += 1
            rest = s[i:]
            if rest.startswith(':'):
                j = 0
                while j < len(rest) and rest[j] not in (',', '(', ')', ';'):
                    j += 1
            children = [parse(p) for p in parts]
            return ('internal', children)
        else:
            label = s
            if ':' in label:
                label = label.split(':')[0]
            return ('leaf', label)
    
    return parse(newick)


def _build_adjacency(tree):
    """Build adjacency list from tree structure."""
    adj = {}
    node_types = {}
    counter = [0]
    
    def assign_id(node, parent_id=None):
        if node[0] == 'leaf':
            nid = ('leaf', node[1])
            node_types[nid] = ('leaf', node[1])
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            return nid
        else:
            nid = ('internal', counter[0])
            counter[0] += 1
            node_types[nid] = ('internal', None)
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            for child in node[1]:
                cid = assign_id(child, nid)
            return nid
    
    root = assign_id(tree)
    node_ids = list(adj.keys())
    return adj, node_ids, node_types


def _root_tree(adj, node_ids, node_types):
    """Root the tree and build parent/children structure."""
    root = None
    for nid in node_ids:
        if node_types[nid][0] == 'internal':
            root = nid
            break
    if root is None:
        root = node_ids[0]
    
    parent = {root: None}
    children = {nid: [] for nid in node_ids}
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                children[node].append(nbr)
                queue.append(nbr)
    
    return root, parent, children


def _postorder(root, children):
    """Get postorder traversal."""
    order = []
    def _post(node):
        for child in children[node]:
            _post(child)
        order.append(node)
    _post(root)
    return order


def _jc_transition_matrix(length):
    """Compute JC transition probability matrix for given branch length."""
    # P(same) = 1/4 + 3/4 * exp(-4*length/3)
    # P(diff) = 1/4 * (1 - exp(-4*length/3))
    exp_term = math.exp(-4.0 * length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = (1.0 - exp_term) / 3.0
    # Matrix: rows = from state, cols = to state
    # Due to symmetry, we just need p_same and p_diff
    return p_same, p_diff


def _ml_score_tree(adj, node_ids, node_types, sequences, labels):
    """Compute ML log-likelihood of the tree given the sequences using Felsenstein's pruning."""
    n_taxa = len(labels)
    if n_taxa < 2:
        return 0.0
    
    n_sites = len(sequences[labels[0]])
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build rooted structure
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # For each internal edge, estimate branch length using ML
    # We'll use a simple approach: fix all branch lengths to a reasonable value
    # For topology search, the relative comparison is what matters
    
    # Estimate a global branch length scale
    # Use the mean JC distance as a guide
    total_d = 0.0
    count = 0
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            total_d += _jc_distance(sequences[a], sequences[b])
            count += 1
    mean_d = total_d / max(1, count)
    # Typical branch length per edge ~ mean_d / (n_taxa - 2)
    base_length = max(0.01, min(0.5, mean_d / max(1, n_taxa - 2)))
    
    total_ll = 0.0
    
    for site in range(n_sites):
        # For each node, compute P(state at node | data below)
        prob = {}
        
        # Initialize leaves
        for nid in order:
            if node_types[nid][0] == 'leaf':
                label = node_types[nid][1]
                obs = state_map[sequences[label][site]]
                p = [0.0, 0.0, 0.0, 0.0]
                p[obs] = 1.0
                prob[nid] = p
            else:
                prob[nid] = None
        
        # Process internal nodes in postorder
        for nid in order:
            if node_types[nid][0] == 'leaf':
                continue
            
            # Get branch length from this node to its parent
            if parent[nid] is not None:
                blen = base_length
            else:
                blen = base_length  # Root has no parent, but we'll handle it
            
            p_same, p_diff = _jc_transition_matrix(blen)
            
            p_node = [0.0, 0.0, 0.0, 0.0]
            for state in range(4):
                total = 1.0
                for child in children[nid]:
                    child_sum = 0.0
                    for c_state in range(4):
                        p_transition = p_same if c_state == state else p_diff
                        child_sum += p_transition * prob[child][c_state]
                    total *= child_sum
                p_node[state] = total
            
            prob[nid] = p_node
        
        # Root probability: average over all states
        root_prob = sum(prob[root]) / 4.0
        if root_prob > 1e-10:
            total_ll += math.log(root_prob)
        else:
            total_ll += -1000.0
    
    return total_ll


def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure.
    
    Original: u connected to {v, a, b}, v connected to {u, c, d}
    The four subtrees are A (at u, via a), B (at u, via b), C (at v, via c), D (at v, via d)
    
    NNI move 1: swap so that A and C are on one side, B and D on the other
    New: u connected to {a, v}, v connected to {u, b, c, d}? No...
    
    Actually, standard NNI:
    Original quartet: ((A,B),(C,D)) with internal edge (u,v)
    where u connects to A, B, v and v connects to u, C, D
    
    NNI1: ((A,C),(B,D)) - remove (u,v), connect u to A and C, v to B and D? No.
    
    Let me think more carefully:
    Original: 
        A --- u --- v --- C
              |     |
              B     D
    
    NNI1 (swap C and D):
        A --- u --- v --- D
              |     |
              B     C
    
    NNI2 (swap B and C):
        A --- u --- v --- B
              |     |
              C     D
    
    Wait, that's not right either. Let me use the standard definition:
    
    For internal edge (u,v), with u having neighbors {v, x, y} and v having neighbors {u, z, w}:
    Original: (x,y) | (z,w)
    NNI1: (x,z) | (y,w)  
    NNI2: (x,w) | (y,z)
    
    So for NNI1:
    - Remove edges: (u,v), (u,x), (u,y), (v,z), (v,w)
    - Add edges: (u,x), (u,v), (v,y), (v,z), (v,w)? No, that adds too many.
    
    Actually, the correct NNI operation:
    Original: u--x, u--y, u--v, v--z, v--w
    NNI1: u--x, u--v, v--y, v--z, v--w? No.
    
    Let me look at this differently. The NNI move changes the topology from:
    (x, (y, (z, w))) to (x, (z, (y, w))) or (x, (w, (y, z)))
    
    In terms of the internal edge (u,v):
    Original: u is parent of x and y (and v), v is parent of z and w (and u)
    After NNI1: u is parent of x and v, v is parent of y, z, w (and u)
    
    So the edges change from:
    Original: (u,x), (u,y), (u,v), (v,z), (v,w)
    NNI1: (u,x), (u,v), (v,y), (v,z), (v,w)
    
    That means we remove (u,y) and add (v,y).
    
    NNI2: (u,x), (u,v), (v,y), (v,w), (v,z) -> wait, same as NNI1?
    
    No, let me be more careful:
    NNI1: swap the subtrees on one side
    Original: ((x,y),(z,w))
    NNI1: ((x,z),(y,w))
    
    So:
    Original edges: u-x, u-y, u-v, v-z, v-w
    NNI1 edges: u-x, u-v, v-y, v-z, v-w? No, that gives ((x,v),(y,z,w)) which is wrong.
    
    I think the issue is that in the unrooted tree, the "parent" relationship is arbitrary. Let me just implement it as:
    
    For NNI on edge (u,v) where u has neighbors {v,a,b} and v has neighbors {u,c,d}:
    
    NNI1: Remove (u,a), (u,b), (v,c), (v,d), (u,v). Add (u,a), (u,v), (v,b), (v,c), (v,d).
    This changes which subtrees are attached where.
    
    Actually, the simplest correct implementation:
    NNI1: swap b and c
    - Remove edge (u,b) and (v,c)
    - Add edge (u,c) and (v,b)
    
    NNI2: swap b and d
    - Remove edge (u,b) and (v,d)
    - Add edge (u,d) and (v,b)
    
    Let me verify: Original has u connected to a,b,v and v connected to u,c,d.
    NNI1 (swap b,c): u connected to a,c,v and v connected to u,b,d.
    This changes the split from (a,b)|(c,d) to (a,c)|(b,d). Yes!
    
    NNI2 (swap b,d): u connected to a,d,v and v connected to u,c,b.
    This changes the split from (a,b)|(c,d) to (a,d)|(b,c). Yes!
    """
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        # Swap b and c: remove (u,b) and (v,c), add (u,c) and (v,b)
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        # Swap b and d: remove (u,b) and (v,d), add (u,d) and (v,b)
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj


def _nni_local_search(adj, node_ids, node_types, sequences, labels, max_iterations=20):
    """Perform NNI local search to improve tree topology using ML scoring."""
    current_ll = _ml_score_tree(adj, node_ids, node_types, sequences, labels)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        # Find all internal edges
        best_ll = current_ll
        best_move = None
        
        for u in node_ids:
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if v <= u:  # avoid duplicates
                    continue
                if node_types[v][0] != 'internal':
                    continue
                
                # Get neighbors of u and v (excluding each other)
                u_nbrs = [x for x in adj[u] if x != v]
                v_nbrs = [x for x in adj[v] if x != u]
                
                if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                    continue
                
                a = u_nbrs[0]
                b = u_nbrs[1]
                c = v_nbrs[0]
                d = v_nbrs[1]
                
                # Try NNI1: swap b and c
                new_adj1 = _apply_nni(adj, node_types, u, v, a, b, c, d, 1)
                ll1 = _ml_score_tree(new_adj1, node_ids, node_types, sequences, labels)
                if ll1 > best_ll + 1e-6:
                    best_ll = ll1
                    best_move = (new_adj1, 1)
                
                # Try NNI2: swap b and d
                new_adj2 = _apply_nni(adj, node_ids, node_types, u, v, a, b, c, d, 2)
                ll2 = _ml_score_tree(new_adj2, node_ids, node_types, sequences, labels)
                if ll2 > best_ll + 1e-6:
                    best_ll = ll2
                    best_move = (new_adj2, 2)
        
        if best_move is not None:
            adj = best_move[0]
            current_ll = best_ll
            improved = True
    
    return adj


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    # Find a leaf to start from, or any node
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    # Build Newick by traversing from root
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in adj[nid]:
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"


def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Step 1: Compute distance matrix using JC correction
    matrix = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                d = _jc_distance(sequences[a], sequences[b])
                matrix[(a, b)] = d
                matrix[(b, a)] = d
    
    # Step 2: Neighbor-Joining to get initial tree
    nodes = list(labels)
    trees = {name: name for name in labels}
    
    while len(nodes) > 3:
        if not _check_deadline():
            break
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        if best is None:
            break
        
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    
    # Build initial Newick
    newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Step 4: NNI local search with ML scoring
    if _check_deadline():
        adj = _nni_local_search(adj, node_ids, node_types, sequences, labels, max_iterations=15)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            # Fallback to NJ result if something went wrong
            return newick
    
    return result
