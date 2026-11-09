import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _jc_distance_matrix(A):
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d

class _Node:
    __slots__ = ['left', 'right', 'label', 'parent']
    def __init__(self, label=None):
        self.left = None
        self.right = None
        self.label = label
        self.parent = None

def _nj_tree(D, labels):
    n = len(labels)
    if n == 1:
        return _Node(labels[0])
    if n == 2:
        r = _Node()
        a, b = _Node(labels[0]), _Node(labels[1])
        r.left, r.right = a, b
        a.parent = b.parent = r
        return r

    # Use arrays for speed
    active = list(range(n))
    nodes = [_Node(labels[i]) for i in range(n)]
    # dist is a dict keyed by (i,j) with i,j in active set
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]

    next_id = n
    while len(active) > 2:
        m = len(active)
        # Compute r[i] = sum_j D[i,j]
        r = np.zeros(m)
        for i_idx, i in enumerate(active):
            s = 0.0
            for j_idx, j in enumerate(active):
                if j != i:
                    s += dist[(i, j)]
            r[i_idx] = s

        best_val = float('inf')
        best_a = best_b = -1
        for i_idx in range(m):
            i = active[i_idx]
            for j_idx in range(i_idx + 1, m):
                j = active[j_idx]
                crit = (m - 2) * dist[(i, j)] - r[i_idx] - r[j_idx]
                if crit < best_val:
                    best_val = crit
                    best_a, best_b = i, j

        a, b = best_a, best_b
        new_id = next_id
        next_id += 1
        new_node = _Node()
        new_node.left = nodes[a]
        new_node.right = nodes[b]
        nodes[a].parent = nodes[b].parent = new_node
        nodes.append(new_node)

        # Update distances
        d_ab = dist[(a, b)]
        rest = [x for x in active if x != a and x != b]
        for x in rest:
            d_ax = dist[(a, x)]
            d_bx = dist[(b, x)]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            dist[(new_id, x)] = d_new
            dist[(x, new_id)] = d_new

        active.remove(a)
        active.remove(b)
        active.append(new_id)

    a, b = active
    root = _Node()
    root.left = nodes[a]
    root.right = nodes[b]
    nodes[a].parent = nodes[b].parent = root
    return root

def _node_to_newick(node):
    if node.label is not None:
        return node.label
    return "({},{})".format(_node_to_newick(node.left), _node_to_newick(node.right))

def _fitch_cost(root, A, label_to_idx, bit_map):
    """Compute Fitch parsimony cost. Returns (cost, bitset_dict)."""
    n, L = A.shape
    A_int = A.astype(np.int32)
    bitsets = {}
    cost = 0

    def rec(node):
        nonlocal cost
        if node.label is not None:
            idx = label_to_idx[node.label]
            bs = bit_map[A_int[idx]]
            bitsets[node] = bs
            return bs
        lb = rec(node.left)
        rb = rec(node.right)
        inter = lb & rb
        if inter != 0:
            bs = inter
        else:
            cost += 1
            bs = lb | rb
        bitsets[node] = bs
        return bs

    rec(root)
    return cost, bitsets

def _get_internal_edges(root):
    """Return list of (u, v) where u is parent of v, both internal."""
    edges = []

    def walk(node):
        if node.label is not None:
            return
        if node.left is not None and node.left.label is None:
            edges.append((node, node.left))
            walk(node.left)
        if node.right is not None and node.right.label is None:
            edges.append((node, node.right))
            walk(node.right)

    walk(root)
    return edges

def _nni_search(root, A, label_to_idx, deadline, per_instance_budget):
    """NNI local search minimizing Fitch parsimony cost."""
    n, L = A.shape
    bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
    A_int = A.astype(np.int32)

    current_cost, bitsets = _fitch_cost(root, A, label_to_idx, bit_map)
    best_root = root
    best_cost = current_cost

    start = time.time()
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        edges = _get_internal_edges(root)

        for u, v in edges:
            if time.time() > deadline - 2.0:
                return best_root, best_cost

            # u is parent of v, both internal
            # v has two children: v.left and v.right
            # u's other child (sibling of v): let's call it w
            # u's parent: let's call it p (or None if u is root)

            v_left = v.left
            v_right = v.right

            # Find sibling of v (u's other child)
            if u.left == v:
                w = u.right
            else:
                w = u.left

            # Find the "outside" subtree: everything above u
            # If u is root, outside is None (special case)
            p = u.parent

            # For NNI, we need the four subtrees:
            # S1 = v_left, S2 = v_right, S3 = w, S4 = outside (above u)
            # Current: ((S1,S2), S3) with S4 above
            # Move 1: ((S1,S4), S3) -- swap S2 and S4
            # Move 2: ((S3,S1), S2) -- swap S2 and S3

            # We need bitsets for each subtree
            # S1 bitset = bitsets[v_left]
            # S2 bitset = bitsets[v_right]
            # S3 bitset = bitsets[w]
            # S4 bitset: need to compute "complement" of u's subtree in the whole tree

            # Actually, for NNI cost evaluation, we can use a trick:
            # The Fitch cost change for an NNI move can be computed from
            # the bitsets of the four subtrees.

            # Let's compute the "outside" bitset by doing a full recomputation
            # but only for the affected nodes. This is complex.

            # Simpler approach: just do a full Fitch recomputation after each move.
            # For n up to 32, this is fast enough.

            # Try Move 1: swap v_right and the outside
            # Structure: u has children v and w. v has children v_left, v_right.
            # Move 1: make v have children v_left and outside, and u have children v and w
            # But "outside" is not a single node if u is not root.

            # Let me use a different NNI formulation.
            # Internal edge u-v. The four pendant subtrees are:
            # A = v.left, B = v.right, C = w (sibling of v under u), D = rest of tree above u

            # Current topology: ((A,B),C) with D attached to u
            # NNI Move 1: ((A,D),C)  -- B and D swap
            # NNI Move 2: ((C,A),B)  -- B and C swap

            # To implement these, I need to rewire the tree.

            # Let's implement Move 2 first (simpler, doesn't involve "outside"):
            # ((C,A),B) instead of ((A,B),C)
            # This means: u now has children (new_node) and w, where new_node has children C and A
            # Wait, that's not right either.

            # Let me think again. The internal edge is u-v.
            # u's children: v and w
            # v's children: v_left and v_right
            #
            # Move 2 (swap v_right and w):
            # New structure: u has children v' and w', where
            #   v' has children v_left and w
            #   w' ... no, this is getting confusing.

            # Standard NNI: for internal edge connecting nodes X and Y,
            # where X has children A and Y, and Y has children B and C:
            # Current: (A, (B, C)) with X's other side being D
            # Actually in unrooted: the four subtrees hanging off the edge are A, B, C, D
            # where A is on X's side (sibling of Y), B and C are on Y's side, D is X's parent side.

            # Let me just do a full rewire:
            # Current: u -> [v, w], v -> [v_left, v_right]
            #
            # NNI Option 1: swap v_right with w
            # New: u -> [v', w'], v' -> [v_left, w], w' -> ... no
            # Actually: u -> [v', w], v' -> [v_left, v_right]... that's the same.

            # I think the cleanest way is:
            # The edge u-v splits the tree into two halves:
            # Half 1 (v side): {v_left, v_right}
            # Half 2 (u side): {w, ...above u}
            #
            # NNI swaps one leaf from each half.
            # Option A: swap v_left with w -> new v has [w, v_right], u has [v_new, above]
            # Option B: swap v_right with w -> new v has [v_left, w], u has [v_new, above]

            # Let's implement Option B (swap v_right and w):
            # Save old pointers
            old_v_left = v.left
            old_v_right = v.right
            old_w = w

            # New: v has children old_v_left and old_w
            # u has children v and (old_v_right)
            # But old_v_right was a subtree, now it becomes a direct child of u?
            # That would make u have 3 children, which is invalid.

            # I see the issue. In a binary tree, each internal node has exactly 2 children.
            # The NNI move restructures the internal nodes.

            # Correct NNI for edge u-v where u=[v,w] and v=[a,b]:
            # Move 1: u=[v',w], v'=[a,w]... no, w can't be in two places.

            # Let me look at this differently. The four subtrees are:
            # S_a = v.left, S_b = v.right, S_w = w, S_up = (above u)
            #
            # Current: ((S_a, S_b), S_w) with S_up connected to u
            #
            # NNI Move 1: ((S_a, S_up), S_w) -- S_b and S_up swap
            # NNI Move 2: ((S_w, S_a), S_b) -- S_b and S_w swap

            # For Move 2: S_b and S_w swap
            # New structure: u has children [v_new, S_b], v_new has children [S_w, S_a]
            # This keeps everything binary!

            # Let's implement Move 2:
            # v_new = new internal node with children [w, v_left]
            # u's children become [v_new, v_right]

            # Check if this is different from current
            if (v.left == w or v.right == w):
                continue  # shouldn't happen in valid tree

            # Create new node
            v_new = _Node()
            v_new.left = w
            v_new.right = v_left
            w.parent = v_new
            v_left.parent = v_new

            # Rewire u
            u.left = v_new
            u.right = v_right
            v_new.parent = u
            v_right.parent = u

            # Remove old v (it's now orphaned)
            # v is no longer in the tree

            # Compute new cost
            new_cost, _ = _fitch_cost(root, A, label_to_idx, bit_map)

            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
                # v is now orphaned, v_new is the new node
                # Update: the tree is modified in place
                # v_new replaces v in the structure
                # We need to track that v is gone
                pass
            else:
                # Revert
                v_new.left.parent = v
                v_new.right.parent = v
                v.left = old_v_left
                v.right = old_v_right
                w.parent = u
                u.left = v
                u.right = old_w
                v.parent = u
                v_right.parent = v

            # Also try Move 1: swap v_left and v_right roles with w
            # Actually, let's also try the other NNI: swap v_left with w
            # New: v_new2 has children [w, v_right], u has children [v_new2, v_left]

            v_new2 = _Node()
            v_new2.left = w
            v_new2.right = v_right
            w.parent = v_new2
            v_right.parent = v_new2

            u.left = v_new2
            u.right = v_left
            v_new2.parent = u
            v_left.parent = u

            new_cost2, _ = _fitch_cost(root, A, label_to_idx, bit_map)

            if new_cost2 < current_cost:
                current_cost = new_cost2
                improved = True
            else:
                # Revert
                v_new2.left.parent = v
                v_new2.right.parent = v
                v.left = old_v_left
                v.right = old_v_right
                w.parent = u
                u.left = v
                u.right = old_w
                v.parent = u
                v_right.parent = v

    return root, current_cost

def reconstruct(sequences):
    global _reconstruct_start_time
    if not hasattr(reconstruct, '_start_time'):
        reconstruct._start_time = time.time()

    elapsed = time.time() - reconstruct._start_time
    # Each instance gets a proportional budget
    # 20 instances in 270s total, so ~13s per instance with margin
    # But we don't know how many instances are left, so use a conservative per-instance budget
    per_instance_budget = 12.0  # seconds per instance
    deadline = time.time() + per_instance_budget

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"

    L = len(sequences[labels[0]])

    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    A = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            A[i, j] = char_map[c]

    label_to_idx = {label: i for i, label in enumerate(labels)}

    # 1. NJ tree
    D = _jc_distance_matrix(A)
    root = _nj_tree(D, labels)

    # 2. NNI search
    if time.time() < deadline - 1.0 and n >= 4:
        bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
        root, cost = _nni_search(root, A, label_to_idx, deadline, per_instance_budget)

    tree = _node_to_newick(root) + ";"
    return tree

# EVOLVE-BLOCK-END
