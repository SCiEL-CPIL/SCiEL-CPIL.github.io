"""Phylogenetic topology recovery: NJ + fast quartet-based NNI refinement under JC."""
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255.0  # Safety margin before 270s kill


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])

    # Convert sequences to integer arrays
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arr = np.array([[mapping[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)

    # Count differences via broadcasting
    diff = (seq_arr[:, np.newaxis, :] != seq_arr[np.newaxis, :, :]).sum(axis=-1)

    p = diff.astype(np.float64) / L
    p = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) * 0.5

    return D


def _nj_build(D, labels):
    """
    Neighbor Joining returning a tree structure.
    Returns:
        newick_str: Newick string
        leaf_to_node: dict mapping leaf label to leaf node id
        node_to_leaves: dict mapping node id to set of leaf labels in its subtree
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else "", {labels[0]: 0}, {0: {labels[0]}}
    if n == 2:
        return f"({labels[0]},{labels[1]});", {labels[0]: 0, labels[1]: 1}, {0: {labels[0]}, 1: {labels[1]}}
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});", {labels[0]: 0, labels[1]: 1, labels[2]: 2}, {0: {labels[0]}, 1: {labels[1]}, 2: {labels[2]}}

    # Active nodes: start with 0..n-1
    active = list(range(n))
    newick = {i: labels[i] for i in range(n)}
    node_to_leaves = {i: {labels[i]} for i in range(n)}

    # Distance dictionary
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]

    next_id = n

    while len(active) > 3:
        m = len(active)
        if time.time() > _DEADLINE - 10.0:
            break

        # Compute Q criterion
        r = {a: sum(dist[(a, b)] for b in active if b != a) for a in active}

        best_val = float('inf')
        best_a, best_b = active[0], active[1]

        for i in range(m):
            a = active[i]
            for j in range(i + 1, m):
                b = active[j]
                q_val = (m - 2) * dist[(a, b)] - r[a] - r[b]
                if q_val < best_val:
                    best_val = q_val
                    best_a, best_b = a, b

        a, b = best_a, best_b

        # Create new node
        new_id_val = next_id
        next_id += 1
        new_newick = f"({newick[a]},{newick[b]})"
        new_leaves = node_to_leaves[a] | node_to_leaves[b]

        # Update distances
        ab_dist = dist[(a, b)]
        for x in active:
            if x != a and x != b:
                new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - ab_dist)
                dist[(new_id_val, x)] = new_d
                dist[(x, new_id_val)] = new_d

        # Remove a, b; add new node
        active = [x for x in active if x != a and x != b]
        active.append(new_id_val)
        newick[new_id_val] = new_newick
        node_to_leaves[new_id_val] = new_leaves

    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        newick_str = f"({newick[a]},({newick[b]},{newick[c]}));"
    else:
        newick_str = "(" + ",".join(newick[x] for x in active) + ");"

    # Build leaf_to_node mapping
    leaf_to_node = {}
    for i in range(n):
        leaf_to_node[labels[i]] = i

    return newick_str, leaf_to_node, node_to_leaves


def _parse_newick_leaves(newick_str, label_set):
    """
    Parse a Newick string and return a list of leaf labels in order.
    Assumes leaves are from label_set.
    """
    leaves = []
    i = 0
    n = len(newick_str)

    while i < n:
        ch = newick_str[i]
        if ch == '(':
            # Find matching ')'
            depth = 0
            j = i
            while j < n:
                if newick_str[j] == '(':
                    depth += 1
                elif newick_str[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            # Recurse into the subtree
            subtree = newick_str[i+1:j]
            sub_leaves = _parse_newick_leaves(subtree, label_set)
            leaves.extend(sub_leaves)
            i = j + 1
            # Skip any label after ')'
            if i < n and newick_str[i] == ':':
                i += 1
                while i < n and newick_str[i] not in ',;':
                    i += 1
            elif i < n and newick_str[i] == ',':
                i += 1
        elif ch in ',;':
            i += 1
        else:
            # Read a label
            j = i
            while j < n and newick_str[j] not in '(),;:':
                j += 1
            label = newick_str[i:j]
            if label in label_set:
                leaves.append(label)
            i = j

    return leaves


def _nni_refine_quartet(D, newick_str, labels, label_set):
    """
    NNI refinement using quartet scores under JC model.
    
    For each internal edge, try the two alternative topologies and accept
    if the quartet score improves.
    
    Returns the refined Newick string.
    """
    n = len(labels)
    if n < 4:
        return newick_str

    # Build label index
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE - 5.0:
            break

        improved = False
        newick = newick_str

        # Find all positions where we have an internal edge.
        # An internal edge in Newick: the pattern is (subtree1,subtree2),subtree3
        # where subtree1 and subtree2 are both internal (contain '(').
        # In Newick, this appears as "),(" with the left part ending in ")"
        # and the right part starting with "(".

        positions = []
        for pos in range(len(newick) - 3):
            if newick[pos:pos+2] == ")," and newick[pos+2:pos+3] == "(":
                positions.append(pos)

        for pos in positions:
            if time.time() > _DEADLINE - 3.0:
                break

            # Extract the four subtrees
            # Left subtree: the part before ")," that is inside the left parens
            # Right subtree: the part after ")," that is inside the right parens

            # Find the matching open paren for the left part
            left_end = pos  # position of ')'
            depth = 0
            left_start = left_end
            for k in range(left_end, -1, -1):
                if newick[k] == ')':
                    depth += 1
                elif newick[k] == '(':
                    depth -= 1
                    if depth == 0:
                        left_start = k
                        break

            # Find the matching close paren for the right part
            right_start = pos + 2  # position of '('
            depth = 0
            right_end = right_start
            for k in range(right_start, len(newick)):
                if newick[k] == '(':
                    depth += 1
                elif newick[k] == ')':
                    depth -= 1
                    if depth == 0:
                        right_end = k
                        break

            # Left subtree content
            left_content = newick[left_start+1:left_end]
            # Right subtree content
            right_content = newick[right_start+1:right_end]

            # Split left into two subtrees
            # Find the top-level comma in left_content
            depth = 0
            comma_pos = -1
            for k in range(len(left_content)):
                if left_content[k] == '(':
                    depth += 1
                elif left_content[k] == ')':
                    depth -= 1
                elif left_content[k] == ',' and depth == 0:
                    comma_pos = k
                    break

            if comma_pos == -1:
                continue

            left_a = left_content[:comma_pos]
            left_b = left_content[comma_pos+1:]

            # Split right into two subtrees
            depth = 0
            comma_pos2 = -1
            for k in range(len(right_content)):
                if right_content[k] == '(':
                    depth += 1
                elif right_content[k] == ')':
                    depth -= 1
                elif right_content[k] == ',' and depth == 0:
                    comma_pos2 = k
                    break

            if comma_pos2 == -1:
                continue

            right_c = right_content[:comma_pos2]
            right_d = right_content[comma_pos2+1:]

            # Get leaf sets for each subtree
            leaves_a = _parse_newick_leaves(left_a, label_set)
            leaves_b = _parse_newick_leaves(left_b, label_set)
            leaves_c = _parse_newick_leaves(right_c, label_set)
            leaves_d = _parse_newick_leaves(right_d, label_set)

            if not leaves_a or not leaves_b or not leaves_c or not leaves_d:
                continue

            # Compute quartet score for current topology: (a,b)|(c,d)
            # and the two alternatives
            # The quartet score Q_ab_cd = D_ab + D_cd - D_ac - D_ad - D_bc - D_bd
            # The best topology is the one with the most negative Q.

            def compute_quartet_score(leaves1, leaves2, leaves3, leaves4):
                """
                Compute sum of quartet scores for all quartets with one leaf from each set.
                For topology (1,2)|(3,4), Q = D12 + D34 - D13 - D14 - D23 - D24
                """
                total = 0.0
                for la in leaves1:
                    ia = label_to_idx[la]
                    for lb in leaves2:
                        ib = label_to_idx[lb]
                        for lc in leaves3:
                            ic = label_to_idx[lc]
                            for ld in leaves4:
                                id_ = label_to_idx[ld]
                                # Q for (la,lb)|(lc,ld)
                                Q = (D[ia, ib] + D[ic, id_] - D[ia, ic] - D[ia, id_] - 
                                     D[ib, ic] - D[ib, id_])
                                total += Q
                return total

            # Current topology: (a,b)|(c,d)
            score_current = compute_quartet_score(leaves_a, leaves_b, leaves_c, leaves_d)

            # Alternative 1: (a,c)|(b,d)
            score_alt1 = compute_quartet_score(leaves_a, leaves_c, leaves_b, leaves_d)

            # Alternative 2: (a,d)|(b,c)
            score_alt2 = compute_quartet_score(leaves_a, leaves_d, leaves_b, leaves_c)

            # We want to minimize the quartet score (more negative is better)
            best_score = min(score_current, score_alt1, score_alt2)

            if best_score < score_current - 1e-9:
                # Perform the NNI move
                if best_score == score_alt1:
                    # New topology: (a,c)|(b,d)
                    # The new Newick should have (a,c) on the left and (b,d) on the right
                    new_left = f"({left_a},{right_c})"
                    new_right = f"({left_b},{right_d})"
                    new_segment = f"({new_left},{new_right})"
                else:
                    # New topology: (a,d)|(b,c)
                    new_left = f"({left_a},{right_d})"
                    new_right = f"({left_b},{right_c})"
                    new_segment = f"({new_left},{new_right})"

                # Replace the old segment with the new one
                old_segment = f"({left_content},{right_content})"
                if old_segment in newick:
                    newick = newick.replace(old_segment, new_segment, 1)
                    improved = True
                    break  # Restart the scan

        newick_str = newick
        if not improved:
            break

    return newick_str


def _build_newick_from_nj(D, labels):
    """Build Newick string using NJ."""
    newick_str, _, _ = _nj_build(D, labels)
    return newick_str


def reconstruct(sequences):
    global _START_TIME, _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    try:
        # 1. Distance Matrix
        D = _jc_distance_matrix(sequences, labels)

        # 2. NJ
        newick = _build_newick_from_nj(D, labels)

        # 3. NNI Refinement with Quartet Scores
        if time.time() < _DEADLINE - 10.0:
            label_set = set(labels)
            newick = _nni_refine_quartet(D, newick, labels, label_set)

        # Ensure the Newick ends with ';'
        if not newick.endswith(';'):
            newick += ';'

        return newick

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
