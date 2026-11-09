"""Phylogenetic topology recovery: vectorized Neighbor-Joining baseline plus
fast quartet-based split refinement with a hard deadline guard."""
import time
import math
import re
import numpy as np

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 255.0  # safety margin under the 270s wall-clock kill


def _encode(sequences, labels):
    """Map sequences to an int8 array of shape (n, L) over {0,1,2,3}."""
    n = len(labels)
    L = len(sequences[labels[0]])
    arr = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # vectorized char->code via a translation table
        arr[i] = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
    # map ASCII codes: A=65,C=67,G=71,T=84 -> 0,1,2,3
    lut = np.zeros(256, dtype=np.int8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    return lut[arr.astype(np.uint8)]


def _jc_distance_matrix(codes):
    """Jukes-Cantor corrected distance matrix from int8 code array (n, L)."""
    n, L = codes.shape
    D = np.zeros((n, n), dtype=np.float64)
    # Hamming: for each base, count matches via one-hot matmul
    for b in range(4):
        M = (codes == b).astype(np.int8)
        D += M @ M.T
    D = L - D
    np.fill_diagonal(D, 0.0)
    p = np.clip(D / float(L), 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 * p) / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels, deadline=None):
    """Fully vectorized Neighbor-Joining. Returns a Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + ",(" + labels[1] + "," + labels[2] + "));"

    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    dist = D.copy()
    next_id = n

    while len(active) > 3:
        if deadline is not None and time.time() > deadline:
            break
        m = len(active)
        # row sums over active
        sub = dist[np.ix_(active, active)]
        row_sums = sub.sum(axis=1)
        # NJ criterion matrix
        crit = (m - 2) * sub - row_sums[:, None] - row_sums[None, :]
        # ignore diagonal
        np.fill_diagonal(crit, np.inf)
        i, j = np.unravel_index(np.argmin(crit), crit.shape)
        a = active[i]
        b = active[j]
        new_id = next_id
        next_id += 1

        rest = [active[k] for k in range(m) if k != i and k != j]
        # new distances
        new_d = 0.5 * (dist[a, :][rest] + dist[b, :][rest] - dist[a, b])
        for x, val in zip(rest, new_d):
            dist[new_id, x] = val
            dist[x, new_id] = val

        tree_str[new_id] = "(" + tree_str[a] + "," + tree_str[b] + ")"
        active = rest + [new_id]

    rem = active
    return "(" + ",".join(tree_str[x] for x in rem) + ");"


def _quartet_votes(codes, deadline):
    """
    Fast quartet voting. For every 4-taxon quartet, pick the split whose two
    within-group Hamming distances sum to the minimum. Accumulate votes and
    margins per normalized split. Returns dict frozenset -> (count, margin).
    """
    n, L = codes.shape
    allset = frozenset(range(n))
    t_start = time.time()

    # Precompute pairwise agreement (number of equal sites) for all pairs.
    agree = np.zeros((n, n), dtype=np.int32)
    for a in range(n):
        ca = codes[a]
        for b in range(a + 1, n):
            agree[a, b] = int(np.sum(ca == codes[b]))
            agree[b, a] = agree[a, b]

    votes = {}
    for a in range(n):
        for b in range(a + 1, n):
            if time.time() - t_start > deadline:
                return votes
            ab = agree[a, b]
            for c in range(b + 1, n):
                ac = agree[a, c]
                bc = agree[b, c]
                for d in range(c + 1, n):
                    ad = agree[a, d]
                    bd = agree[b, d]
                    cd = agree[c, d]
                    s_ab_cd = (L - ab) + (L - cd)
                    s_ac_bd = (L - ac) + (L - bd)
                    s_ad_bc = (L - ad) + (L - bc)
                    m = min(s_ab_cd, s_ac_bd, s_ad_bc)
                    if m == s_ab_cd:
                        split = frozenset((a, b))
                        margin = min(s_ac_bd, s_ad_bc) - m
                    elif m == s_ac_bd:
                        split = frozenset((a, c))
                        margin = min(s_ab_cd, s_ad_bc) - m
                    else:
                        split = frozenset((a, d))
                        margin = min(s_ab_cd, s_ac_bd) - m
                    comp = allset - split
                    if len(comp) < len(split):
                        split = comp
                    if split in votes:
                        votes[split] = (votes[split][0] + 1, votes[split][1] + margin)
                    else:
                        votes[split] = (1, margin)
    return votes


def _select_compatible_splits(votes, n, deadline):
    """Greedy maximum-weight independent set on the split conflict graph."""
    allset = frozenset(range(n))
    t_start = time.time()
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    chosen = []
    chosen_comps = []
    for split, (cnt, margin) in ranked:
        if time.time() - t_start > deadline:
            break
        if len(chosen) >= n - 3:
            break
        comp = allset - split
        ok = True
        for existing, ecomp in zip(chosen, chosen_comps):
            if (split & existing) and (split & ecomp) and (comp & existing) and (comp & ecomp):
                ok = False
                break
        if ok:
            chosen.append(split)
            chosen_comps.append(comp)
    return chosen


def _build_tree_from_splits(chosen_splits, labels, D, deadline):
    """Build a tree from compatible splits via recursive decomposition,
    falling back to NJ on any subtree that no split partitions."""
    allset = frozenset(range(len(labels)))

    def build(taxa_fset):
        if time.time() > deadline:
            idx_list = sorted(taxa_fset)
            sub = D[np.ix_(idx_list, idx_list)]
            return _neighbor_joining(sub, [labels[i] for i in idx_list])
        if len(taxa_fset) == 1:
            return labels[list(taxa_fset)[0]]
        if len(taxa_fset) == 2:
            a, b = sorted(taxa_fset)
            return "(" + labels[a] + "," + labels[b] + ")"
        best_part = None
        for sp in chosen_splits:
            part = sp & taxa_fset
            if 0 < len(part) < len(taxa_fset):
                best_part = part
                break
        if best_part is None:
            idx_list = sorted(taxa_fset)
            sub = D[np.ix_(idx_list, idx_list)]
            return _neighbor_joining(sub, [labels[i] for i in idx_list])
        other = taxa_fset - best_part
        return "(" + build(best_part) + "," + build(other) + ")"

    return build(allset)


def _valid(newick, labels, n):
    found = re.findall(r't\d+', newick)
    if len(found) != n:
        return False
    for lab in labels:
        if found.count(lab) != 1:
            return False
    return True


def reconstruct(sequences):
    t0 = time.time()
    deadline = DEADLINE_SECONDS

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        return "(" + ",".join(labels) + ");"

    codes = _encode(sequences, labels)
    D = _jc_distance_matrix(codes)

    # Refinement: quartet-based split consensus, only if we have time to spare.
    try:
        elapsed = time.time() - t0
        if elapsed < deadline - 10.0:
            votes = _quartet_votes(codes, deadline * 0.45)
            if votes:
                chosen = _select_compatible_splits(votes, n, deadline * 0.2)
                if chosen:
                    newick = _build_tree_from_splits(chosen, labels, D, deadline * 0.85)
                    if _valid(newick, labels, n):
                        return newick
    except Exception:
        pass

    # Robust baseline: vectorized Neighbor-Joining (always a valid binary tree).
    try:
        newick = _neighbor_joining(D, labels, deadline)
        if _valid(newick, labels, n):
            return newick
    except Exception:
        pass

    return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
