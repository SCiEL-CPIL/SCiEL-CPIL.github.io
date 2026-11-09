"""Phylogenetic topology recovery: bootstrap majority-rule NJ consensus."""
import time
import math
import numpy as np
from collections import Counter
import re

# EVOLVE-BLOCK-START

# ---- Distance matrix (Jukes-Cantor correction), vectorized -----------------
def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    # Map each sequence to 4 one-hot integer arrays for fast hamming
    codes = np.zeros((n, L, 4), dtype=np.int8)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            codes[i, j, base_map[ch]] = 1
    # Hamming distance matrix via matrix multiplication per base
    D = np.zeros((n, n), dtype=np.float64)
    for b in range(4):
        M = codes[:, :, b]  # (n, L)
        agree = M @ M.T      # (n, n) counts of both having base b
        D += agree
    D = 4.0 * L - D          # hamming distances
    np.fill_diagonal(D, 0.0)
    p = D / float(L)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels):
    """Vectorized Neighbor Joining. Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + ",(" + labels[1] + "," + labels[2] + "));"

    # dist matrix indexed by node id; nodes 0..n-1 are leaves
    dist = D.copy()
    tree_str = {i: labels[i] for i in range(n)}
    active = set(range(n))
    next_id = n

    while len(active) > 3:
        m = len(active)
        # row sums over active nodes
        total = np.zeros(n, dtype=np.float64)
        for a in active:
            total[a] = np.sum([dist[a, b] for b in active if b != a])
        # find best pair
        best_crit = float('inf')
        best_a, best_b = None, None
        act_list = sorted(active)
        for ia in range(len(act_list)):
            a = act_list[ia]
            for ib in range(ia + 1, len(act_list)):
                b = act_list[ib]
                crit = (m - 2) * dist[a, b] - total[a] - total[b]
                if crit < best_crit:
                    best_crit = crit
                    best_a, best_b = a, b
        a, b = best_a, best_b
        new = next_id
        next_id += 1
        # extend dist matrix
        old_n = dist.shape[0]
        new_dist = np.zeros((old_n + 1, old_n + 1), dtype=np.float64)
        new_dist[:old_n, :old_n] = dist
        rest = [x for x in active if x != a and x != b]
        for x in rest:
            val = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            new_dist[new, x] = val
            new_dist[x, new] = val
        dist = new_dist
        tree_str[new] = "(" + tree_str[a] + "," + tree_str[b] + ")"
        active.discard(a)
        active.discard(b)
        active.add(new)

    rem = sorted(active)
    return "(" + ",".join(tree_str[x] for x in rem) + ");"


def _splits_from_newick(newick, n):
    """Return set of non-trivial splits as frozensets of indices 0..n-1."""
    # Parse Newick to collect internal node leaf sets
    allset = frozenset(range(n))
    internals = []
    pos = 0
    s = newick.rstrip(';')
    
    def parse(pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse(pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            merged = set()
            for c in children:
                merged |= c
            internals.append(merged)
            return merged, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',);':
                pos += 1
            name = s[start:pos].strip()
            # Extract index from label like t0, t1, etc.
            idx = int(name[1:])
            return {idx}, pos

    parse(0)
    splits = set()
    for m in internals:
        mf = frozenset(m)
        comp = allset - mf
        if 0 < len(mf) < len(allset):
            splits.add(min(mf, comp, key=len))
    return splits


def _build_tree_from_splits(splits, labels):
    """Build a Newick tree from a set of compatible splits."""
    n = len(labels)
    allset = frozenset(range(n))
    
    if not splits:
        return "(" + ",".join(labels) + ");"
    
    def build(taxa):
        if len(taxa) == 1:
            return labels[list(taxa)[0]]
        if len(taxa) == 2:
            a, b = sorted(taxa)
            return "(" + labels[a] + "," + labels[b] + ")"
        # Find a split that partitions taxa
        for sp in splits:
            part = sp & taxa
            if 0 < len(part) < len(taxa):
                left = build(part)
                right = build(taxa - part)
                return "(" + left + "," + right + ")"
        # No split found: return star
        return "(" + ",".join(labels[i] for i in sorted(taxa)) + ")"
    
    return build(allset) + ";"


def _bootstrap_consensus(sequences, labels, deadline):
    """Run bootstrap NJ and return majority-rule consensus tree."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    # Compute full distance matrix first
    D_full = _jc_distance_matrix(sequences, labels)
    
    # Determine number of bootstrap replicates based on available time
    # Each NJ takes roughly O(n^3) operations; estimate time per NJ
    # For n=32, NJ is fast with vectorized ops. Let's do as many as we can.
    
    # Start with full-data NJ as baseline
    t0 = time.time()
    newick_full = _neighbor_joining(D_full, labels)
    splits_full = _splits_from_newick(newick_full, n)
    
    elapsed = time.time() - t0
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return newick_full
    
    # Bootstrap: resample columns with replacement
    # Use numpy random for speed
    rng = np.random.RandomState(42)
    
    # Collect split votes
    split_votes = Counter()
    split_votes.update(splits_full)  # full data gets 1 vote
    
    # Number of bootstrap replicates: aim to use ~80% of remaining time
    # Estimate time per bootstrap replicate
    # First do a quick timing
    t_start = time.time()
    # Do one bootstrap replicate to estimate time
    boot_idx = rng.randint(0, L, size=L)
    boot_seqs = {lab: ''.join(sequences[lab][i] for i in boot_idx) for lab in labels}
    D_boot = _jc_distance_matrix(boot_seqs, labels)
    newick_boot = _neighbor_joining(D_boot, labels)
    splits_boot = _splits_from_newick(newick_boot, n)
    split_votes.update(splits_boot)
    t_one = time.time() - t_start
    
    # Estimate how many more replicates we can afford
    # Leave 10% safety margin
    budget = remaining * 0.8
    max_reps = int(budget / max(t_one, 0.01))
    max_reps = min(max_reps, 500)  # cap at 500
    
    n_reps_done = 1
    for rep in range(max_reps):
        if time.time() - t0 > deadline * 0.9:
            break
        boot_idx = rng.randint(0, L, size=L)
        # Build bootstrapped sequences
        boot_seqs = {}
        for lab in labels:
            seq = sequences[lab]
            boot_seqs[lab] = ''.join(seq[i] for i in boot_idx)
        D_boot = _jc_distance_matrix(boot_seqs, labels)
        newick_boot = _neighbor_joining(D_boot, labels)
        splits_boot = _splits_from_newick(newick_boot, n)
        split_votes.update(splits_boot)
        n_reps_done += 1
    
    # Majority rule: keep splits with >50% support
    total_votes = sum(split_votes.values())
    majority_splits = set()
    for sp, count in split_votes.items():
        if count > total_votes / 2.0:
            majority_splits.add(sp)
    
    if not majority_splits:
        return newick_full
    
    # Build tree from majority splits
    try:
        newick = _build_tree_from_splits(majority_splits, labels)
        # Validate
        found = re.findall(r't\d+', newick)
        cnt = Counter(found)
        if len(cnt) == n and all(cnt[lab] == 1 for lab in labels):
            return newick
    except Exception:
        pass
    
    return newick_full


def reconstruct(sequences):
    t0 = time.time()
    deadline = 240.0  # safe margin under 270s

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        return "(" + ",".join(labels) + ");"

    try:
        newick = _bootstrap_consensus(sequences, labels, deadline - (time.time() - t0))
        found = re.findall(r't\d+', newick)
        cnt = Counter(found)
        if len(cnt) == n and all(cnt[lab] == 1 for lab in labels):
            return newick
    except Exception:
        pass

    # Fallback: single NJ
    try:
        D = _jc_distance_matrix(sequences, labels)
        newick = _neighbor_joining(D, labels)
        found = re.findall(r't\d+', newick)
        cnt = Counter(found)
        if len(cnt) == n and all(cnt[lab] == 1 for lab in labels):
            return newick
    except Exception:
        pass

    # Last resort: star tree
    return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
