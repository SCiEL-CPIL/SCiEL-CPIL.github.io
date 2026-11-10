"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Quartet Fitting: count quartet topologies, build consensus tree via greedy split selection.'''
import time
import numpy as np
from itertools import combinations

_DEADLINE_START = None
_DEADLINE_LIMIT = 265.0


def _elapsed():
    return time.time() - _DEADLINE_START


def _quartet_fitting(sequences, labels):
    n = len(labels)
    if n < 4:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    L = len(sequences[labels[0]])
    enc = np.empty((n, L), dtype=np.int8)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for pos, ch in enumerate(s):
            enc[i, pos] = code[ch]

    # Compute JC distances
    matches = np.zeros((n, n), dtype=np.float64)
    for c in range(4):
        Mc = (enc == c).astype(np.float32)
        matches += Mc @ Mc.T
    mismatches = L - matches
    p = mismatches / L
    np.fill_diagonal(p, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    d = np.where(p >= 0.75, 3.0, d)
    d = np.where(np.isnan(d) | np.isinf(d), 3.0, d)
    np.fill_diagonal(d, 0.0)

    # Precompute quartet votes for all 4-subsets
    # vote_cache: dict from sorted tuple (a,b,c,d) to winning topology index
    # 0: ((a,b),(c,d)), 1: ((a,c),(b,d)), 2: ((a,d),(b,c))
    vote_cache = {}
    for quartet in combinations(range(n), 4):
        a, b, c, dd = quartet
        s1 = d[a, b] + d[c, dd]
        s2 = d[a, c] + d[b, dd]
        s3 = d[a, dd] + d[b, c]
        best = min(s1, s2, s3)
        if best == s1:
            vote_cache[quartet] = 0
        elif best == s2:
            vote_cache[quartet] = 1
        else:
            vote_cache[quartet] = 2

    # Build tree using greedy algorithm: recursively find best split
    def _build_tree(taxa_set):
        if _elapsed() > 255.0:
            # Deadline: return star tree
            return "(" + ",".join(labels[t] for t in sorted(taxa_set)) + ");"
        if len(taxa_set) == 1:
            return labels[list(taxa_set)[0]]
        if len(taxa_set) == 2:
            t = sorted(taxa_set)
            return "(" + labels[t[0]] + "," + labels[t[1]] + ");"
        if len(taxa_set) == 3:
            t = sorted(taxa_set)
            return "(" + labels[t[0]] + "," + labels[t[1]] + "," + labels[t[2]] + ");"

        taxa_list = sorted(taxa_set)
        m = len(taxa_list)

        # Try candidate splits: single taxon vs rest, and pair vs rest
        best_score = -1e18
        best_split = None

        # Single taxon splits
        for i in range(m):
            left = frozenset([taxa_list[i]])
            right = frozenset(taxa_list) - left
            score = _score_split(left, right, vote_cache)
            if score > best_score:
                best_score = score
                best_split = (left, right)

        # Pair splits
        for i in range(m):
            for j in range(i + 1, m):
                left = frozenset([taxa_list[i], taxa_list[j]])
                right = frozenset(taxa_list) - left
                if not right:
                    continue
                score = _score_split(left, right, vote_cache)
                if score > best_score:
                    best_score = score
                    best_split = (left, right)

        if best_split is None:
            return "(" + ",".join(labels[t] for t in taxa_list) + ");"

        left, right = best_split
        left_tree = _build_tree(left)
        right_tree = _build_tree(right)
        return "(" + left_tree + "," + right_tree + ");"

    def _score_split(left, right, votes):
        """Score split by counting quartets consistent with it."""
        if not left or not right:
            return 0.0
        score = 0.0
        left_list = sorted(left)
        right_list = sorted(right)

        # Count 2-2 quartets (2 from left, 2 from right)
        for a in left_list:
            for b in left_list:
                if a >= b:
                    continue
                for c in right_list:
                    for dd in right_list:
                        if c >= dd:
                            continue
                        key = tuple(sorted([a, b, c, dd]))
                        vote = votes.get(key, 0)
                        # Determine if vote is consistent with left|right
                        # left has {a,b}, right has {c,dd}
                        # Consistent if vote is 0: ((a,b),(c,dd))
                        # Need to check based on actual positions
                        # Sort the quartet and check
                        quartet_sorted = key
                        # Find which are in left
                        in_left = [x in left for x in quartet_sorted]
                        # vote 0: ((q0,q1),(q2,q3))
                        # vote 1: ((q0,q2),(q1,q3))
                        # vote 2: ((q0,q3),(q1,q2))
                        if vote == 0:
                            # ((q0,q1),(q2,q3))
                            if in_left[0] == in_left[1] and in_left[2] == in_left[3] and in_left[0] != in_left[2]:
                                score += 1.0
                            else:
                                score -= 0.5
                        elif vote == 1:
                            # ((q0,q2),(q1,q3))
                            if in_left[0] == in_left[2] and in_left[1] == in_left[3] and in_left[0] != in_left[1]:
                                score += 1.0
                            else:
                                score -= 0.5
                        else:
                            # ((q0,q3),(q1,q2))
                            if in_left[0] == in_left[3] and in_left[1] == in_left[2] and in_left[0] != in_left[1]:
                                score += 1.0
                            else:
                                score -= 0.5

        return score

    return _build_tree(frozenset(range(n)))


def _reconstruct_single(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"

    return _quartet_fitting(sequences, labels)


def reconstruct(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    try:
        return _reconstruct_single(sequences)
    except Exception:
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        if len(labels) <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
