"""Stronger algorithm: precomputed K(2,2,2) conflicts + multi-round hill climbing."""
import random
import time
import itertools
import numpy as np

# Precompute all triples and their conflict sets
_ALL_TRIPLES = list(itertools.combinations(range(16), 3))
_N = len(_ALL_TRIPLES)
_TRIPLE_TO_IDX = {t: i for i, t in enumerate(_ALL_TRIPLES)}

# For each triple, precompute the set of triples that form a K(2,2,2) with it
# Two triples t1, t2 conflict if there exist 3 disjoint pairs of points such that
# all 8 transversal triples are present (including t1 and t2).
#
# K(2,2,2) structure: 6 points partitioned into 3 pairs {a1,a2}, {b1,b2}, {c1,c2}
# The 8 transversal triples are: (a_i, b_j, c_k) for all i,j,k in {1,2}
# So a K(2,2,2) is a set of 8 triples on 6 points where each point appears in 4 triples.
#
# Two triples t1=(x1,x2,x3) and t2=(y1,y2,y3) can be part of the same K(2,2,2) iff
# together they use exactly 6 distinct points (i.e., they are disjoint).
# Given two disjoint triples, there's exactly one way to partition the 6 points into 3 pairs
# such that both triples are transversal... actually let me think more carefully.
#
# Given 6 points and two disjoint triples, we need to check if there's a pairing of the 6
# points into 3 pairs such that both triples are transversal to the pairing.
# A triple (p,q,r) is transversal to pairing {(a1,a2),(b1,b2),(c1,c2)} iff {p,q,r}
# contains exactly one element from each pair.
#
# For two disjoint triples T1 and T2 on 6 points: T1 is transversal to pairing P iff
# P is a "perfect matching" that splits T1 into one-from-each-pair. Same for T2.
# The number of such pairings is limited. Let's enumerate:
# Given T1 = {a,b,c} and T2 = {d,e,f} (all distinct), a pairing P makes T1 transversal
# iff each pair in P contains exactly one of {a,b,c} and one of {d,e,f}.
# So P must be a perfect matching between T1 and T2. There are 3! = 6 such matchings.
# For each such matching, T2 is automatically transversal (by symmetry).
# So two disjoint triples are in a common K(2,2,2) iff they are disjoint, and the K(2,2,2)
# is determined by choosing one of the 6 matchings. But the K(2,2,2) is the set of all 8
# transversal triples for that matching. Two triples conflict if they belong to the same
# K(2,2,2), i.e., if there exists a matching making both transversal.
# Since any two disjoint triples have 6 matchings making both transversal, ANY two
# disjoint triples are in a common K(2,2,2)!
#
# Wait, that means the conflict is not just pairwise. The constraint is: no 8-triple
# K(2,2,2) is fully present. So the "conflict" for a triple is: if adding it would
# complete any K(2,2,2).
#
# Let me re-approach: precompute for each K(2,2,2) (there are finitely many), the set
# of 8 triples. Then for the hill climbing, when considering adding triple t, check
# if it would complete any K(2,2,2) that already has 7 of its 8 triples present.

# Enumerate all K(2,2,2) configurations
# A K(2,2,2) is determined by: choice of 6 points, and a partition into 3 pairs.
# Number: C(16,6) * (1/2^3) * (6!/(2!^3 * 3!)) = C(16,6) * 15
# C(16,6) = 8008, so 8008 * 15 = 120120 K(2,2,2)s.
# For each, the 8 transversal triples.

def _enumerate_k222s():
    """Return list of frozensets of 8 triple-indices forming a K(2,2,2)."""
    k222s = []
    for six in itertools.combinations(range(16), 6):
        s = set(six)
        # Generate all partitions of 6 points into 3 unordered pairs
        # Canonical: a<b, c<d, e<f with a<c<e
        a, b, c, d, e, f = six
        # All ways to partition {a,b,c,d,e,f} into 3 pairs
        # Fix a's partner, then partition remaining 4
        for partner in (b, c, d, e, f):
            rem = [x for x in (b, c, d, e, f) if x != partner]
            # Partition rem (4 elements) into 2 pairs: fix first's partner
            p1 = rem[0]
            for p2 in (rem[1], rem[2], rem[3]):
                rem2 = [x for x in rem[1:] if x != p2]
                p3, p4 = rem2
                pairs = [(a, partner), (p1, p2), (p3, p4)]
                # Check no duplicate pairs
                if len(set(pairs)) == 3:
                    # Compute 8 transversal triples
                    triples = []
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                t = (pairs[0][i], pairs[1][j], pairs[2][k])
                                t = tuple(sorted(t))
                                triples.append(t)
                    # Deduplicate
                    triples = list(set(triples))
                    if len(triples) == 8:
                        idxs = frozenset(_TRIPLE_TO_IDX[t] for t in triples)
                        k222s.append(idxs)
    return k222s

_K222S = _enumerate_k222s()
# Deduplicate
_K222S = list(set(_K222S))

# For each triple index, list of K(2,2,2) indices it belongs to
_TRIPLE_TO_K222S = [[] for _ in range(_N)]
for ki, k222 in enumerate(_K222S):
    for ti in k222:
        _TRIPLE_TO_K222S[ti].append(ki)


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    rng = random.Random(42)
    
    best_set = set()
    best_count = 0
    
    def _is_valid_addition(current_set, triple_idx):
        """Check if adding triple_idx to current_set creates a complete K(2,2,2)."""
        for ki in _TRIPLE_TO_K222S[triple_idx]:
            k222 = _K222S[ki]
            # Check if all other 7 triples in this K(2,2,2) are in current_set
            # Use a quick check: count how many of the 8 are in current_set
            # If 7 are already present, adding the 8th completes it.
            present = 0
            for ti in k222:
                if ti == triple_idx:
                    continue
                if ti in current_set:
                    present += 1
            if present == 7:
                return False
        return True
    
    def _greedy_init(order):
        """Greedy: add triples in given order if valid."""
        s = set()
        for ti in order:
            if _is_valid_addition(s, ti):
                s.add(ti)
        return s
    
    def _hill_climb(s):
        """Try to add triples; if stuck, remove one to escape."""
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each absent triple
            candidates = [ti for ti in range(_N) if ti not in s]
            rng.shuffle(candidates)
            for ti in candidates:
                if time.time() >= deadline:
                    break
                if _is_valid_addition(s, ti):
                    s.add(ti)
                    improved = True
            if improved:
                continue
            # Stuck: remove a random triple to escape local optimum
            if s:
                # Prefer removing triples that are in many K(2,2,2)s (more constraining)
                # Or just random
                rem = rng.choice(list(s))
                s.discard(rem)
                # After removal, try to add again
                candidates2 = [ti for ti in range(_N) if ti not in s]
                rng.shuffle(candidates2)
                for ti in candidates2:
                    if time.time() >= deadline:
                        break
                    if _is_valid_addition(s, ti):
                        s.add(ti)
                        improved = True
    
    # Round 1: multiple greedy initializations
    for attempt in range(20):
        if time.time() >= deadline:
            break
        order = list(range(_N))
        rng.shuffle(order)
        s = _greedy_init(order)
        if len(s) > best_count:
            best_count = len(s)
            best_set = set(s)
    
    # Round 2: hill climb from best and perturbed versions
    if best_set:
        s = _hill_climb(set(best_set))
        if len(s) > best_count:
            best_count = len(s)
            best_set = set(s)
    
    # Round 3: multiple perturbed restarts
    for attempt in range(50):
        if time.time() >= deadline:
            break
        if best_set:
            s = set(best_set)
            # Perturb: remove k random triples
            k = rng.randint(3, 8)
            for _ in range(min(k, len(s))):
                if s:
                    s.discard(rng.choice(list(s)))
        else:
            order = list(range(_N))
            rng.shuffle(order)
            s = _greedy_init(order)
        
        s = _hill_climb(s)
        if len(s) > best_count:
            best_count = len(s)
            best_set = set(s)
    
    # Convert back to list of triples
    result = [_ALL_TRIPLES[ti] for ti in sorted(best_set)]
    return [list(t) for t in result]
