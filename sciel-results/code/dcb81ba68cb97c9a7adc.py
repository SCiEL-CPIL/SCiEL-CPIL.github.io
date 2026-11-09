"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _build_conflict_graph():
    """Build conflict graph: two triples conflict if together with a third triple they form K(2,2,2)."""
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n = len(all_triples)
    
    # For each pair of disjoint triples, check if they are part of a K(2,2,2)
    # Two triples {a,b,c} and {a',b',c'} are in a K(2,2,2) iff the 6 points
    # {a,b,c,a',b',c'} are exactly 6 distinct points AND all 8 transversals exist.
    # Actually, for the conflict graph, we need: triple t1 and t2 conflict if
    # there exists a triple t3 such that t1,t2,t3 are part of a K(2,2,2).
    # 
    # More precisely: t1 and t2 conflict if adding both would make it impossible
    # to add any t3 without completing a K(2,2,2). But actually the standard approach:
    # t1 and t2 are in conflict if there exists a set of 6 points {a,a',b,b',c,c'}
    # such that t1 and t2 are two of the 8 transversals. Then if we also have t3
    # (another transversal), we might complete the K(2,2,2).
    #
    # Wait, the constraint is: no three disjoint pairs have all eight transversals present.
    # So a K(2,2,2) is formed when all 8 triples on 6 points {a,a',b,b',c,c'} are present.
    # Two triples t1, t2 conflict if they are both transversals on some common 6-point set.
    # That is, if there exist 6 points such that t1 and t2 are each a transversal on those 6 points.
    
    # For a 6-point set {p0,p1,p2,p3,p4,p5}, the 8 transversals are:
    # {p0,p1,p2}, {p0,p1,p3}, {p0,p2,p4}, {p0,p2,p5}, {p0,p3,p4}, {p0,p3,p5}, {p0,p4,p5}, {p1,p2,p3},
    # {p1,p2,p4}, {p1,p2,p5}, {p1,p3,p4}, {p1,p3,p5}, {p2,p3,p4}, {p2,p3,p5}, {p1,p4,p5}, {p2,p4,p5}
    # Wait, that's 16. Let me reconsider.
    
    # K(2,2,2) is the complete tripartite graph with parts of size 2.
    # The 8 transversals (triples with one from each part) on parts {a,a'}, {b,b'}, {c,c'} are:
    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # So two triples t1, t2 conflict if they share a common 6-point support where they are both transversals.
    # That is, if t1 = {x,y,z} and t2 = {x',y',z'}, and the 6 points {x,y,z,x',y',z'} are all distinct,
    # and t1 and t2 can be written as transversals on some partition of these 6 points into 3 pairs.
    
    # For two disjoint triples t1={a,b,c} and t2={d,e,f} (all 6 distinct), they are transversals
    # on the same 6-point set if there exists a partition of {a,b,c,d,e,f} into 3 pairs such that
    # both t1 and t2 pick one from each pair.
    
    # Actually, any two disjoint triples on 6 distinct points are always transversals on some partition?
    # No. Consider t1={a,b,c}, t2={d,e,f}. We need to partition {a,b,c,d,e,f} into 3 pairs
    # {p0,p0'}, {p1,p1'}, {p2,p2'} such that t1 has one from each pair and t2 has one from each pair.
    # Since t1 and t2 are disjoint and together cover all 6 points, t2 is the complement of t1.
    # So we need a perfect matching between t1 and t2. There are 3! = 6 such matchings.
    # So yes, any two disjoint triples on 6 distinct points are always "conflicting" in the sense
    # that they are transversals on the same 6-point set (with multiple possible pairings).
    
    # Wait, but that would mean ANY two disjoint triples conflict, which is too strong.
    # Let me re-read the problem. "no three disjoint pairs have all eight transversal triples present"
    # So the forbidden configuration is: 6 points partitioned into 3 pairs, and all 8 transversals present.
    # Two triples t1, t2 are in conflict if they are both transversals on some common 6-point set.
    # That is, if there exist 6 points and a partition into 3 pairs such that both t1 and t2 are transversals.
    
    # For t1={a,b,c} and t2={d,e,f} with all 6 distinct: they are transversals on the same 6-point set
    # if there's a partition of {a,b,c,d,e,f} into 3 pairs such that t1 picks one from each and t2 picks one from each.
    # Since t1 and t2 are disjoint and cover all 6 points, this is equivalent to: there exists a bijection
    # from t1 to t2. Which always exists. So yes, any two disjoint triples on 6 distinct points conflict.
    
    # But wait, that means the conflict graph is the disjointness graph, and we want a maximum set of
    # triples such that no two are disjoint? That would be a star (all triples through one point), which
    # gives C(15,2) = 105 triples. But the current program gets 2.3048 * seed, and seed is 105,
    # so 2.3048 * 105 ≈ 242. That's much larger than 105. So my understanding is wrong.
    
    # Let me reconsider. The issue is that t1 and t2 being transversals on the same 6-point set
    # doesn't mean they conflict. They only conflict if TOGETHER with a third triple t3, they complete
    # the K(2,2,2). So the conflict is not pairwise but involves a third triple.
    
    # Actually, for the independent set formulation: two triples t1, t2 are in conflict if
    # there exists a triple t3 such that {t1, t2, t3} are all transversals on the same 6-point set.
    # In other words, t1 and t2 conflict if they are "close" to completing a K(2,2,2) with some t3.
    
    # For t1={a,b,c} and t2={d,e,f} with all 6 distinct: t1 and t2 are transversals on the 6-point set
    # {a,b,c,d,e,f} if there's a partition into 3 pairs. As established, this is always possible.
    # So t1 and t2 conflict if there exists a third transversal t3 on the same 6-point set.
    # Since there are 8 transversals total, and t1, t2 are two of them, there are 6 others.
    # So t1 and t2 always conflict (since there are always other transversals that could be t3).
    
    # Hmm, but that still gives the same conclusion. Let me think differently.
    
    # The correct interpretation: we want to select triples such that for NO 6-point set
    # (partitioned into 3 pairs) are ALL 8 transversals present.
    # Two triples t1, t2 are in conflict if they are both transversals on some 6-point set
    # AND the 6-point set has at most 2 transversals in our selection... no, that's circular.
    
    # Let me just use the original validity check approach but make it faster.
    # The key insight: when adding triple t={a,b,c}, we only need to check 6-point sets
    # that contain {a,b,c}. For each such 6-point set, we check if the other 7 transversals
    # are all present.
    
    # A 6-point set containing {a,b,c} is {a,b,c,x,y,z} where x,y,z are distinct and not in {a,b,c}.
    # For this to be a valid K(2,2,2) support, {a,b,c,x,y,z} must be partitionable into 3 pairs
    # such that {a,b,c} is a transversal. This means x,y,z must be matchable to a,b,c.
    # Specifically, we need a partition of {a,b,c,x,y,z} into 3 pairs where one element of each
    # pair is in {a,b,c} and the other is in {x,y,z}.
    
    # For a given {a,b,c} and {x,y,z}, the number of ways to partition into 3 pairs with one from
    # {a,b,c} and one from {x,y,z} is 3! = 6. For each such partition, the 8 transversals are determined.
    # {a,b,c} is one of them. We need to check if the other 7 are all in our set.
    
    # So for adding t={a,b,c}, we iterate over all choices of x,y,z (C(13,3) = 286 choices),
    # and for each, check if any of the 6 partitions has all 8 transversals present.
    
    # This is much faster than the original which iterated over a', b', c' independently.
    
    # Let me implement this correctly.
    pass

def _is_valid_addition_fast(triple, triple_set):
    """Check if adding triple to triple_set creates a K(2,2,2)."""
    a, b, c = triple
    # For each choice of 3 other points x, y, z
    for x in range(16):
        if x in (a, b, c):
            continue
        for y in range(16):
            if y in (a, b, c, x):
                continue
            for z in range(16):
                if z in (a, b, c, x, y):
                    continue
                # Now {a,b,c,x,y,z} is a 6-point set. Check all 6 partitions into 3 pairs
                # where one element is from {a,b,c} and one from {x,y,z}.
                # The 6 partitions correspond to the 6 bijections from {a,b,c} to {x,y,z}.
                for perm in itertools.permutations((x, y, z)):
                    # perm = (x', y', z') is a bijection from {a,b,c} to {x,y,z}
                    # Pairs: {a, perm[0]}, {b, perm[1]}, {c, perm[2]}
                    p0, p1, p2 = a, perm[0]
                    q0, q1 = b, perm[1]
                    r0, r1 = c, perm[2]
                    # The 8 transversals:
                    # (a,b,c), (a,b,r1), (a,q1,c), (a,q1,r1), (p1,b,c), (p1,b,r1), (p1,q1,c), (p1,q1,r1)
                    t1 = tuple(sorted((a, b, r1)))
                    t2 = tuple(sorted((a, q1, c)))
                    t3 = tuple(sorted((a, q1, r1)))
                    t4 = tuple(sorted((p1, b, c)))
                    t5 = tuple(sorted((p1, b, r1)))
                    t6 = tuple(sorted((p1, q1, c)))
                    t7 = tuple(sorted((p1, q1, r1)))
                    if (t1 in triple_set and t2 in triple_set and t3 in triple_set and
                        t4 in triple_set and t5 in triple_set and t6 in triple_set and t7 in triple_set):
                        return False
    return True

def _greedy_construct(deadline):
    """Greedy construction with randomization."""
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    random.shuffle(all_triples)
    
    triple_set = set()
    for triple in all_triples:
        if time.time() > deadline:
            break
        if _is_valid_addition_fast(triple, triple_set):
            triple_set.add(triple)
    return list(triple_set)

def _local_search(triples, deadline, max_iters=50):
    """Local search: try to add more triples."""
    triple_set = set(tuple(t) for t in triples)
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        added = False
        for t in all_triples:
            if t in triple_set:
                continue
            if _is_valid_addition_fast(t, triple_set):
                triple_set.add(t)
                added = True
                break
        if not added:
            break
    
    return list(triple_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple random greedy constructions
    best_result = []
    num_attempts = 10
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        random.seed(i * 42 + 123)
        result = _greedy_construct(deadline)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search on best result
    if time.time() < deadline:
        best_result = _local_search(best_result, deadline, max_iters=30)
    
    return best_result

# EVOLVE-BLOCK-END
