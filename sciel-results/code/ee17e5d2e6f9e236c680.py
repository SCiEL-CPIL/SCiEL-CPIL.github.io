import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # 1. Precomputation
    # All 560 triples
    all_triples = list(combinations(range(16), 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflict graph
    # Two triples T1 and T2 conflict if they belong to the same K(2,2,2).
    # T1=(a,b,c), T2=(d,e,f).
    # They are in the same K(2,2,2) if there exist a', b', c' such that
    # the set of 8 transversals of pairs (x1,x2), (y1,y2), (z1,z2) contains both T1 and T2.
    # This implies that the union of points in T1 and T2 must be subset of the 6 points of the K(2,2,2).
    # Let U = set(T1) U set(T2).
    # If len(U) > 6, they cannot be in the same K(2,2,2).
    # If len(U) == 6, they are in a K(2,2,2) if and only if they are in the same part of the partition?
    # No, in a K(2,2,2), any two distinct transversals differ in exactly one coordinate.
    # So T1 and T2 must differ in exactly one point?
    # Let's check.
    # K(2,2,2) transversals:
    # (x1, y1, z1)
    # (x1, y1, z2) -> differs in 1
    # (x1, y2, z1) -> differs in 1
    # (x1, y2, z2) -> differs in 2
    # (x2, y1, z1) -> differs in 1
    # (x2, y1, z2) -> differs in 2
    # (x2, y2, z1) -> differs in 2
    # (x2, y2, z2) -> differs in 3
    #
    # So T1 and T2 are in the same K(2,2,2) if and only if:
    # 1. The symmetric difference of their point sets has size 2 (they share 1 point, differ in 2).
    #    OR
    # 2. The symmetric difference has size 4 (they share 0 points? No, max size 3. So they must be disjoint?
    #    If they are disjoint, |U|=6. Do they form a K(2,2,2)?
    #    Let T1=(x1,y1,z1), T2=(x2,y2,z2).
    #    The other 6 are (x1,y1,z2), (x1,y2,z1), (x2,y1,z1), (x2,y1,z2), (x2,y2,z1).
    #    Yes, any two disjoint triples on 6 distinct points define a unique K(2,2,2) if we assume the pairing is consistent?
    #    Wait, the pairing is determined by the points.
    #    If T1=(a,b,c) and T2=(d,e,f) are disjoint.
    #    The 6 points are {a,b,c,d,e,f}.
    #    For them to be in a K(2,2,2), there must be a partition into 3 pairs such that a and d are in the same pair?
    #    No, a and d are just two points.
    #    Actually, ANY two disjoint triples on 6 distinct points are part of exactly ONE K(2,2,2)?
    #    Let's verify.
    #    Points: a,b,c,d,e,f.
    #    T1=(a,b,c), T2=(d,e,f).
    #    For a K(2,2,2), we need 3 pairs.
    #    One pair must contain a. Let's say (a, ?).
    #    If the pair is (a,d), then the other pairs must partition {b,c,e,f} into (b,e) and (c,f) OR (b,f) and (c,e).
    #    Case 1: Pairs (a,d), (b,e), (c,f).
    #    Transversals include (a,b,c) and (d,e,f). Yes.
    #    Case 2: Pairs (a,d), (b,f), (c,e).
    #    Transversals include (a,b,c) and (d,f,e)=(d,e,f). Yes.
    #    So actually, disjoint triples T1, T2 are in MULTIPLE K(2,2,2)s?
    #    Wait, the K(2,2,2) is defined by the SET of 8 triples.
    #    If we have pairs (a,d), (b,e), (c,f), the 8 triples are fixed.
    #    If we have pairs (a,d), (b,f), (c,e), the 8 triples are different.
    #    So T1 and T2 are in different K(2,2,2)s.
    #    But the condition is "no three disjoint pairs have all eight transversal triples present".
    #    This means we cannot have a configuration of 6 points and a pairing such that all 8 are present.
    #    So T1 and T2 "conflict" if there EXISTS a pairing of the 6 points {a..f} such that both T1 and T2 are in the set of 8 transversals.
    #    As shown, if T1, T2 are disjoint, there are multiple pairings.
    #    Specifically, if T1=(a,b,c), T2=(d,e,f).
    #    Pairing 1: (a,d), (b,e), (c,f) -> T1, T2 in K1.
    #    Pairing 2: (a,e), (b,d), (c,f) -> T1 in K2? (a,b,c) is in K2. Is T2=(d,e,f) in K2?
    #    K2 transversals: (a,b,c), (a,b,f), (a,e,c), (a,e,f), (d,b,c), (d,b,f), (d,e,c), (d,e,f).
    #    Yes, T2 is in K2.
    #    So any two disjoint triples on 6 distinct points are in at least one common K(2,2,2).
    #    In fact, they are in 3 distinct K(2,2,2)s?
    #    Let's check (a,d), (b,f), (c,e).
    #    Transversals: (a,b,c), (a,b,e), (a,f,c), (a,f,e), (d,b,c), (d,b,e), (d,f,c), (d,f,e).
    #    T2=(d,e,f) is NOT in this set. (d,f,e) is (d,e,f). Wait, (d,f,e) IS T2.
    #    So yes, they are in 3 K(2,2,2)s.
    #
    # What if T1, T2 share 1 point?
    # T1=(a,b,c), T2=(a,d,e). U={a,b,c,d,e}, size 5.
    # Need a 6th point f.
    # Pairs must cover {a,b,c,d,e,f}.
    # a is in one pair, say (a, ?).
    # If (a,f), then b,c,d,e must be paired.
    # T1 has b,c. T2 has d,e.
    # For T1 to be in K, b and c must be in different pairs? No, T1 takes one from each pair.
    # So b is in pair (b, ?), c is in pair (c, ?).
    # T2 takes d, e. So d is in pair (d, ?), e is in pair (e, ?).
    # The pairs are (a,f), (b, ?), (c, ?), (d, ?), (e, ?).
    # We have 5 pairs? No, 3 pairs.
    # So b must be paired with d or e?
    # If (b,d), then (c,e).
    # Pairs: (a,f), (b,d), (c,e).
    # Transversals include (a,b,c) and (a,d,e). Yes.
    # So if they share 1 point, they conflict if there exists a 6th point f such that we can pair the remaining 4 points into 2 pairs?
    # Yes, always possible for any f.
    #
    # What if T1, T2 share 2 points?
    # T1=(a,b,c), T2=(a,b,d). U={a,b,c,d}, size 4.
    # Need 2 more points e, f.
    # Pairs: (a,?), (b,?), (c,?), (d,?).
    # T1 takes a,b,c. T2 takes a,b,d.
    # a is in pair (a, x). T1 takes a, T2 takes a. OK.
    # b is in pair (b, y). T1 takes b, T2 takes b. OK.
    # c is in pair (c, z). T1 takes c.
    # d is in pair (d, w). T2 takes d.
    # We have points c, d, e, f.
    # Pairs must be (c,e) and (d,f) OR (c,f) and (d,e).
    # If (c,e), (d,f):
    # Transversals: (a,b,c), (a,b,f), (a,e,c), (a,e,f), (x,b,c), (x,b,f), (x,e,c), (x,e,f).
    # T1=(a,b,c) is present.
    # T2=(a,b,d) is NOT present (d is not in the transversals? Wait, d is in pair (d,f), so d is a valid choice.
    # The transversals are combinations of one from each pair.
    # Pair 1: (a,x). Choices a, x.
    # Pair 2: (b,y). Choices b, y.
    # Pair 3: (c,e), (d,f) -> This is TWO pairs? No, we only have 3 pairs total.
    # We have points a,b,c,d,e,f.
    # Pairs: (a,x), (b,y), (c,z).
    # If x=e, y=f, z=d?
    # Pairs: (a,e), (b,f), (c,d).
    # Transversals: (a,b,c), (a,b,d), (a,f,c), (a,f,d), (e,b,c), (e,b,d), (e,f,c), (e,f,d).
    # T1=(a,b,c) present.
    # T2=(a,b,d) present.
    # So yes, they conflict.
    
    # Conclusion: Two triples conflict if their union has size <= 6.
    # Since max size of union of two triples is 6, ALL pairs of triples conflict?
    # Wait.
    # If T1=(0,1,2), T2=(3,4,5). Union size 6. Conflict.
    # If T1=(0,1,2), T2=(0,3,4). Union size 5. Conflict.
    # If T1=(0,1,2), T2=(0,1,3). Union size 4. Conflict.
    # If T1=(0,1,2), T2=(0,1,2). Same triple. Not an edge in simple graph, but we don't add same triple twice.
    #
    # So the conflict graph is the COMPLEMENT of the graph where edges connect triples with union size > 6?
    # But union size is ALWAYS <= 6.
    # So EVERY pair of distinct triples conflicts?
    # That would mean the maximum independent set is 1.
    # But we know we can have more.
    # Where is the error?
    
    # Re-read definition: "no three disjoint pairs have all eight transversal triples present".
    # This is a constraint on the SET of triples.
    # It does NOT mean that any two triples conflict.
    # It means we cannot have a specific SET of 8 triples.
    # So the "conflict" is not pairwise. It's an 8-way constraint (clause).
    # Maximum Independent Set on a graph is for 2-way constraints (clauses of size 2).
    # Here we have clauses of size 8.
    # So we are looking for a Maximum Satisfiable Set (MaxSAT) or Maximum Independent Set in a Hypergraph (3-uniform? No, 8-uniform hypergraph of clauses).
    # Actually, it's a Maximum Independent Set in a 3-uniform hypergraph? No.
    # The "variables" are triples. The "constraints" are K(2,2,2)s.
    # We want to maximize variables subject to no constraint being fully satisfied.
    # This is the "Maximum Independent Set in a Hypergraph" problem, where the hyperedges are the K(2,2,2)s.
    # A hyperedge is a set of 8 triples.
    # We want a set of triples that intersects every hyperedge in at most 7 elements?
    # No, "no hyperedge is fully contained in our set".
    # This is equivalent to finding a Maximum Set of Vertices that is NOT a hitting set for the complements?
    # It's just Max Independent Set in a Hypergraph where independence means no hyperedge is fully included.
    
    # The "Largest Degree First" greedy on a graph assumes pairwise conflicts.
    # If we treat it as a graph where edges connect triples that are in a common K(2,2,2), we are solving a RELAXED problem?
    # No, if T1 and T2 are in a common K, adding both doesn't violate the constraint immediately (need 8).
    # So they are NOT conflicting in the pairwise sense.
    # The "Largest Degree First" strategy mentioned in the directive might be a misinterpretation of the problem structure by the inspiration source, OR it refers to a different representation.
    
    # However, the directive says: "Model the problem as a maximum independent set in a graph where vertices are triples and edges connect conflicting triples... leveraging the fact that the conflict graph is sparse".
    # If the conflict graph is sparse, it implies that most pairs do NOT conflict.
    # But I just deduced that most pairs DO share a K(2,2,2).
    # Let's re-evaluate "Sparse".
    # Maybe "conflicting" means "cannot be in the same valid system"?
    # No, two triples can be in the same system as long as the 8th isn't there.
    #
    # Perhaps the directive implies a different conflict definition?
    # Or maybe I should just implement the greedy algorithm on the 8-way constraints directly?
    # "Largest Degree First" in a hypergraph context:
    # Degree of a vertex (triple) = number of hyperedges (K(2,2,2)s) it belongs to.
    # All triples belong to 1716 hyperedges. So degrees are equal.
    # So "Largest Degree" is trivial.
    #
    # Let's try a different structural change:
    # Instead of local search on a random start, use a **Greedy Construction** that adds triples in a specific order (e.g., random permutation) and accepts if it doesn't complete any K(2,2,2).
    # This is the standard greedy for Max Independent Set in Hypergraphs.
    # To improve, we can use **Randomized Greedy with Restarts**.
    # This is structurally different from the current local search (which starts random and flips).
    # The current program already has a `greedy_construct` stub but doesn't use it effectively.
    # I will implement a robust Randomized Greedy with many restarts.
    #
    # Optimization:
    # Precompute for each triple the list of K(2,2,2)s it belongs to (1716 lists of 7 indices).
    # Maintain a counter for each K(2,2,2): how many of its 8 triples are currently in the set.
    # Initially all 0.
    # When adding triple T:
    #   For each K in T's list:
    #     Increment count[K].
    #     If count[K] == 8: Violation!
    #   If violation, don't add T (or backtrack? No, greedy doesn't backtrack).
    #   If no violation, add T.
    #
    # This is O(1716) per candidate.
    # 560 candidates * 1716 ops = ~1M ops per restart.
    # 100 restarts = 100M ops.
    # In Python, 100M simple ops might take 5-10 seconds.
    # This fits well within 135s.
    #
    # To maximize, we can try different random permutations.
    #
    # Let's implement this.

    # Precompute hyperedges (conflict sets) for each triple
    # conflict_cache[t_idx] = list of lists of 7 indices (the other triples in the K(2,2,2))
    # We also need to map each K(2,2,2) to a unique ID to maintain counts?
    # Or just maintain a count array indexed by... what?
    # There are 120,120 K(2,2,2)s.
    # We can map each K to an index 0..120119.
    # Then `k_counts` is an array of size 120120.
    # `triple_to_ks` is a list of lists of K-indices for each triple.
    
    # Generation of K(2,2,2)s:
    # Iterate 6 points. 15 pairings.
    # This gives 120,120 unique Ks.
    
    k_indices = []
    triple_to_ks = [[] for _ in range(n_triples)]
    k_to_triples = [] # List of 8 triple indices for each K
    
    # Map 6-point sets and pairings to unique Ks?
    # Actually, we can just generate them and assign IDs.
    
    k_id = 0
    # To ensure uniqueness, we can use a set of frozensets of the 8 triples?
    # Or just rely on the generation being unique?
    # Each K is defined by 6 points and a pairing.
    # Different 6-point sets -> different Ks.
    # Same 6-point set, different pairing -> different Ks (different sets of 8 triples).
    # So yes, unique.
    
    # Iterate over combinations of 6 points
    for pts in combinations(range(16), 6):
        # 15 pairings of 6 items
        # We can generate pairings by fixing the first point's partner, then the next, etc.
        # p0, p1, p2, p3, p4, p5
        # Pair p0 with p1, p2, p3, p4, p5 (5 choices)
        # Then pair remaining 4.
        # If p0-p1: remaining p2,p3,p4,p5.
        #   Pair p2 with p3, p4, p5 (3 choices).
        #   Then p4-p5 fixed.
        #   So 3 pairings for this branch.
        # Total 5 * 3 = 15.
        
        # Generate pairings
        pairings = []
        p0, p1, p2, p3, p4, p5 = pts
        
        # Helper to add a pairing
        def add_pairing(pairs):
            nonlocal k_id
            # Generate 8 triples
            triples_in_k = []
            # pairs is list of 3 tuples (x, x')
            (x1, x2), (y1, y2), (z1, z2) = pairs
            # 8 combinations
            for x in (x1, x2):
                for y in (y1, y2):
                    for z in (z1, z2):
                        t = tuple(sorted((x, y, z)))
                        triples_in_k.append(t)
            
            # Convert to indices
            idxs = [triple_to_idx[t] for t in triples_in_k]
            
            # Assign ID
            current_k_id = k_id
            k_id += 1
            
            # Store
            k_to_triples.append(idxs)
            
            # Update triple_to_ks
            for idx in idxs:
                triple_to_ks[idx].append(current_k_id)
        
        # Branch 1: p0-p1
        add_pairing([(p0, p1), (p2, p3), (p4, p5)])
        add_pairing([(p0, p1), (p2, p4), (p3, p5)])
        add_pairing([(p0, p1), (p2, p5), (p3, p4)])
        
        # Branch 2: p0-p2
        add_pairing([(p0, p2), (p1, p3), (p4, p5)])
        add_pairing([(p0, p2), (p1, p4), (p3, p5)])
        add_pairing([(p0, p2), (p1, p5), (p3, p4)])
        
        # Branch 3: p0-p3
        add_pairing([(p0, p3), (p1, p2), (p4, p5)])
        add_pairing([(p0, p3), (p1, p4), (p2, p5)])
        add_pairing([(p0, p3), (p1, p5), (p2, p4)])
        
        # Branch 4: p0-p4
        add_pairing([(p0, p4), (p1, p2), (p3, p5)])
        add_pairing([(p0, p4), (p1, p3), (p2, p5)])
        add_pairing([(p0, p4), (p1, p5), (p2, p3)])
        
        # Branch 5: p0-p5
        add_pairing([(p0, p5), (p1, p2), (p3, p4)])
        add_pairing([(p0, p5), (p1, p3), (p2, p4)])
        add_pairing([(p0, p5), (p1, p4), (p2, p3)])

    n_ks = k_id
    k_counts = [0] * n_ks
    
    def greedy_random():
        random.shuffle(all_triples)
        active = [False] * n_triples
        current_k_counts = k_counts[:] # Copy
        count = 0
        
        for t in all_triples:
            idx = triple_to_idx[t]
            # Check if adding t causes any K to reach 8
            # Iterate through Ks containing t
            safe = True
            for k_idx in triple_to_ks[idx]:
                if current_k_counts[k_idx] == 7:
                    safe = False
                    break
            
            if safe:
                active[idx] = True
                count += 1
                for k_idx in triple_to_ks[idx]:
                    current_k_counts[k_idx] += 1
            else:
                # Optional: try to remove something? No, pure greedy.
                pass
                
        return active, count

    best_active = None
    best_count = 0
    
    # Run many restarts
    restarts = 0
    while time.time() < deadline - 5:
        active, count = greedy_random()
        if count > best_count:
            best_count = count
            best_active = active
            restarts += 1
            # If we find a very good one, maybe slow down? No, keep going.
        
        # Check time
        if time.time() > deadline - 2:
            break
            
    if best_active is None:
        return [[0, 1, 2]]
        
    result = []
    for i, is_active in enumerate(best_active):
        if is_active:
            result.append(list(all_triples[i]))
            
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
