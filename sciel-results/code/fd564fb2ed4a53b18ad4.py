"""Triple systems on 16 points without K(2,2,2)."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _build_conflict_info():
    """
    Precompute for the complete 3-uniform hypergraph on 16 points:
    - all_triples: list of frozensets (or sorted tuples)
    - For each triple, the set of triples that conflict with it.
    
    Two triples conflict if together they can form a K(2,2,2) with some third triple.
    Actually, the conflict is: triple T1 and T2 are "close" to forming K(2,2,2) if there
    exist three disjoint pairs such that T1 and T2 are among the eight transversal triples.
    
    More precisely: K(2,2,2) on 6 vertices (3 parts of size 2) has 8 transversal triples.
    A set of triples contains K(2,2,2) if there exist 6 distinct vertices partitioned into
    3 pairs {a1,a2}, {b1,b2}, {c1,c2} such that all 8 triples {ai, bj, ck} are present.
    
    For two triples T1, T2 to be in conflict (i.e., cannot both be in the system without
    risking K(2,2,2)), we need to check if there exists a third triple T3 such that
    T1, T2, T3 are all transversals of some K(2,2,2). But actually, the constraint is on
    the whole set: no 6 vertices form a K(2,2,2) with all 8 transversals present.
    
    So the conflict graph approach: vertices are all triples. Two triples are adjacent if
    they are both transversals of the same K(2,2,2) structure on some 6 vertices. Then a
    valid system is an independent set in this conflict graph? No, that's not quite right.
    
    Actually, it's more subtle. Let me think again.
    
    The forbidden configuration: 6 distinct points p1,p2,q1,q2,r1,r2 such that all 8 triples
    {pi, qj, rk} are in the system.
    
    For the greedy independent set approach, I need a conflict relation. Two triples conflict
    if they cannot both be in a valid system. But this is not a pairwise property in general—
    it's a global constraint. However, we can use a sufficient condition: if two triples T1, T2
    are such that there exist three disjoint pairs where T1 and T2 are two of the eight transversals,
    then including both T1 and T2 doesn't immediately violate the constraint (we'd need all 8).
    
    So pairwise conflict isn't the right model. Let me use a different approach.
    
    Alternative: Use a greedy algorithm that, at each step, checks if adding a triple would
    complete a K(2,2,2). This is more expensive but correct.
    """
    n = 16
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
    
    return all_triples


def _check_k222(triple, existing):
    """
    Check if adding 'triple' to the set 'existing' creates a K(2,2,2).
    This is expensive, so we only check relevant cases.
    
    A K(2,2,2) involves 6 points. The new triple uses 3 points. The other 3 points
    must be in existing triples. We need to find 6 points partitioned into 3 pairs
    such that all 8 transversals are in existing ∪ {triple}.
    """
    a, b, c = triple
    # The 6 points must include a, b, c and 3 others.
    # The 3 pairs must be such that a, b, c are in different pairs.
    # So the other 3 points, say x, y, z, must be paired with a, b, c in some way.
    # There are 3! = 6 ways to pair {x,y,z} with {a,b,c}.
    # For each pairing, we check if all 8 transversals are present.
    
    # Get all other points
    other_points = [p for p in range(16) if p not in (a, b, c)]
    
    # Try all triples of other points
    for x, y, z in itertools.combinations(other_points, 3):
        # Try all 6 pairings of {x,y,z} with {a,b,c}
        for perm in itertools.permutations([x, y, z]):
            # perm[0] pairs with a, perm[1] with b, perm[2] with c
            p1, p2, p3 = perm
            # The 6 points are a, p1, b, p2, c, p3
            # The 8 transversals are:
            # a,b,c; a,b,p3; a,p2,c; a,p2,p3; p1,b,c; p1,b,p3; p1,p2,c; p1,p2,p3
            transversals = [
                (a, b, c),
                (a, b, p3),
                (a, p2, c),
                (a, p2, p3),
                (p1, b, c),
                (p1, b, p3),
                (p1, p2, c),
                (p1, p2, p3),
            ]
            # Check if all are in existing ∪ {triple}
            all_present = True
            for t in transversals:
                t_set = frozenset(t)
                if t_set != frozenset(triple):
                    if t_set not in existing:
                        all_present = False
                        break
            if all_present:
                return True
    return False


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    all_triples = _build_conflict_info()
    n_triples = len(all_triples)
    
    best_system = []
    best_size = 0
    
    # Multiple random restarts with greedy
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        random.shuffle(all_triples)
        
        system = []
        system_set = set()
        
        for t in all_triples:
            if time.time() > deadline - 2:
                break
            
            # Check if adding t creates a K(2,2,2)
            if not _check_k222(t, system_set):
                system.append(t)
                system_set.add(frozenset(t))
        
        if len(system) > best_size:
            best_size = len(system)
            best_system = system[:]
        
        # Early termination if we've found a large system
        if best_size >= 100:
            break
    
    # Convert to list of lists
    return [list(t) for t in best_system]

# EVOLVE-BLOCK-END
