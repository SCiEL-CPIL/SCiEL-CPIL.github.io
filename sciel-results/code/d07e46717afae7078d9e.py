"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

# Precompute all forbidden triples: triples (a,b,c) with a<b<c from 1..128
# such that a+b, a+c, b+c are all in 1..128.
# A valid set S must not contain all three of any such triple.
# We model this as a 3-uniform hypergraph and find a large independent set.

def _precompute_forbidden_triples():
    """Return list of (a,b,c) triples where a<b<c in 1..128 and
    a+b, a+c, b+c are all in 1..128."""
    triples = []
    for a in range(1, 129):
        for b in range(a+1, 129):
            ab = a + b
            if ab > 128:
                break
            for c in range(b+1, 129):
                ac = a + c
                if ac > 128:
                    break
                bc = b + c
                if bc <= 128:
                    triples.append((a, b, c))
    return triples

def _precompute_conflict_graph():
    """For each vertex v in 1..128, compute the set of vertices u such that
    there exists a forbidden triple containing both v and u.
    This gives us a graph where edges indicate 'conflict potential'.
    But actually for greedy MIS on 3-uniform hypergraph, we need:
    For each vertex v, the set of other vertices that form a forbidden triple
    with v and some third vertex.
    
    Actually, let's think differently. For greedy construction:
    When considering adding vertex x, we need to check if there exist a,b
    already in S such that {x,a,b} is a forbidden triple.
    
    Better approach: precompute for each pair (x,y) the set of z such that
    {x,y,z} is a forbidden triple. Then when building S, for each new x,
    check all pairs (a,b) in S: if there's a forbidden triple {x,a,b}, reject.
    
    Even better: for each vertex x, precompute the set of 'forbidden pairs'
    (a,b) such that {x,a,b} is a forbidden triple. Then x is addable to S
    iff no pair in S is in x's forbidden pair set.
    
    Let's use: for each x, forbid_pairs[x] = set of (a,b) with a<b, a!=x, b!=x
    such that {x,a,b} is a forbidden triple.
    """
    triples = _precompute_forbidden_triples()
    # For each vertex, store the set of pairs that would form a forbidden triple with it
    forbid_pairs = [set() for _ in range(129)]  # index 1..128
    for (a, b, c) in triples:
        # {a,b,c} is forbidden
        # For vertex a: pair (b,c) is forbidden
        # For vertex b: pair (a,c) is forbidden  
        # For vertex c: pair (a,b) is forbidden
        forbid_pairs[a].add((b, c))
        forbid_pairs[b].add((a, c))
        forbid_pairs[c].add((a, b))
    return forbid_pairs, triples

def _is_valid_hypergraph(S, forbid_pairs):
    """Check if S is a valid independent set in the 3-uniform hypergraph."""
    Sset = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            # Check if there exists c in S such that {a,b,c} is a forbidden triple
            # i.e., (a,b) is in forbid_pairs[c] for some c in S, c != a, c != b
            # Equivalent: check if (b,c) in forbid_pairs[a] for some c in S, c != a, c != b
            # OR (a,c) in forbid_pairs[b] for some c in S, c != a, c != b
            # Simplest: for each c in S, c != a, c != b, check if (a,b) in forbid_pairs[c]
            for k in range(n):
                c = S[k]
                if c != a and c != b:
                    if (a, b) in forbid_pairs[c] or (b, a) in forbid_pairs[c]:
                        return False
                    if (a, c) in forbid_pairs[b] or (c, a) in forbid_pairs[b]:
                        return False
    return True

def _greedy_randomized(forbid_pairs, deadline, seed=42):
    """Randomized greedy maximal independent set on the 3-uniform hypergraph.
    
    Strategy: repeatedly pick a random unselected vertex that can be added
    without creating a forbidden triple. To make it efficient, we maintain
    the current set and for each candidate check if it conflicts.
    
    For efficiency: when considering vertex x, it conflicts with current S
    if there exist a,b in S such that {x,a,b} is a forbidden triple.
    This is equivalent to: there exists a pair (a,b) in S such that 
    (a,b) in forbid_pairs[x] (with a<b, a!=x, b!=x).
    """
    rng = random.Random(seed)
    S = []
    Sset = set()
    # For fast checking: maintain a set of all pairs present in S
    # Actually, for each x, we need to check if any pair from forbid_pairs[x] is in S.
    # Let's maintain S as a set and check each pair in forbid_pairs[x].
    
    candidates = list(range(1, 129))
    rng.shuffle(candidates)
    
    while candidates and time.time() < deadline:
        # Pick a random candidate from the remaining
        # But to be smarter, we could try multiple random ones
        x = candidates.pop()
        if x in Sset:
            continue
        # Check if x can be added: no pair (a,b) in S such that (a,b) in forbid_pairs[x]
        can_add = True
        fp = forbid_pairs[x]
        if fp:
            # Check if any pair in fp is a subset of S
            # fp contains pairs (a,b) where a<b
            for (a, b) in fp:
                if a in Sset and b in Sset:
                    can_add = False
                    break
        if can_add:
            Sset.add(x)
            S.append(x)
    
    return S

def _greedy_with_removal(forbid_pairs, deadline, seed=42):
    """Start with a large initial set, then remove vertices to resolve conflicts.
    This can sometimes yield larger sets than pure greedy."""
    rng = random.Random(seed)
    
    # Start with all 128 vertices
    S = set(range(1, 129))
    
    # Repeatedly find a forbidden triple and remove one of its vertices
    # Randomize which vertex to remove
    for iteration in range(1000):
        if time.time() > deadline:
            break
        # Find a forbidden triple in current S
        found = False
        Slist = list(S)
        rng.shuffle(Slist)
        for i in range(len(Slist)):
            a = Slist[i]
            for j in range(i+1, len(Slist)):
                b = Slist[j]
                # Check if there's c in S such that {a,b,c} is forbidden
                # i.e., (a,b) in forbid_pairs[c] for some c in S
                for c in S:
                    if c != a and c != b:
                        if (a, b) in forbid_pairs[c] or (b, a) in forbid_pairs[c]:
                            # Remove one of a, b, c randomly
                            victim = rng.choice([a, b, c])
                            S.discard(victim)
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            break
    
    return list(S)

# Precompute at module level for speed
_FORBID_PAIRS, _TRIPLES = _precompute_conflict_graph()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_S = None
    best_size = 0
    
    # Known valid seed
    seed_S = [1, 62] + list(range(64, 129))
    best_S = list(seed_S)
    best_size = len(seed_S)
    
    # Try multiple randomized greedy approaches
    # Approach 1: Randomized greedy MIS with different seeds
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        S = _greedy_randomized(_FORBID_PAIRS, deadline - 5.0, seed=trial)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Approach 2: Greedy with removal (start full, remove conflicts)
    for trial in range(50):
        if time.time() > deadline - 3.0:
            break
        S = _greedy_with_removal(_FORBID_PAIRS, deadline - 3.0, seed=trial + 1000)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Final validation
    if best_S is None:
        return [1, 62] + list(range(64, 129))
    
    # Validate the best set
    Sset = set(best_S)
    valid = True
    n = len(best_S)
    for i in range(n):
        a = best_S[i]
        for j in range(i+1, n):
            b = best_S[j]
            for k in range(j+1, n):
                c = best_S[k]
                if (a+b) in Sset and (a+c) in Sset and (b+c) in Sset:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known valid seed
        return [1, 62] + list(range(64, 129))
    
    return [int(x) for x in sorted(best_S)]

# EVOLVE-BLOCK-END
