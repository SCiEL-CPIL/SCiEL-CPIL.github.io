"""Rewritten program: precompute conflict hyperedges and use stochastic local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128

# Precompute pairwise gcds
gcd_table = [[0]*(N+1) for _ in range(N+1)]
for i in range(1, N+1):
    for j in range(i+1, N+1):
        g = gcd(i, j)
        gcd_table[i][j] = g
        gcd_table[j][i] = g

# Precompute all bad triples: (a,b,c) where gcd(a,b)==gcd(a,c)==gcd(b,c)
# We store for each element the set of pairs it forms bad triples with
# Actually, we need: for a candidate set S, adding x is valid if for all a,b in S,
# gcd(a,b) != gcd(a,x) or gcd(a,b) != gcd(b,x) or gcd(a,x) != gcd(b,x)
# i.e., not all three equal.

# Precompute for each pair (a,b), the gcd value
# For each x, and pair (a,b), check if gcd(a,b)==gcd(a,x)==gcd(b,x)

# We'll precompute a conflict structure:
# conflict[x][a] = set of b > a such that (a,b,x) is a bad triple
# But that's O(n^3) memory. Instead, for fast validity checking:
# can_add(x, S): for all pairs (a,b) in S, not (gcd(a,b)==gcd(a,x)==gcd(b,x))

# For efficiency, precompute for each x and each pair (a,b):
# bad_triple[x][a][b] = True if the triple (a,b,x) is bad
# This is n^3 = 128^3 = ~2M booleans, fine.

# Actually, let's precompute for each x, a set of pairs that would conflict
# conflicts_with_x[x] = set of (a,b) pairs such that adding x to a set containing a,b is bad

# But for local search, we need to quickly check if x can be added to current set.
# Let's precompute: for each x, for each a, the set of b values such that (a,b,x) is bad.
# Then can_add(x, S) checks: for all a in S, for all b in S where b != a, 
#   is b in bad_pairs[x][a]? 

# Better: precompute bad_pair_sets[x][a] = set of b such that gcd(a,b)==gcd(a,x)==gcd(b,x)
# Then when checking if x can be added to S:
#   for each a in S: for each b in S with b != a: if b in bad_pair_sets[x][a], fail

# This is O(|S|^2) per check. With |S| up to ~70, that's ~5000 checks per add attempt.
# For local search with many iterations, this might be slow.

# Alternative: precompute for each triple, and maintain a count of how many bad triples 
# are "active" in the current set. When we add/remove elements, update counts.

# Let me use a different approach: precompute all bad triples as frozensets of 3 elements.
# Then a set S is valid iff no bad triple is fully contained in S.

# Number of bad triples: at most C(128,3) ~ 340k. Let's count them.

bad_triples = set()
for a in range(1, N+1):
    for b in range(a+1, N+1):
        gab = gcd_table[a][b]
        for c in range(b+1, N+1):
            if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                bad_triples.add(frozenset((a, b, c)))

# Now, for each element x, precompute the set of pairs (a,b) such that {a,b,x} is a bad triple
# pair_conflicts[x] = set of frozenset({a,b}) for all bad triples containing x
# Actually, let's store: for each x, a dict mapping a -> set of b (with a < b) such that {a,b,x} bad

# For fast can_add: when checking if x can be added to S:
#   For each pair (a,b) in S with a<b, check if frozenset({a,b,x}) is in bad_triples
# This is O(|S|^2 * hash_lookup)

# Let me just precompute a lookup: for x, a, b -> is_bad_triple
# is_bad[x][a][b] for all x,a,b

# Actually simplest: keep bad_triples as a set of frozensets, and for can_add,
# check all pairs in S.

def can_add(x, S_set):
    """Check if x can be added to S_set (a set of ints) without forming a bad triple."""
    for a, b in combinations(S_set, 2):
        if frozenset((a, b, x)) in bad_triples:
            return False
    return True

def greedy_build(order):
    """Build a set greedily following the given order."""
    S = set()
    for x in order:
        if can_add(x, S):
            S.add(x)
    return S

def local_search(S, time_limit):
    """Local search to improve set S. Returns best set found."""
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start = time.time()
    
    # Precompute complement
    all_elements = set(range(1, N+1))
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try adding one element
        candidates = [x for x in all_elements if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                break
        
        if not improved:
            # Try swap: remove y, add x
            in_set = list(S)
            out_set = [x for x in all_elements if x not in S]
            random.shuffle(in_set)
            random.shuffle(out_set)
            
            for y in in_set:
                if time.time() - start > time_limit:
                    break
                for x in out_set:
                    if time.time() - start > time_limit:
                        break
                    # Try removing y and adding x
                    S.remove(y)
                    if can_add(x, S):
                        S.add(x)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                            no_improve = 0
                        else:
                            no_improve += 1
                        improved = True
                        break
                    else:
                        S.add(y)  # restore
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart with new random greedy
                order = list(range(1, N+1))
                random.shuffle(order)
                new_S = greedy_build(order)
                if len(new_S) > best_size:
                    best_size = len(new_S)
                    best_S = set(new_S)
                S = new_S
                no_improve = 0
    
    return best_S

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    
    random.seed(42)
    
    # Initial greedy with multiple random orders
    best_S = set()
    best_size = 0
    
    # Try several random greedy orders
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Local search from the best initial solution
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        best_S = local_search(best_S, remaining)
    
    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
