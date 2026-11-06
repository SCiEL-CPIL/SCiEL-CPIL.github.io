"""
Structural change: Start with the FULL universe [1..128] and iteratively REMOVE
elements that are part of "equal-gcd triples" until no such triples remain.
This is the dual of the greedy-add approach. Since the constraint forbids only
a sparse set of specific triples (those where all three pairwise gcds are equal),
starting from the maximum possible set (size 128) and pruning only the necessary
conflicts often yields larger valid sets than building up from scratch, because
it avoids the "greedy trap" where an early exclusion prevents many later inclusions.

Algorithm:
1. Precompute the set of all "bad" triples (a, b, c) such that gcd(a,b)==gcd(a,c)==gcd(b,c).
2. Represent the universe as a set of all numbers 1..128.
3. While there is a bad triple within the current set:
   a. Select one such triple.
   b. Identify which element in this triple is "least useful" or "most frequent" in other bad triples within the current set.
   c. Remove that element.
4. To improve further, try local search: can we re-add any removed elements?
5. Additionally, try different removal heuristics (random vs greedy) to find the best result.
"""

import time
import random
from math import gcd
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Leave 5s margin for the 90s kill switch
    N = 128
    UNIVERSE = list(range(1, N + 1))
    
    # Precompute GCDs
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Identify all bad triples
    # A triple (a, b, c) with a < b < c is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    # Build adjacency: for each element, list of bad triples containing it
    # This helps in deciding which element to remove
    triple_incidence = defaultdict(list)
    for idx, triple in enumerate(bad_triples):
        for v in triple:
            triple_incidence[v].append(idx)
            
    # Helper: Check if a set is valid
    def is_valid(S_set):
        for a, b, c in bad_triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True

    def prune_greedy(S_set, random_seed=None, strategy="least_used"):
        """
        Prune a set until it is valid.
        strategy: 'least_used' removes the element in the current violating triple that appears in the fewest other violating triples (or just remove any).
                  'random' picks a random violating triple and removes a random element from it.
        """
        S_set = set(S_set)
        if random_seed is not None:
            random.seed(random_seed)
        
        # We need to efficiently find if any bad triple is active.
        # Since N is small (128), we can just iterate.
        # To be efficient, we can maintain a count of how many bad triples are "active" for each pair or just check on the fly.
        
        while True:
            # Find one active bad triple
            active_triple = None
            for triple in bad_triples:
                if all(v in S_set for v in triple):
                    active_triple = triple
                    break
            
            if not active_triple:
                break
            
            # Decide which element to remove from active_triple
            a, b, c = active_triple
            
            if strategy == "random":
                victim = random.choice(active_triple)
            else:
                # Greedy: Count how many OTHER active bad triples each candidate is part of.
                # If we remove 'a', how many bad triples do we break?
                # Actually, we just want to remove the element that is "least valuable" to keep.
                # An element that is in many bad triples is a "hub" of conflicts. Removing it might fix many triples at once.
                # So we should remove the one that is in the MOST active bad triples?
                # Let's calculate the degree of each vertex in the subgraph of active bad triples.
                
                counts = {a: 0, b: 0, c: 0}
                # Iterate through all bad triples again to count? O(M) per step is expensive if M is large.
                # M is ~ (128^3)/6 ~ 300k? No, much less because gcds must match.
                # Let's just count occurrences in the full bad_triples list that are active.
                
                # Optimization: Precompute for each triple the list of triples that share elements?
                # For now, simple approach:
                for other_triple in bad_triples:
                    if all(v in S_set for v in other_triple):
                        if a in other_triple:
                            counts[a] += 1
                        if b in other_triple:
                            counts[b] += 1
                        if c in other_triple:
                            counts[c] += 1
                            
                # Remove the one with the highest count (most central)
                victim = max(counts, key=counts.get)
                
            S_set.discard(victim)
            
            # Check time
            if time.time() > DEADLINE - 2:
                break
                
        return S_set

    def local_search_add(S_set, random_seed=0):
        """
        Try to re-add elements that were removed, if possible.
        """
        random.seed(random_seed)
        S_set = set(S_set)
        all_elements = set(range(1, N + 1))
        removed = all_elements - S_set
        
        # Sort removed elements by some heuristic? Maybe by how "isolated" they are?
        # For now, just try them in random order or sorted order.
        # To maximize size, we want to add as many as possible.
        # We can't just add one by one greedily because adding one might block another.
        # But since we started from full and pruned, the set is "saturated" in a sense.
        # Let's try to add any element x such that no bad triple containing x is fully active in S_set U {x}.
        
        candidates = list(removed)
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            can_add = True
            # A bad triple is formed if there exist a, b in S_set such that (x, a, b) is a bad triple.
            # We need to check all bad triples containing x.
            for triple in bad_triples:
                if x not in triple:
                    continue
                # Check if the other two are in S_set
                others = [v for v in triple if v != x]
                if len(others) == 2:
                    if others[0] in S_set and others[1] in S_set:
                        can_add = False
                        break
            
            if can_add:
                S_set.add(x)
                
            if time.time() > DEADLINE - 2:
                break
                
        return S_set

    # Strategy 1: Greedy Pruning with "Most Central" removal
    best_solution = set()
    
    # We can also try a "Randomized Greedy Pruning" multiple times to escape local optima.
    # The pruning process is deterministic if we always pick the "most central" element,
    # but the order of finding the first active triple can vary if we shuffle the bad_triples list?
    # Actually, the iteration order of bad_triples is fixed.
    # To diversify, we can shuffle the list of bad_triples before pruning.
    
    # Let's create a few shuffled lists of bad triples to induce different pruning paths.
    results = []
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
            
        # Create a shuffled copy of bad triples to change the order in which conflicts are addressed
        shuffled_triples = bad_triples[:]
        random.seed(i)
        random.shuffle(shuffled_triples)
        
        # We need to modify prune_greedy to accept the list of triples to check
        # Let's write a specific pruning function for this run
        
        S_set = set(UNIVERSE)
        
        while True:
            active_triple = None
            for triple in shuffled_triples:
                if all(v in S_set for v in triple):
                    active_triple = triple
                    break
            
            if not active_triple:
                break
            
            a, b, c = active_triple
            
            # Calculate "centrality" in the current active subgraph
            counts = {a: 0, b: 0, c: 0}
            for other_triple in shuffled_triples:
                if all(v in S_set for v in other_triple):
                    if a in other_triple:
                        counts[a] += 1
                    if b in other_triple:
                        counts[b] += 1
                    if c in other_triple:
                        counts[c] += 1
                        
            # Remove the one with max count
            # Tie-breaker: random
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            victim = random.choice(candidates)
            
            S_set.discard(victim)
            
            if time.time() > DEADLINE - 3:
                break
                
        # After pruning, try to add back elements
        S_set = local_search_add(S_set, random_seed=i + 100)
        
        if len(S_set) > len(best_solution):
            best_solution = S_set
            results.append((len(S_set), i))
            
    # If best_solution is empty (shouldn't happen), fallback
    if not best_solution:
        best_solution = {1}
        
    return list(best_solution)

# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
