"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

DEADLINE = time.time() + 85  # safety margin before 90s kill

def _solve():
    N = 128
    # 1. Precompute GCD matrix
    # G[i][j] is gcd(i+1, j+1)
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]

    # 2. Identify all "bad triples" (a, b, c) with a < b < c such that
    #    gcd(a,b) == gcd(a,c) == gcd(b,c)
    #    If a triple is bad, it creates a "conflict" among its members.
    #    Specifically, if we include any two members of a bad triple, the third
    #    cannot be included.
    #    This is equivalent to saying: in the set of chosen elements, no three
    #    can form a bad triple.
    
    # Wait, the condition is: "no three distinct elements whose three pairwise greatest common divisors are equal".
    # This means the set S is valid if for ALL distinct a,b,c in S, it is NOT the case that gcd(a,b)==gcd(a,c)==gcd(b,c).
    
    # Let's re-verify the structure.
    # If we have a bad triple (a,b,c), then we cannot have all three in S.
    # This is a 3-uniform hypergraph independent set problem.
    # However, note that if gcd(a,b) == gcd(a,c) == gcd(b,c) = g, then a, b, c are all multiples of g.
    # Let a=ga', b=gb', c=gc'.
    # gcd(ga', gb') = g => gcd(a', b') = 1
    # gcd(ga', gc') = g => gcd(a', c') = 1
    # gcd(gb', gc') = g => gcd(b', c') = 1
    # So a', b', c' are pairwise coprime.
    
    # Strategy:
    # We want to find the maximum size subset S of [1, 128] such that no three elements form such a triple.
    
    # Let's use a different approach: Branch and Bound on the elements 1..128.
    # But 2^128 is too big.
    # However, the constraints are local.
    
    # Let's look at the structure of bad triples again.
    # If g > 1, then a,b,c are multiples of g.
    # If g = 1, then a,b,c are pairwise coprime.
    # Are there bad triples with g=1?
    # If gcd(a,b)=1, gcd(a,c)=1, gcd(b,c)=1. Yes, e.g., (2,3,5).
    # So (2,3,5) is a bad triple. We cannot have all three.
    # (2,3,7) is a bad triple.
    # (2,3,11) is a bad triple.
    # This implies that if we pick 2 and 3, we cannot pick ANY prime > 3.
    # Actually, if we pick 2 and 3, we cannot pick any x such that gcd(2,x)=1 and gcd(3,x)=1.
    # i.e., x must not be coprime to 6.
    # So if 2,3 in S, then for all x in S, x must share a factor with 2 or 3.
    
    # This looks like a complex constraint satisfaction.
    
    # Let's try a Greedy + Local Search approach but optimized.
    # Or better: Use the fact that the "conflict" is a 3-clique in a certain graph?
    # No, it's a 3-uniform hypergraph.
    
    # However, there is a known result for f_3(n).
    # For r=3, the extremal function is related to sets with no 3 elements having equal pairwise gcds.
    
    # Let's try a heuristic that is very fast:
    # 1. Start with a large set.
    # 2. Remove elements that are part of many bad triples.
    
    # Let's precompute the "badness" of each element.
    # For each element x, count how many pairs (a,b) such that (a,b,x) is a bad triple.
    # If we remove x, we eliminate all those triples.
    
    # Let's compute for each x, the set of pairs (a,b) that form a bad triple with x.
    # This is still O(N^3) to compute fully, which is 128^3 ~ 2 million. This is fast in Python if optimized?
    # 2 million operations is trivial.
    
    bad_triples = []
    # a < b < c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                g_ac = G[a][c]
                if g_ac != g_ab:
                    continue
                g_bc = G[b][c]
                if g_bc == g_ab:
                    bad_triples.append((a, b, c))
    
    # Now we have a list of bad triples.
    # We want to find the maximum subset S such that no triple in bad_triples is fully contained in S.
    # This is the Maximum 3-SAT Independent Set or similar.
    # Equivalently, we want to remove the minimum number of elements to hit all bad triples.
    # This is the Hitting Set problem, which is NP-hard.
    # But N=128 is small enough for a greedy hitting set or local search to work well.
    
    # Greedy Hitting Set:
    # 1. Start with S = {1..128}.
    # 2. While there is a bad triple fully in S:
    #      Pick the element in the triple that is in the most other bad triples (within S).
    #      Remove it from S.
    
    # Let's implement this.
    
    # Precompute for each element, the list of triples it belongs to.
    triples_by_elem = {i: [] for i in range(1, N + 1)}
    for t in bad_triples:
        for x in t:
            triples_by_elem[x].append(t)
    
    S = set(range(1, N + 1))
    
    # To speed up, we can maintain a count of "active" triples for each element.
    # A triple is active if all 3 elements are in S.
    
    # Initial active triples
    active_triples = set(bad_triples)
    
    # For each element, count how many active triples it is in.
    active_count = {i: 0 for i in range(1, N + 1)}
    for t in active_triples:
        for x in t:
            active_count[x] += 1
    
    while active_triples:
        if time.time() > DEADLINE:
            break
        
        # Find an active triple
        # To be greedy, pick the triple with the element that has the highest active_count?
        # Or pick any triple, and remove the element with highest active_count in that triple?
        
        # Let's pick the element with the highest active_count globally.
        # If we remove the most "connected" element, we break the most triples.
        
        # Find max active_count
        max_count = -1
        best_elem = -1
        for x in S:
            if active_count[x] > max_count:
                max_count = active_count[x]
                best_elem = x
        
        if best_elem == -1 or max_count == 0:
            # No active triples? Or no elements?
            # If active_triples is not empty, there must be some element with count > 0.
            # Unless active_triples contains triples with elements not in S?
            # We must ensure active_triples only contains triples fully in S.
            
            # Let's just pick any element from the first active triple.
            t = next(iter(active_triples))
            for x in t:
                if x in S:
                    best_elem = x
                    break
        
        if best_elem == -1:
            break
            
        # Remove best_elem from S
        S.remove(best_elem)
        
        # Update active_triples and active_count
        # All triples containing best_elem are no longer active.
        # Iterate through triples_by_elem[best_elem]
        for t in triples_by_elem[best_elem]:
            # Check if t is still active (i.e., all other 2 elements are in S)
            # Actually, if we just removed best_elem, t is definitely not active anymore.
            # But we need to remove it from active_triples if it was there.
            if t in active_triples:
                active_triples.remove(t)
                # Decrement counts for the other 2 elements
                for x in t:
                    if x != best_elem:
                        active_count[x] -= 1
                        # If active_count[x] becomes 0, we don't necessarily remove x from S,
                        # but it means x is not in any active triple.
        
        # Note: We don't need to check if removing best_elem makes other triples inactive
        # that don't contain best_elem. Only triples containing best_elem become inactive.
        
    return sorted(S)

def construct():
    # We can also try a few different greedy strategies if time permits
    # But the "remove max degree" strategy is usually very strong for hitting set.
    
    S = _solve()
    
    if not S:
        return [1]
        
    return S

# EVOLVE-BLOCK-END
