"""Maximum-cardinality subset of [1,100] with no subset summing to 200."""
import time

_DEADLINE = 90.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Based on analysis, the set {1..19} U {66..90} has cardinality 19 + 25 = 44
    # and no subset sums to 200.
    # 
    # Verification:
    # K = {1, ..., 19}
    # L = {66, ..., 90}
    # 
    # 1. Subsets of K: Max sum = 19*20/2 = 190 < 200.
    # 2. Subsets of L: 
    #    Min 3-sum = 66+67+68 = 201 > 200.
    #    Max 2-sum = 90+89 = 179 < 200.
    #    So no subset of L sums to 200.
    # 3. Mixed subsets:
    #    Let T be a subset of K U L summing to 200.
    #    Let T_K = T intersect K, T_L = T intersect L.
    #    sum(T_K) + sum(T_L) = 200.
    #    
    #    Case 1: T_L is empty. sum(T_K) = 200. Impossible (max 190).
    #    Case 2: T_K is empty. sum(T_L) = 200. Impossible (see above).
    #    Case 3: Both non-empty.
    #       Let k = |T_K|, l = |T_L|.
    #       Min sum(T_L) for l elements:
    #         l=1: 66
    #         l=2: 66+67=133
    #         l>=3: >=201
    #       Max sum(T_K) for k elements:
    #         k=1: 19
    #         k=2: 19+18=37
    #         k=3: 19+18+17=54
    #         ...
    #       
    #       If l>=3, sum(T_L) >= 201 > 200. Impossible.
    #       So l must be 1 or 2.
    #       
    #       If l=2:
    #          sum(T_L) is between 133 and 179.
    #          sum(T_K) = 200 - sum(T_L) is between 200-179=21 and 200-133=67.
    #          Max sum(T_K) for k=1 is 19 < 21.
    #          Max sum(T_K) for k=2 is 37.
    #          So we need sum(T_K) in [21, 37] with k=2.
    #          Is it possible?
    #          We need x1+x2 in [21, 37] and x1+x2 + y1+y2 = 200.
    #          y1+y2 = 200 - (x1+x2).
    #          We know y1,y2 in L sum to y1+y2.
    #          We established that for x in K, y+z = 200-x has no solution in L.
    #          Here x1+x2 is the "x".
    #          Let S_K = x1+x2. S_K in [3, 37].
    #          We need y1+y2 = 200 - S_K.
    #          200 - S_K is in [163, 197].
    #          Max sum in L is 179.
    #          So we need y1+y2 in [163, 179].
    #          Does L contain pairs summing to values in [163, 179]?
    #          L = {66..90}.
    #          Pairs summing to S:
    #          If S=179: 89+90. Both in L.
    #          So if S_K = 21, then y1+y2=179. 89+90=179.
    #          Is there x1,x2 in K summing to 21?
    #          10+11=21. Both in K.
    #          So {10, 11, 89, 90} sums to 200.
    #          WAIT.
    #          My previous check said "1K+2L safe".
    #          But here we have 2K+2L.
    #          I only checked 1K+2L and 2K+1L.
    #          I did NOT check 2K+2L.
    
    # Let's re-evaluate 2K+2L.
    # We need x1+x2 + y1+y2 = 200.
    # x1,x2 in K={1..19}. y1,y2 in L={66..90}.
    # x1+x2 in [3, 37].
    # y1+y2 in [133, 179].
    # Sum in [136, 216].
    # We need sum = 200.
    # So y1+y2 = 200 - (x1+x2).
    # Let S_K = x1+x2.
    # S_K in [3, 37].
    # y1+y2 in [163, 197].
    # But max y1+y2 is 179.
    # So we need y1+y2 in [163, 179].
    # This implies S_K = 200 - (163..179) = 37..21.
    # So we need x1+x2 in [21, 37].
    # And y1+y2 = 200 - (x1+x2) must be achievable by two elements in L.
    # 
    # Example: x1=10, x2=11. S_K=21.
    # y1+y2 = 179.
    # 89+90=179. Both in L.
    # So {10, 11, 89, 90} is a valid subset of S summing to 200.
    # 
    # So S = {1..19} U {66..90} is INVALID.

    # We need to fix this.
    # We must remove elements to break all such sums.
    # The conflicts are pairs (x1,x2) in K and (y1,y2) in L such that sums add to 200.
    # This is a bipartite graph problem or hitting set.
    
    # Given the time constraint and the complexity, let's stick to the safe 35-element solution 
    # or try a slightly improved one that is definitely safe.
    # 
    # The set {66..100} (size 35) is safe.
    # The set {1..19} U {66..90} is NOT safe.
    
    # Let's try removing the problematic large elements.
    # The conflicts involve large sums.
    # If we restrict L to {66..80} (size 15).
    # Max 2-sum 80+79=159.
    # Then 2K+2L: y1+y2 <= 159.
    # x1+x2 = 200 - y1+y2 >= 41.
    # Max x1+x2 in K={1..19} is 37.
    # 37 < 41. So no 2K+2L conflicts.
    # 1K+2L: y1+y2 <= 159. x = 200 - y1+y2 >= 41.
    # Max x in K is 19. 19 < 41. No 1K+2L conflicts.
    # 2K+1L: y <= 80. x1+x2 = 200 - y >= 120.
    # Max x1+x2 is 37. No conflicts.
    # 
    # So K={1..19} and L={66..80} is SAFE.
    # Size: 19 + 15 = 34.
    # This is worse than 35.
    
    # Let's try K={1..10} and L={66..90}.
    # Size: 10 + 25 = 35.
    # Check 2K+2L:
    # Max x1+x2 = 10+9=19.
    # y1+y2 = 200 - S_K >= 181.
    # Max y1+y2 in L is 179.
    # 181 > 179. No conflicts.
    # So {1..10} U {66..90} is SAFE.
    # Size 35. Same as baseline.
    
    # Can we do better?
    # K={1..11}?
    # Max x1+x2 = 11+10=21.
    # y1+y2 = 200 - 21 = 179.
    # 89+90=179. Conflict.
    # So {1..11} U {66..90} is INVALID.
    
    # It seems 35 is the max for this simple block strategy.
    
    return list(range(66, 101))

def _greedy_fallback():
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            selected.append(x)
            reachable |= new_sums
    return selected
