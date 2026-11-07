"""Solve as Maximum Independent Set on conflict graph with branch-and-bound."""
import time
import random

DEADLINE = 85.0

def _build_conflict_graph():
    """Build conflict graph: edge (a,b) if ∃c∈[2,256], c≠a,b, s.t. a|bc or b|ac."""
    n = 257
    # For each a, precompute which b's conflict with a (via some c)
    # a and b conflict if ∃c: a|b*c or b|a*c
    # a|b*c: for each prime p^e in a, need v_p(b)+v_p(c) >= e
    # This means: a|b*c iff c is in the "required set" for (a,b)
    # Precompute: for each a, for each b, does there exist c≠a,b with a|b*c?
    
    # Factorize all numbers
    spf = list(range(257))
    for i in range(2, 257):
        if spf[i] == i:
            for j in range(i*i, 257, i):
                if spf[j] == j:
                    spf[j] = i
    
    fact = [None] * 257
    for i in range(2, 257):
        f = {}
        x = i
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            f[p] = e
        fact[i] = f
    
    # For each pair (a,b), check if there exists c such that a|b*c
    # a|b*c iff for all p^e in a: v_p(b) + v_p(c) >= e
    # i.e., for all p^e in a: v_p(c) >= e - v_p(b)
    # So c must be divisible by L = ∏ p^max(0, e-v_p(b))
    # And c must be in [2,256], c≠a, c≠b
    # Such c exists iff L <= 256 and there's a multiple of L in [2,256] that's not a or b
    
    # Actually, c just needs to be a multiple of L in [2,256].
    # If L==1, any c works. If L>1, need L<=256 (always true since a,b<=256).
    # The smallest multiple is L itself (if L>=2) or 2 (if L==1).
    # So a|b*c has a solution c iff:
    #   - L <= 256 (always true)
    #   - There exists a multiple of L in [2,256] that is not a and not b
    #   If L==1: multiples are 2,3,...,256. Need one that's not a,b. Since a,b>=2, 
    #     if {a,b} != {2,3} or similar, there's always another. Actually with 255 numbers,
    #     removing 2 leaves 253 options. So always exists.
    #   If L>1: multiples are L, 2L, 3L, ... up to 256. Count = floor(256/L).
    #     Need at least one that's not a and not b.
    #     If floor(256/L) >= 3, always exists.
    #     If floor(256/L) == 2, need at least one of the two multiples not in {a,b}.
    #     If floor(256/L) == 1, need L not in {a,b}.
    
    # Simpler: just check if there's any c in [2,256] \ {a,b} with a|b*c
    # For efficiency, note that if a|b, then a|b*1 but c must be >=2, so a|b*c for any c.
    # Actually if a|b, then b = a*k, so b*c = a*k*c, so a|b*c always. So a and b conflict.
    
    # Let me just compute it directly but efficiently.
    # For each a, b: compute L(a,b) = ∏ p^max(0, v_p(a) - v_p(b))
    # Then a|b*c iff L(a,b) | c
    # So need c in [2,256] \ {a,b} with L(a,b)|c
    
    conflict = [[False]*257 for _ in range(257)]
    
    for a in range(2, 257):
        fa = fact[a]
        for b in range(2, 257):
            if a == b:
                continue
            fb = fact[b]
            
            # Compute L(a,b): the part of a not covered by b
            L = 1
            for p, e in fa.items():
                need = e - fb.get(p, 0)
                if need > 0:
                    L *= p ** need
            
            # Check if there exists c in [2,256]\{a,b} with L|c
            if L == 1:
                # Any c works. Need c in [2,256] not equal to a or b.
                # There are 255 numbers in [2,256], removing 2 leaves 253. Always exists.
                conflict[a][b] = True
            else:
                # Multiples of L in [2,256]: L, 2L, 3L, ..., floor(256/L)*L
                # But we need c >= 2, so if L==1 it's handled above.
                # Count multiples: floor(256/L)
                # Need at least one that's not a and not b.
                count = 256 // L
                if count >= 3:
                    conflict[a][b] = True
                elif count == 2:
                    # Multiples are L and 2L
                    # Need one not in {a,b}
                    if not (L == a or L == b) or not (2*L == a or 2*L == b):
                        # Actually need: (L not in {a,b}) OR (2L not in {a,b})
                        if (L != a and L != b) or (2*L != a and 2*L != b):
                            conflict[a][b] = True
                elif count == 1:
                    # Only multiple is L
                    if L != a and L != b:
                        conflict[a][b] = True
    
    # Symmetrize (should already be symmetric by construction but let's be safe)
    # Actually it IS symmetric: a|b*c iff b|a*c? No! Not necessarily.
    # Wait: a|b*c means a divides b*c. b|a*c means b divides a*c.
    # These are different conditions!
    # Let me re-examine.
    # The condition for conflict is: ∃c: a|b*c OR b|a*c
    # I only checked a|b*c. Let me also check b|a*c.
    
    # Actually, by symmetry of the problem, if a|b*c then we have a conflict.
    # And b|a*c is a separate condition. Let me redo.
    
    # Hmm, actually let me reconsider. The condition is:
    # a and b conflict if ∃c≠a,b: a|b*c OR b|a*c
    # 
    # a|b*c: L(a,b) | c where L(a,b) = a / gcd(a,b) * ... no.
    # a|b*c iff for all p: v_p(a) <= v_p(b) + v_p(c)
    # iff for all p: v_p(c) >= v_p(a) - v_p(b)
    # iff c is divisible by ∏ p^max(0, v_p(a)-v_p(b))
    # Let's call this R(a,b) = ∏ p^max(0, v_p(a)-v_p(b))
    # Note R(a,b) = a / gcd(a,b) is NOT correct in general.
    # Actually: a = gcd(a,b) * (a/gcd(a,b)), and a/gcd(a,b) and b/gcd(a,b) are coprime.
    # v_p(a) - v_p(b) for p|a: if p|b, v_p(b)>=1, so max(0,v_p(a)-v_p(b)) could be less.
    # R(a,b) is actually a / gcd(a,b) only if... no.
    # Example: a=12=2^2*3, b=6=2*3. R(12,6) = 2^(2-1)*3^(1-1) = 2. a/gcd(12,6)=12/6=2. Yes.
    # a=8=2^3, b=4=2^2. R(8,4)=2^(3-2)=2. a/gcd(8,4)=8/4=2. Yes.
    # a=12, b=18=2*3^2. R(12,18)=2^(2-1)*3^(1-2)=2*1=2. a/gcd(12,18)=12/6=2. Yes.
    # So R(a,b) = a/gcd(a,b). 
    
    # So a|b*c iff (a/gcd(a,b)) | c.
    # Similarly b|a*c iff (b/gcd(a,b)) | c.
    
    # Let me redo with this cleaner formulation.
    import math
    conflict = [[False]*257 for _ in range(257)]
    
    for a in range(2, 257):
        for b in range(a+1, 257):
            g = math.gcd(a, b)
            Ra = a // g  # a|b*c iff Ra | c
            Rb = b // g  # b|a*c iff Rb | c
            
            # Check if ∃c in [2,256]\{a,b} with Ra|c or Rb|c
            # If Ra==1, then Ra|c for all c, so always exists (255-2=253 options).
            # Similarly for Rb.
            
            found = False
            if Ra == 1 or Rb == 1:
                found = True
            else:
                # Need c in [2,256]\{a,b} with Ra|c or Rb|c
                # Multiples of Ra: Ra, 2Ra, ..., floor(256/Ra)*Ra
                # Multiples of Rb: Rb, 2Rb, ..., floor(256/Rb)*Rb
                # Union of these, minus {a,b}, must be non-empty.
                # For efficiency, check count.
                count_a = 256 // Ra
                count_b = 256 // Rb
                # Total distinct multiples could be up to count_a + count_b
                # But we just need to know if the union minus {a,b} is non-empty.
                # Simplest: if count_a + count_b >= 3, likely exists.
                # More carefully:
                if count_a >= 3:
                    found = True
                elif count_b >= 3:
                    found = True
                else:
                    # Small counts, enumerate
                    multiples = set()
                    for k in range(1, count_a + 1):
                        multiples.add(k * Ra)
                    for k in range(1, count_b + 1):
                        multiples.add(k * Rb)
                    # Remove a and b
                    multiples.discard(a)
                    multiples.discard(b)
                    if multiples:
                        found = True
            
            if found:
                conflict[a][b] = True
                conflict[b][a] = True
    
    return conflict

def _mis_branch_and_bound(conflict, vertices, deadline):
    """Find Maximum Independent Set using branch-and-bound."""
    n = len(vertices)
    if n == 0:
        return []
    
    # Index mapping
    idx = {v: i for i, v in enumerate(vertices)}
    N = n
    
    # adjacency as bitmasks for speed
    adj = [0] * N
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices):
            if i != j and conflict[v][w]:
                adj[i] |= (1 << j)
    
    best_set = []
    best_size = 0
    
    def expand(selected, candidates_mask):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        # Upper bound: |selected| + popcount(candidates_mask)
        # Simple bound: |selected| + number of candidates
        cand_count = bin(candidates_mask).count('1')
        if len(selected) + cand_count <= best_size:
            return
        
        if cand_count == 0:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
        
        # Choose a vertex to branch on
        # Pick vertex with highest degree in candidates subgraph
        best_v = -1
        best_deg = -1
        temp_mask = candidates_mask
        while temp_mask:
            lowbit = temp_mask & (-temp_mask)
            v = lowbit.bit_length() - 1
            temp_mask ^= lowbit
            deg = bin(adj[v] & candidates_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v < 0:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
        
        # Branch 1: include best_v
        selected.append(vertices[best_v])
        new_mask = candidates_mask & ~(1 << best_v)
        # Remove all neighbors of best_v from candidates
        new_mask &= ~adj[best_v]
        expand(selected, new_mask)
        selected.pop()
        
        # Branch 2: exclude best_v
        expand(selected, candidates_mask & ~(1 << best_v))
    
    initial_mask = (1 << N) - 1
    expand([], initial_mask)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Build conflict graph
    conflict = _build_conflict_graph()
    
    # Find all vertices
    all_vertices = list(range(2, 257))
    
    # Use branch-and-bound MIS
    result = _mis_branch_and_bound(conflict, all_vertices, deadline)
    
    if not result:
        # Fallback: all primes
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0:
                    return False
            return True
        result = [x for x in range(2, 257) if is_prime(x)]
    
    # Verify
    def _valid(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not _valid(result):
        # Fallback
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0:
                    return False
            return True
        result = [x for x in range(2, 257) if is_prime(x)]
    
    return list(result)
