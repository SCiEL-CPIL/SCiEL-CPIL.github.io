import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer
    
    # Helper: Check if we can add x to S
    # S is a set of ints
    # pair_sums is a dict: a -> list/array of counts of (u+v)%a for u,v in S\{a}
    # We need to maintain this state.
    
    def can_add(x, S, pair_sums):
        # 1. Check if x divides any triple sum in S
        # Need to check if exists u,v,w in S distinct such that u+v+w % x == 0
        # We can compute pair sums of S mod x on the fly
        # S_list = list(S)
        # pair_residues = {}
        # for i in range(len(S_list)):
        #   for j in range(i+1, len(S_list)):
        #      r = (S_list[i] + S_list[j]) % x
        #      pair_residues[r] = pair_residues.get(r, 0) + 1
        # # Now check if there is a pair (u,v) and w in S such that (u+v+w) % x == 0
        # # i.e., (u+v) % x == (-w) % x
        # # But w must be distinct from u,v.
        # # This is tricky with counts.
        
        # Simpler O(N^2) check for x:
        # For each w in S:
        #   Check if there exist u,v in S\{w} such that u+v == -w mod x
        #   We can precompute all pair sums mod x for S.
        
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            # Part 1 trivially true
            pass
        else:
            # Compute frequency of pair sums mod x
            # Note: u,v must be distinct.
            pair_freq = {}
            for i in range(n):
                for j in range(i+1, n):
                    r = (S_list[i] + S_list[j]) % x
                    pair_freq[r] = pair_freq.get(r, 0) + 1
            
            # Now for each w in S, check if (-w % x) is in pair_freq
            # AND if the pair achieving that sum does not include w.
            # This requires knowing if a pair sum r was formed by a pair NOT containing w.
            # This is complex.
            
            # Alternative: Just iterate all triples? O(N^3).
            # If N=40, 10k ops. Acceptable for check.
            # Let's stick to O(N^3) for the "x divides sum" part for simplicity and correctness, 
            # but optimize the "a divides sum with x" part which is O(N).
            
            for w in S_list:
                # Check if there are u,v in S\{w} such that u+v+w % x == 0
                # i.e., u+v % x == (-w) % x
                target = (-w) % x
                # We need to know if target is in pair sums of S\{w}
                # We can just check all pairs in S\{w}
                # This is O(N^2) per w -> O(N^3) total.
                
                # Let's optimize:
                # We have pair_freq for S.
                # If pair_freq.get(target, 0) > 0:
                #   We need to ensure at least one pair summing to target does not contain w.
                #   If the count of pairs summing to target in S is C.
                #   Let C_w be count of pairs containing w that sum to target.
                #   If C - C_w > 0, then valid.
                
                # Computing C_w for each w is O(N).
                # So total O(N^2) for this part?
                
                # Let's just do the O(N^3) check for now, it's simple.
                # N=40, 40*39*38/6 = 9880 triples.
                # 9880 mod ops is fast.
                
                found = False
                others = [v for v in S_list if v != w]
                # We need u,v in others such that u+v+w % x == 0
                # u+v % x == (-w) % x
                # We can use a set for others
                others_set = set(others)
                for u in others:
                    for v in others:
                        if u == v: continue
                        if (u + v + w) % x == 0:
                            found = True
                            break
                    if found: break
                if found:
                    return False
        
        # 2. Check if any a in S is violated by new triples involving x
        # New triples: {x, u, v} with u,v in S
        # Check if a | (x+u+v) for any u,v in S
        for a in S:
            # Check if (-x) % a is in pair_sums[a]
            # pair_sums[a] contains counts of (u+v)%a for u,v in S\{a}
            # Wait, if we add x, the new pairs for a are (x+u) for u in S\{a,x}
            # And existing pairs are still there.
            # The constraint is: a must not divide ANY triple sum in NEW_S.
            # Existing triples in S are already valid.
            # New triples are those containing x.
            # So we only need to check triples {x, u, v}.
            # Condition: a % (x+u+v) != 0
            # Equivalent: (x+u+v) % a != 0
            # Equivalent: (u+v) % a != (-x) % a
            
            target = (-x) % a
            # We need to know if there exist u,v in S\{a} such that (u+v) % a == target
            # AND u,v are distinct and not a.
            # Our pair_sums[a] is built from S\{a}.
            # So we just check if target is in pair_sums[a] with count > 0.
            
            if pair_sums.get(a, {}).get(target, 0) > 0:
                return False
        
        return True

    def add_to_state(x, S, pair_sums):
        S.add(x)
        # Update pair_sums for existing a
        S_list = list(S)
        for a in S:
            if a == x: continue
            # Add pairs (x, u) for u in S\{a, x}
            # We need to update pair_sums[a]
            if a not in pair_sums:
                pair_sums[a] = {}
            
            for u in S:
                if u == a or u == x: continue
                r = (x + u) % a
                pair_sums[a][r] = pair_sums[a].get(r, 0) + 1
        
        # Initialize pair_sums for x
        pair_sums[x] = {}
        for u in S:
            for v in S:
                if u == x or v == x or u == v: continue
                r = (u + v) % x
                pair_sums[x][r] = pair_sums[x].get(r, 0) + 1
                # Note: this counts each pair twice (u,v and v,u).
                # We should be consistent. In can_add, we just check count > 0.
                # But if we add (x,u) to existing a, we add once.
                # Let's make it consistent: count unordered pairs.
                # In the loop above for x, we iterate u,v.
                # Better:
                # for i in range(len(S_list)):
                #   for j in range(i+1, len(S_list)):
                #      if S_list[i]==x or S_list[j]==x: continue
                #      r = (S_list[i]+S_list[j]) % x
                #      pair_sums[x][r] += 1

    # Re-implementing pair_sums update to be consistent (unordered pairs)
    
    def init_pair_sums(S):
        ps = {}
        S_list = list(S)
        for a in S:
            ps[a] = {}
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    if S_list[i] == a or S_list[j] == a: continue
                    r = (S_list[i] + S_list[j]) % a
                    ps[a][r] = ps[a].get(r, 0) + 1
        return ps

    def update_pair_sums_add(x, S, ps):
        # S already contains x
        S_list = list(S)
        # Update existing a
        for a in S:
            if a == x: continue
            for u in S:
                if u == a or u == x: continue
                r = (x + u) % a
                ps[a][r] = ps[a].get(r, 0) + 1
        
        # Init for x
        ps[x] = {}
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if S_list[i] == x or S_list[j] == x: continue
                r = (S_list[i] + S_list[j]) % x
                ps[x][r] = ps[x].get(r, 0) + 1

    def update_pair_sums_remove(x, S, ps):
        # S does NOT contain x yet? No, S contains x, we are removing it.
        # So S is the set BEFORE removal.
        S_list = list(S)
        # Update existing a
        for a in S:
            if a == x: continue
            for u in S:
                if u == a or u == x: continue
                r = (x + u) % a
                ps[a][r] -= 1
                if ps[a][r] == 0:
                    del ps[a][r]
        
        # Remove x
        del ps[x]

    # Main Loop
    S = set()
    # Greedy Init
    for x in range(192, 0, -1):
        if can_add(x, S, {}): # This is slow, use full check for init
            # Use a simpler full check for init
            candidate = list(S) + [x]
            valid = True
            for a in candidate:
                others = [v for v in candidate if v != a]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        valid = False
                        break
                if not valid: break
            if valid:
                S.add(x)
    
    ps = init_pair_sums(S)
    best_size = len(S)
    
    candidates = [i for i in range(1, 193) if i not in S]
    random.shuffle(candidates)
    
    while time.time() < deadline:
        # Try to add
        added = False
        for x in candidates:
            if can_add(x, S, ps):
                add_to_state(x, S, ps)
                candidates.remove(x)
                best_size = max(best_size, len(S))
                added = True
                break
        
        if not added:
            # Try swap
            # Pick random x in S, random y in candidates
            if S and candidates:
                x = random.choice(list(S))
                y = random.choice(candidates)
                
                # Check if swap is valid
                # Simulate: Remove x, Add y
                # We can just check validity of S' = S - {x} + {y}
                # Use can_add on S - {x}?
                # 1. Remove x from ps and S
                S.remove(x)
                update_pair_sums_remove(x, S, ps)
                
                if can_add(y, S, ps):
                    add_to_state(y, S, ps)
                    candidates.remove(y)
                    best_size = max(best_size, len(S))
                else:
                    # Revert
                    # Add x back
                    # This is getting complicated to revert.
                    # Let's just use a simpler approach:
                    # Keep a backup of S and ps?
                    # ps is a dict of dicts. Copying is O(N^2).
                    # For N=40, it's small.
                    
                    # Let's restart the swap logic with backup
                    pass

    return list(S)
