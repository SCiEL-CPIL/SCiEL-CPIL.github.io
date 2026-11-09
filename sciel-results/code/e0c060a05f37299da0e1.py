"""Rewritten program: Maximum Independent Set on conflict structure with greedy + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(vals, seed_for_rng):
        n = len(vals)
        if n == 0:
            return []
        rng = random.Random(seed_for_rng * 1000003 + 7)
        val_set = set(vals)
        
        # Build conflict graph G: u ~ v if u + v in val_set
        # This ensures that if u, v are in S, u+v is NOT in val_set, so u+v is NOT in S.
        # Thus S is sum-free.
        adj = {v: set() for v in vals}
        for i in range(n):
            u = vals[i]
            for j in range(i + 1, n):
                v = vals[j]
                if (u + v) in val_set:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Greedy MIS: Pick vertex with min degree in remaining graph
        available = set(vals)
        S = set()
        while available:
            # Find vertex with min degree in available
            min_deg = float('inf')
            best_v = None
            for v in available:
                deg = len(adj[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            S.add(best_v)
            # Remove best_v and its neighbors from available
            available.discard(best_v)
            for nb in adj[best_v]:
                available.discard(nb)
        
        # S is a valid sum-free set.
        # Now improve with Simulated Annealing
        # We want to add vertices from V \ S if possible, or swap.
        # Conflict check for adding x to S:
        # Is there a, b in S such that a+b=x, a+x=b, or b+x=a?
        # This is equivalent to: 
        # 1. x in S+S
        # 2. S in x+S (i.e., exists a in S such that a+x in S)
        
        def check_conflicts(x, S_set):
            # Check if adding x to S_set creates a sum-free violation
            # Violation if:
            # 1. Exists a, b in S_set such that a + b = x
            # 2. Exists a in S_set such that a + x in S_set (which implies a+x=b for b in S_set)
            #    Note: if a+x in S_set, then a+x=b. Since a,x,b in S_set, this is a violation.
            #    Also covers x+a=b.
            #    What about x+b=a? Same as x+a=b.
            #    What about x+x=a? i.e. 2x in S_set.
            #    This is covered by "Exists a in S_set such that a+x in S_set" if we consider a=x? No, x is not in S_set yet.
            #    So we must check:
            #    - For each a in S_set: if (a + x) in S_set, conflict.
            #    - For each a in S_set: if (a - x) in S_set, conflict? 
            #      If a - x = b, then a = b + x. So b + x = a. Conflict.
            #      So checking (a+x) in S_set and (a-x) in S_set covers all cases where x is a summand.
            #    - Also need to check if x is the sum: a+b=x.
            #      This is not covered by the above if we only check a+x.
            #      So we need to check if x is in S_set + S_set.
            
            # Optimization:
            # 1. Check if x is in S_set + S_set.
            #    This is O(N^2) or O(N) if we use sets?
            #    We can precompute S_set + S_set? No, S changes.
            #    For a single x, checking if x in S+S is O(N) if we iterate S and check if (x-a) in S.
            # 2. Check if x creates a pair sum in S.
            #    Iterate a in S: if (a+x) in S or (a-x) in S.
            
            # Total O(N) per check.
            
            # Check 1: x is sum of two in S
            for a in S_set:
                if (x - a) in S_set:
                    return True
            # Check 2: x is summand
            for a in S_set:
                if (a + x) in S_set or (a - x) in S_set:
                    return True
            return False

        cur = set(S)
        best = set(cur)
        best_size = len(cur)
        T = 1.0
        cooling = 0.9995
        steps = 10000
        removed = val_set - cur
        
        for step in range(steps):
            if T < 1e-6:
                break
            
            move = rng.random()
            if move < 0.7 and removed:
                # Try to add a random removed element
                x = rng.choice(list(removed))
                if check_conflicts(x, cur):
                    # Conflict found. Accept with prob exp(-1/T) ?
                    # We don't know the "cost" exactly, but 1 conflict is bad.
                    if rng.random() < math.exp(-1.0 / max(T, 1e-9)):
                        cur.add(x)
                        removed.discard(x)
                else:
                    cur.add(x)
                    removed.discard(x)
                    if len(cur) > best_size:
                        best = set(cur)
                        best_size = len(cur)
            else:
                # Try to remove a random element from cur to allow better additions later
                # Or just do a swap
                if cur:
                    x = rng.choice(list(cur))
                    cur.discard(x)
                    removed.add(x)
            
            T *= cooling

        # Ensure best is valid
        # Re-verify best
        bs = set(best)
        valid = True
        for a in bs:
            for b in bs:
                if (a + b) in bs:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to greedy S
            return sorted(S)
            
        return sorted(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        sub = solve_one(vals, seed)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
