"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Deadline guard
_DEADLINE = 85.0  # seconds total; we will aim to finish well before this

def construct():
    t_start = time.time()
    n = 192
    
    # We want to maximize the number of selected integers.
    # Strategy: Greedy with strong pruning and incremental checks.
    # Key observation: Larger numbers are "safer" as divisors because they are less likely to divide sums of smaller numbers.
    # However, small numbers are "dangerous" as divisors because many sums are multiples of them.
    # So, we should probably include as many large numbers as possible.
    
    # Let's try starting with large numbers, but with a smarter check.
    # The check in the seed is O(k^3 * k) which is slow.
    # We can optimize the check.
    
    # Let's precompute sums of triples? No, the set changes.
    # But we can precompute for a fixed set.
    
    # Better approach:
    # Iterate through candidates.
    # For each candidate, check if it violates the condition with the current set.
    # Condition: For all a in current_set U {x}, for all triples t from (current_set U {x}) \ {a}, sum(t) % a != 0.
    
    # This is expensive to check from scratch.
    # We can maintain:
    # 1. The current set S.
    # 2. For each a in S, the set of sums of triples from S\{a}.
    # This is hard to update incrementally.
    
    # Let's stick to the greedy but optimize the inner loop.
    # The inner loop in the seed:
    # for a in candidate:
    #   for t in combinations([v for v in candidate if v!=a], 3):
    #     if sum(t) % a == 0: return False
    
    # We can precompute the sums of triples for the current set.
    # When we add a new element x, the new triples involving x are:
    # (x, u, v) where u, v are in S.
    # The old triples remain the same.
    
    # So, for each a in S, we need to check:
    # 1. Old triples from S\{a} (already checked previously? No, because a might be part of those triples? No, the triples are from S\{a}).
    # Wait, the condition is: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
    # So for a fixed a, we look at triples from S\{a}.
    
    # If we maintain a dictionary `sums_by_a` where `sums_by_a[a]` is a set of sums of triples from S\{a},
    # then when we add x:
    # 1. For each a in S, we need to add new triples (x, u, v) to `sums_by_a[a]` IF a != x (which is true) and u,v != x (true).
    #    Actually, the new triples involving x are (x, u, v) where u,v in S.
    #    For a given a in S, the new triples from (S U {x})\{a} that involve x are (x, u, v) where u,v in S\{a} (and u!=v, u!=x, v!=x).
    #    So for each a in S, we iterate over pairs (u,v) in S\{a} and check if (x+u+v) % a == 0.
    # 2. For the new element x, we need to check if x divides any triple from S\{x} (which is S).
    #    So we need to check all triples from S.
    
    # This incremental update seems feasible.
    
    # Let's implement this.
    
    S = []
    # sums_by_a: dict mapping a -> set of sums of triples from S\{a}
    # Initially empty.
    
    # We want to maximize cardinality.
    # Heuristic: Add numbers in a specific order.
    # Let's try descending order (like seed) but with the optimized check.
    # Or maybe ascending? Small numbers are more likely to divide sums.
    # If we pick small numbers first, they might block many large numbers?
    # If we pick large numbers first, they are less likely to be divisors of sums (since sums of other large numbers are large, and large divisors are rare).
    # Actually, if a is large, sum(t) % a == 0 is rare.
    # So large numbers are "safe" as divisors.
    # But small numbers are "dangerous" as divisors.
    # So we should probably include as many large numbers as possible.
    # And avoid small numbers that divide many sums.
    
    # Let's try: Sort candidates by descending value.
    # But also consider: maybe we should skip numbers that are divisors of many potential sums.
    
    # Let's just do the optimized greedy with descending order first.
    
    candidates = list(range(n, 0, -1))
    
    for x in candidates:
        if time.time() - t_start > _DEADLINE - 5:
            break
        
        # Check if x can be added
        ok = True
        
        # 1. Check if x divides any triple from S
        # We need to check all triples from S.
        if len(S) >= 3:
            # Precompute sums of triples from S?
            # We can just iterate.
            for t in combinations(S, 3):
                if (t[0] + t[1] + t[2]) % x == 0:
                    ok = False
                    break
        
        if not ok:
            continue
        
        # 2. Check if any a in S divides a new triple involving x
        # New triples are (x, u, v) where u, v in S, u != v.
        for a in S:
            # We need to check if a divides (x + u + v) for any pair u, v in S\{a}
            # Note: u, v are distinct and not x (x is not in S yet).
            # Also u, v must be distinct from a?
            # The triple is (x, u, v). The "other" elements for a are S U {x} \ {a}.
            # So u, v can be any distinct elements from S\{a}.
            # We need to check if (x + u + v) % a == 0.
            
            # Optimization:
            # x + u + v ≡ 0 (mod a)
            # u + v ≡ -x (mod a)
            
            # We can precompute pair sums mod a?
            # Or just iterate pairs.
            
            found = False
            # Iterate over pairs u, v in S
            # We need u != v, and if a is in S, we need to handle u,v != a?
            # Actually, the triple elements are distinct. x, u, v are distinct.
            # a is one of the selected integers. The triple is from S\{a} U {x}.
            # So u, v are from S\{a}.
            
            S_without_a = [v for v in S if v != a]
            
            # Check pairs in S_without_a
            for i in range(len(S_without_a)):
                u = S_without_a[i]
                for j in range(i+1, len(S_without_a)):
                    v = S_without_a[j]
                    if (x + u + v) % a == 0:
                        found = True
                        break
                if found:
                    break
            
            if found:
                ok = False
                break
        
        if ok:
            S.append(x)
    
    return S
# EVOLVE-BLOCK-END
