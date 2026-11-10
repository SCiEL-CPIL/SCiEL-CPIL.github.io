# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 80  # Leave 10s margin for safety
    N = 512
    MAX_SIZE = 48
    
    # Precompute conflict bitsets using the 3-uniform hypergraph model.
    # A conflict exists for triple (a, b, c) where a = (b+c)/2, i.e., 2a = b+c,
    # with a, b, c all distinct. If any such triple is fully in our set, it's invalid.
    # We precompute for each element x, a bitset of all y such that there exists z
    # with {x, y, z} forming a valid conflict triple.
    # Actually, we need: for a given set S, check if any triple in S is a conflict.
    # Precompute: conflict_pairs[x] = bitset of all y where (x, y) can be part of a conflict triple.
    # But that's not quite right. Let's think differently.
    
    # For each pair (b, c), if (b+c) is even, then a = (b+c)//2. If a is distinct from b and c,
    # then {a, b, c} is a conflict triple.
    # So for a set S to be valid, no such triple should be fully contained in S.
    
    # Precompute: for each element a, which pairs (b, c) satisfy 2a = b+c with b,c != a?
    # Then for a to be in S, we need that NOT both b and c are in S for any such pair.
    
    # Even better: precompute for each element x, a bitset bad[x] where bad[x] has bit y set
    # if there exists z such that {x, y, z} is a conflict triple AND z != x, z != y.
    # Wait, that's still not quite right because the conflict involves all three.
    
    # Let me reconsider. The constraint is: no element a in S is the mean of any subset T of S\{a} with |T| >= 2.
    # For |T| = 2: a = (b+c)/2 means 2a = b+c. So triples {a,b,c} with 2a=b+c, all distinct, are conflicts.
    # For |T| >= 3: it's more complex. But let's first handle the |T|=2 case with bitsets,
    # and then verify the full constraint with DP at the end.
    
    # Precompute conflict bitsets for the |T|=2 case:
    # conflict[a] = bitset of all b such that there exists c != a, c != b with 2a = b+c.
    # Then when adding x to S, we need: for each a in S, (conflict[a] & S_bitset) must not contain
    # any b where the corresponding c is also in S. This is getting complex.
    
    # Simpler approach: precompute all conflict triples as (a, b, c) where 2a = b+c, a,b,c distinct.
    # Then use a bitset-based check: for each a in S, check if there's a pair (b,c) in S\{a} with 2a=b+c.
    
    # Let's precompute for each a, the set of all b such that (b, a) could be in a conflict.
    # Actually, let's just precompute all valid (b,c) pairs for each a where 2a = b+c.
    
    # For fast checking, precompute:
    # pairs_for_a[a] = list of (b, c) where 2a = b+c, b != a, c != a, b != c, 1 <= b,c <= 512
    
    # But storing all pairs is memory-heavy. Instead, for each a, precompute a bitset:
    # partner_bitset[a] = bitset where bit b is set if there exists c such that 2a = b+c, c != a, c != b.
    # Then to check if a conflicts with S\{a}: we need to know if there's a pair (b,c) both in S\{a} with 2a=b+c.
    # This requires checking all b in S\{a} where partner_bitset[a] has bit b set, and then checking if the
    # corresponding c is also in S\{a}.
    
    # Even faster: precompute for each a, for each possible b, the unique c = 2a - b.
    # Then conflict check for a against set S: for each b in S\{a}, if c = 2a-b is in S\{a} and c != b and c != a, conflict.
    
    # Let's precompute a 2D structure: for each a, a bitset of valid b values.
    # valid_b_for_a[a] = bitset of b such that c = 2a-b is in [1,512], c != a, c != b.
    
    valid_b_for_a = [0] * (N + 1)
    for a in range(1, N + 1):
        bs = 0
        for b in range(1, N + 1):
            c = 2 * a - b
            if 1 <= c <= N and c != a and c != b:
                bs |= (1 << b)
        valid_b_for_a[a] = bs
    
    def is_valid_set_fast(S):
        """
        Check if set S is valid using precomputed bitsets for |T|=2 conflicts.
        Then do a full DP check for |T|>=3.
        """
        if len(S) < 3:
            return True
        
        S_set = set(S)
        S_bitset = 0
        for x in S:
            S_bitset |= (1 << x)
        
        # Check |T|=2 conflicts: for each a in S, check if there's b in S\{a} with 2a=b+c and c in S\{a}
        for a in S:
            # Get all b in S\{a} that could form a conflict with a
            candidates_b = valid_b_for_a[a] & (S_bitset & ~(1 << a))
            while candidates_b:
                # Get lowest set bit
                b = (candidates_b & -candidates_b).bit_length() - 1
                candidates_b &= candidates_b - 1
                c = 2 * a - b
                if c in S_set and c != a and c != b:
                    return False
        
        # Check |T|>=3 conflicts using DP
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in S:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        
        for a in S:
            S_excl_a = [x for x in S if x != a]
            if len(S_excl_a) < 2:
                continue
            dp_excl = [0] * (len(S_excl_a) + 1)
            dp_excl[0] = 1
            for x in S_excl_a:
                for k in range(len(S_excl_a), 0, -1):
                    if dp_excl[k-1]:
                        dp_excl[k] |= (dp_excl[k-1] << x)
            m = len(S_excl_a)
            sum_excl = sum(S_excl_a)
            for k in range(2, m + 1):
                target_sum = k * a
                if target_sum > sum_excl:
                    break
                if (dp_excl[k] >> target_sum) & 1:
                    return False
        return True
    
    def is_valid_addition_fast(S, x):
        """
        Check if x can be added to S without violating constraints.
        """
        if x in S:
            return False
        
        S_set = set(S)
        S_bitset = 0
        for y in S:
            S_bitset |= (1 << y)
        
        # 1. Check if x is the mean of any subset of S of size k >= 2
        n = len(S)
        if n >= 2:
            dp_S = [0] * (n + 1)
            dp_S[0] = 1
            for y in S:
                for k in range(n, 0, -1):
                    if dp_S[k-1]:
                        dp_S[k] |= (dp_S[k-1] << y)
            sum_S = sum(S)
            for k in range(2, n + 1):
                target_sum = k * x
                if target_sum > sum_S:
                    break
                if (dp_S[k] >> target_sum) & 1:
                    return False
        
        # 2. Check if any a in S is the mean of a subset of S U {x} containing x
        for a in S:
            S_excl_a = [y for y in S if y != a]
            if not S_excl_a:
                continue
            m = len(S_excl_a)
            dp_excl = [0] * (m + 1)
            dp_excl[0] = 1
            for y in S_excl_a:
                for k in range(m, 0, -1):
                    if dp_excl[k-1]:
                        dp_excl[k] |= (dp_excl[k-1] << y)
            sum_excl = sum(S_excl_a)
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if target_sum > sum_excl:
                    break
                if (dp_excl[j] >> target_sum) & 1:
                    return False
        
        return True
    
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition_fast(selected, x):
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= MAX_SIZE:
                    break
        return selected
    
    # Strategy: 
    # 1. Generate candidate orders with prime bias.
    # 2. Run greedy for multiple orders.
    # 3. Local search with swaps.
    
    candidates = list(range(1, N + 1))
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    orders = []
    
    # Order 1: Primes descending, then composites descending
    orders.append(sorted(primes, reverse=True) + sorted(composites, reverse=True))
    
    # Order 2: Primes ascending, then composites ascending
    orders.append(sorted(primes) + sorted(composites))
    
    # Order 3: Primes descending, then composites ascending
    orders.append(sorted(primes, reverse=True) + sorted(composites))
    
    # Order 4: Composites descending, then primes descending
    orders.append(sorted(composites, reverse=True) + sorted(primes, reverse=True))
    
    # Random shuffles
    for _ in range(10):
        r = list(candidates)
        random.shuffle(r)
        orders.append(r)
    
    best_size = 0
    best_result = []
    
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_result = res
            
        if best_size == MAX_SIZE:
            break

    # Local Search / Improvement Phase
    if best_size < MAX_SIZE and time.time() < deadline - 10:
        current_best = best_result[:]
        current_set = set(current_best)
        unused = [x for x in range(1, N + 1) if x not in current_set]
        
        # Try to add more elements directly
        random.shuffle(unused)
        for x in unused:
            if time.time() > deadline - 2:
                break
            if len(current_best) >= MAX_SIZE:
                break
            if is_valid_addition_fast(current_best, x):
                current_best.append(x)
                current_set.add(x)
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_result = current_best
        
        # If still not at max, try swaps
        if len(current_best) < MAX_SIZE:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                idx_to_remove = random.randint(0, len(current_best) - 1)
                elem_to_remove = current_best[idx_to_remove]
                
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    if not unused:
                        break
                    elem_to_add = random.choice(unused)
                    
                    new_set_list = [x for x in current_best if x != elem_to_remove]
                    if elem_to_add in new_set_list:
                        continue
                    new_set_list.append(elem_to_add)
                    
                    if not is_valid_set_fast(new_set_list):
                        continue
                    
                    potential_set = new_set_list[:]
                    potential_set_set = set(potential_set)
                    remaining_unused = [x for x in unused if x != elem_to_add and x not in potential_set_set]
                    random.shuffle(remaining_unused)
                    
                    for x in remaining_unused:
                        if len(potential_set) >= MAX_SIZE:
                            break
                        if is_valid_addition_fast(potential_set, x):
                            potential_set.append(x)
                            potential_set_set.add(x)
                    
                    if len(potential_set) > len(current_best):
                        current_best = potential_set
                        current_set = potential_set_set
                        best_size = len(current_best)
                        best_result = current_best
                        improved = True
                        unused = [x for x in range(1, N + 1) if x not in current_set]
                        break

    return best_result
# EVOLVE-BLOCK-END
