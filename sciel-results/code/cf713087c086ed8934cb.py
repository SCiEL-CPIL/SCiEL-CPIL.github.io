import math
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    
    N = 256
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    small_primes = [p for p in primes if p <= 16]  # 2,3,5,7,11,13
    large_primes = [p for p in primes if p > 16]   # 17,19,...,256
    
    # For each number 2..256, compute its prime factorization (set of primes)
    # and separate into small-prime part and large-prime part
    def factorize(x):
        factors = set()
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                factors.add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            factors.add(x)
        return factors
    
    # For each number, store: (small_prime_set, large_prime_set, value)
    # small_prime_set: which small primes divide it
    # large_prime_set: which large primes divide it (at most one, since product of two > 256)
    num_info = {}
    for v in range(2, N+1):
        fac = factorize(v)
        small_part = frozenset(p for p in fac if p in small_primes)
        large_part = frozenset(p for p in fac if p in large_primes)
        num_info[v] = (small_part, large_part, v)
    
    # Generate all set partitions of subsets of small_primes
    # A partition is a list of disjoint subsets whose union is a subset of small_primes
    # Each subset in the partition forms a "skeleton" (product of those primes)
    
    def get_all_partitions(elements):
        """Yield all partitions of subsets of elements (including empty)."""
        if not elements:
            yield []
            return
        first = elements[0]
        rest = elements[1:]
        
        # Get all partitions of subsets of rest
        for part in get_all_partitions(rest):
            # Option 1: don't include first
            yield part
            
            # Option 2: include first in a new block
            yield part + [[first]]
            
            # Option 3: include first in an existing block
            for i in range(len(part)):
                new_part = [list(b) for b in part]
                new_part[i] = new_part[i] + [first]
                yield new_part
    
    # Precompute all skeleton configurations
    # Each config: list of frozensets (each is a subset of small_primes)
    # Deduplicate by sorting each block and the list of blocks
    all_configs = set()
    for part in get_all_partitions(small_primes):
        # Normalize: sort each block, sort the list of blocks
        normalized = tuple(sorted(tuple(sorted(block)) for block in part))
        all_configs.add(normalized)
    
    # For a given skeleton configuration, determine which small primes are "used"
    # and which numbers are compatible with the skeletons
    
    def solve_for_config(config):
        """
        config: tuple of tuples, each inner tuple is a subset of small_primes
        Returns: (total_sum, selected_numbers_list)
        """
        if time.time() > deadline:
            return (-1, [])
        
        # Determine which small primes are used in skeletons
        used_small = set()
        for block in config:
            for p in block:
                used_small.add(p)
        
        # For each skeleton, compute its product
        skeleton_products = []
        for block in config:
            prod = 1
            for p in block:
                prod *= p
            skeleton_products.append(prod)
        
        # A number v is compatible with a skeleton config if:
        # - Its small prime set is a subset of used_small (i.e., it only uses small primes that are in skeletons)
        # - For each skeleton block, v's small primes intersect with that block in at most... 
        #   Actually, v must be coprime with all other selected numbers.
        #   The skeletons themselves are selected numbers (or multipliers).
        
        # Let me rethink: The selected numbers must be pairwise coprime.
        # Each selected number v has small_prime_set(v) and large_prime_set(v).
        # Two numbers are coprime iff they share no prime factors.
        
        # Strategy: 
        # 1. We select some numbers that have small prime factors. These must have disjoint small prime sets.
        # 2. We select some numbers that have large prime factors. Each large prime can appear in at most one selected number.
        # 3. A number can have both small and large prime factors, but then it "uses" both.
        
        # For a fixed skeleton config (partition of a subset of small primes):
        # - The skeletons themselves are NOT necessarily selected; they define which small primes are "available" to be used together.
        # - Actually, let me reconsider.
        
        # Better approach:
        # The small primes that appear in any selected number must form a partition into disjoint groups, where each group is the set of small primes in one selected number.
        # So a "skeleton config" tells us how small primes are grouped among selected numbers.
        # For each group (skeleton), we can choose a number that has exactly those small primes (and possibly one large prime).
        # Numbers with no small primes (i.e., powers of large primes, but only single large primes since p^2 > 256 for p>16) can also be selected, each using one large prime.
        
        # So the problem decomposes as:
        # For each skeleton block B (subset of small primes):
        #   Choose at most one number v such that small_prime_set(v) = B and large_prime_set(v) = {p} for some large prime p, OR large_prime_set(v) = empty.
        #   The value is v. We want to maximize sum.
        # Additionally, we can choose numbers with small_prime_set = empty (i.e., pure large prime powers, but only single large primes since 17^2>256).
        #   Each such number uses one large prime.
        
        # Wait, a number like 4 = 2^2 has small_prime_set = {2}, large_prime_set = empty.
        # A number like 2*17 = 34 has small_prime_set = {2}, large_prime_set = {17}.
        # A number like 17 has small_prime_set = empty, large_prime_set = {17}.
        
        # For a fixed partition of small primes into blocks, for each block B:
        #   We can choose a number with small_prime_set = B. The best such number is the largest v in [2,256] with that small prime set, possibly times a large prime.
        #   But we also need to ensure large primes are not reused.
        
        # This is getting complex. Let me use a different approach.
        
        # Simpler: For each skeleton config, do a greedy/matching approach.
        # 1. For each block B in the config, find the best number with small_prime_set = B (maximizing value, considering large prime conflicts).
        # 2. For remaining large primes, find the best number with small_prime_set = empty.
        
        # Actually, let me just do a clean branch-and-bound but with the structural insight to reduce the search space.
        
        # New plan: Use the structural decomposition to create a smaller problem.
        # Step 1: Enumerate all possible "small prime assignments" - i.e., for each number, which small primes it uses.
        # But this is still large.
        
        # Let me try a cleaner approach:
        # 1. Precompute for each subset S of small_primes, the list of numbers with small_prime_set = S.
        # 2. For each such list, the numbers differ only in their large prime factor (or none).
        # 3. For a given partition of small primes into blocks, for each block, we pick one number from the corresponding list.
        # 4. We need to ensure no large prime is used twice.
        
        # Let's precompute:
        # small_prime_groups[S] = list of (large_prime_or_None, value) for numbers with small_prime_set = S
        
        small_prime_groups = {}
        for v in range(2, N+1):
            sp, lp, val = num_info[v]
            if sp not in small_prime_groups:
                small_prime_groups[sp] = []
            large_p = None
            if lp:
                large_p = list(lp)[0]  # at most one large prime
            small_prime_groups[sp].append((large_p, val))
        
        # For each group, sort by value descending
        for sp in small_prime_groups:
            small_prime_groups[sp].sort(key=lambda x: -x[1])
        
        # Now, for a given config (partition of small primes into blocks):
        # For each block B, we can pick a number from small_prime_groups[B].
        # We need to pick numbers such that no large prime is used twice.
        # This is a matching problem: blocks on one side, large primes on the other, plus "no large prime" option.
        
        # For simplicity and speed, let's do a greedy approach:
        # Process blocks in order of decreasing best available value.
        # For each block, pick the best number that doesn't conflict with already-picked large primes.
        
        used_large = set()
        selected = []
        total = 1  # include 1
        
        # Process each block in the config
        for block in config:
            B = frozenset(block)
            if B not in small_prime_groups:
                continue
            # Find best number in this group that doesn't use an already-used large prime
            best_val = -1
            best_large = None
            for (lp, val) in small_prime_groups[B]:
                if lp is None or lp not in used_large:
                    if val > best_val:
                        best_val = val
                        best_large = lp
                    break  # sorted by value desc, first valid is best
            if best_val > 0:
                selected.append(best_val)
                total += best_val
                if best_large is not None:
                    used_large.add(best_large)
        
        # Now, for large primes not yet used, pick the best number with small_prime_set = empty
        # These are numbers that are powers of a single large prime (but only p^1 since p^2 > 256)
        # So they are just the large primes themselves, or 1 (but 1 is already included)
        # Actually, numbers with small_prime_set = empty and value > 1 are just the large primes themselves
        # (since any composite with no small primes would be a product of large primes, but 17*19 > 256)
        
        for p in large_primes:
            if p not in used_large:
                # The best number with small_prime_set = empty and large_prime = p is just p
                # But wait, could there be a number like p^k? No, since p > 16, p^2 > 256
                # So the only number with large_prime = p and no small primes is p itself
                selected.append(p)
                total += p
                used_large.add(p)
        
        return (total, selected)
    
    # Now iterate over all configs and find the best
    best_total = -1
    best_selected = [1]
    
    for config in all_configs:
        if time.time() > deadline:
            break
        
        total, selected = solve_for_config(config)
        if total > best_total:
            best_total = total
            best_selected = selected
    
    # Verify the result
    # Check all pairs are coprime
    for i in range(len(best_selected)):
        for j in range(i+1, len(best_selected)):
            if math.gcd(best_selected[i], best_selected[j]) != 1:
                # Invalid, fall back to a safe answer
                best_selected = [1]
                best_total = 1
                break
    
    return [int(x) for x in best_selected]
