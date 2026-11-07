import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill
    
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
    large_primes = [p for p in primes if p > 16]   # 17,19,...,251
    
    # For each number 2..256, compute its prime factorization (set of primes)
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
    num_info = {}
    for v in range(2, N+1):
        fac = factorize(v)
        small_part = frozenset(p for p in fac if p in small_primes)
        large_part = frozenset(p for p in fac if p in large_primes)
        num_info[v] = (small_part, large_part, v)
    
    # Precompute for each subset S of small_primes, the list of numbers with small_prime_set = S
    # Each entry: (large_prime_or_None, value)
    small_prime_groups = {}
    for v in range(2, N+1):
        sp, lp, val = num_info[v]
        if sp not in small_prime_groups:
            small_prime_groups[sp] = []
        large_p = None
        if lp:
            large_p = list(lp)[0]
        small_prime_groups[sp].append((large_p, val))
    
    # For each group, sort by value descending
    for sp in small_prime_groups:
        small_prime_groups[sp].sort(key=lambda x: -x[1])
    
    # Generate all set partitions of subsets of small_primes
    def get_all_partitions(elements):
        """Yield all partitions of subsets of elements (including empty)."""
        if not elements:
            yield []
            return
        first = elements[0]
        rest = elements[1:]
        
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
    
    # Precompute all skeleton configurations (deduplicated)
    all_configs = set()
    for part in get_all_partitions(small_primes):
        normalized = tuple(sorted(tuple(sorted(block)) for block in part))
        all_configs.add(normalized)
    
    # Build the conflict graph: edge between i and j if gcd(i,j) > 1
    # We'll use bitset representation for speed
    # Vertex 0 = number 1 (no conflicts with anything)
    # Vertices 1..255 = numbers 2..256
    
    # conflict_mask[v] is a Python int bitmask where bit u is set if v conflicts with u
    # We use vertex indices 0..255 corresponding to numbers 1..256
    conflict_mask = [0] * (N + 1)
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if math.gcd(i, j) > 1:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # For MWIS: we want to select a set of vertices (numbers) such that no two are adjacent
    # in the conflict graph, maximizing total weight (= value).
    # Vertex 1 (number 1) has no conflicts, so it's always beneficial to include it.
    
    # Branch and bound for MWIS using bitsets
    # State: (candidate_mask, current_sum)
    # candidate_mask: bitmask of vertices that can still be added
    # We branch on the highest-weight vertex in candidates
    
    # Precompute weights
    weights = [0] * (N + 1)
    for v in range(1, N + 1):
        weights[v] = v
    
    # Upper bound: sum of weights of all candidates (trivial bound)
    # Better bound: greedy coloring bound or clique cover bound
    # For simplicity and speed, use the sum of top-k weights where k is a lower bound
    # on the size of an independent set we can still pick.
    # Actually, the tightest simple bound is: for each candidate vertex, it can contribute
    # at most its weight, and since independent set vertices don't conflict, we can use
    # a greedy coloring bound.
    
    # Let's use a simpler but effective approach:
    # Upper bound = sum of weights of all candidates (since all weights are positive,
    # any independent set is a subset, so sum of all is an upper bound).
    # This is loose but fast. We can tighten with a greedy bound.
    
    # Greedy bound: repeatedly pick the highest-weight vertex, remove it and all its
    # neighbors from candidates, add its weight to bound. This gives a valid independent
    # set weight which is a LOWER bound, not upper. We need an upper bound.
    
    # For upper bound: since all weights are positive, the sum of all candidate weights
    # is a valid (but loose) upper bound. Let's use it and rely on good branching.
    
    # Better: use the fact that in an independent set, for any clique, we can pick at
    # most one vertex. A clique in the conflict graph is a set of pairwise non-coprime
    # numbers. But finding max clique is hard.
    
    # Let's use a different upper bound: sort candidates by weight descending, and the
    # upper bound is the sum of the first k weights where k is an upper bound on the
    # independence number. A simple upper bound on independence number is the number of
    # candidates (trivial). 
    # 
    # Actually, let me just use the sum of all candidate weights as the bound. With
    # good vertex ordering (highest weight first), the branch and bound should work well
    # because we quickly find a good solution and prune.
    
    best_total = -1
    best_selected = []
    
    # Always include 1
    best_total = 1
    best_selected = [1]
    
    # Candidates: all numbers 2..256
    # Use bitmask where bit v (v from 2 to 256) represents whether v is a candidate
    # We'll use Python ints as bitsets
    
    # Initial candidate set: all numbers 2..256
    # Bit v is set for v in 2..256
    initial_candidates = 0
    for v in range(2, N + 1):
        initial_candidates |= (1 << v)
    
    # Branch and bound
    # We process vertices in order of decreasing weight (i.e., decreasing value)
    # For each vertex, we decide: include or exclude
    
    # To make it efficient, we use a recursive approach with memoization-like pruning
    # but given the size (255 vertices), we need to be careful.
    
    # Alternative: use a greedy initial solution, then improve with local search.
    # Greedy: pick vertices in order of decreasing weight, if it doesn't conflict with
    # already picked vertices.
    
    # Greedy solution
    greedy_selected = [1]
    greedy_total = 1
    used_conflicts = 0  # bitmask of all vertices that conflict with selected ones
    for v in range(N, 1, -1):  # from 256 down to 2
        if not (initial_candidates & (1 << v)):
            continue
        if not (used_conflicts & (1 << v)):
            greedy_selected.append(v)
            greedy_total += v
            used_conflicts |= conflict_mask[v]
            used_conflicts |= (1 << v)  # mark v as used
    
    best_total = greedy_total
    best_selected = list(greedy_selected)
    
    # Now try to improve with local search: try swapping or adding
    # Actually, let's do a proper branch and bound on the remaining problem.
    # 
    # Given the time constraint, let's use a more targeted approach:
    # The key insight is that numbers with large prime factors (> 16) only conflict
    # with multiples of those primes. Since primes > 16 up to 256 are all unique to
    # at most a few numbers, most large primes are "free" to include.
    #
    # The hard part is the small primes (2,3,5,7,11,13) which create many conflicts.
    #
    # Let's do branch and bound on just the small-prime-conflicting numbers, and
    # greedily add all large-prime-only numbers that don't conflict.
    
    # Identify numbers whose prime factors are all <= 13 (small-prime-only numbers)
    # and numbers that have at least one prime factor > 13
    small_prime_numbers = []
    large_prime_numbers = []
    
    for v in range(2, N + 1):
        fac = factorize(v)
        has_large = any(p > 13 for p in fac)
        if has_large:
            large_prime_numbers.append(v)
        else:
            small_prime_numbers.append(v)
    
    # For small_prime_numbers, do exact MWIS via branch and bound
    # For large_prime_numbers, greedily add those that don't conflict with selected
    
    # Build conflict graph among small_prime_numbers only
    spn_set = set(small_prime_numbers)
    spn_index = {v: i for i, v in enumerate(small_prime_numbers)}
    n_spn = len(small_prime_numbers)
    
    # Conflict mask among small prime numbers
    spn_conflict = [0] * n_spn
    for i in range(n_spn):
        for j in range(i + 1, n_spn):
            if math.gcd(small_prime_numbers[i], small_prime_numbers[j]) > 1:
                spn_conflict[i] |= (1 << j)
                spn_conflict[j] |= (1 << i)
    
    # Also track conflicts between small prime numbers and large prime numbers
    # For each small prime number, which large prime numbers does it conflict with?
    spn_to_lpn_conflict = []
    for v in small_prime_numbers:
        conflicting_lpn = set()
        for w in large_prime_numbers:
            if math.gcd(v, w) > 1:
                conflicting_lpn.add(w)
        spn_to_lpn_conflict.append(conflicting_lpn)
    
    # Branch and bound on small prime numbers
    # State: (candidate_mask, selected_mask, current_sum)
    # We want to maximize: current_sum + (sum of large prime numbers not conflicting with selected)
    
    # Precompute for each subset of small prime numbers (as bitmask), the set of large
    # prime numbers that conflict with at least one selected small prime number
    # This is expensive for 2^n subsets, so we compute on the fly
    
    # Greedy initial solution for small prime numbers
    greedy_spn = [False] * n_spn
    greedy_spn_sum = 0
    for i in range(n_spn - 1, -1, -1):  # process in order of decreasing value
        # Check if i conflicts with any already selected
        conflict_with_selected = False
        for j in range(n_spn):
            if greedy_spn[j] and (spn_conflict[i] & (1 << j)):
                conflict_with_selected = True
                break
        if not conflict_with_selected:
            greedy_spn[i] = True
            greedy_spn_sum += small_prime_numbers[i]
    
    # Compute which large prime numbers are available given greedy_spn selection
    def compute_available_lpn(selected_spn_mask):
        """Return set of large prime numbers that don't conflict with any selected spn."""
        conflicting = set()
        for i in range(n_spn):
            if selected_spn_mask & (1 << i):
                conflicting |= spn_to_lpn_conflict[i]
        available = set(large_prime_numbers) - conflicting
        return available
    
    greedy_lpn_available = compute_available_lpn(0)
    for i in range(n_spn):
        if greedy_spn[i]:
            greedy_lpn_available -= spn_to_lpn_conflict[i]
    
    greedy_lpn_sum = sum(greedy_lpn_available)
    greedy_total_full = 1 + greedy_spn_sum + greedy_lpn_sum
    greedy_selected_full = [1] + [small_prime_numbers[i] for i in range(n_spn) if greedy_spn[i]] + sorted(greedy_lpn_available)
    
    best_total = greedy_total_full
    best_selected = list(greedy_selected_full)
    
    # Now branch and bound
    # Upper bound: current_spn_sum + sum of remaining candidate spn weights + max possible lpn sum
    # Max possible lpn sum = sum of all lpn (since we can always choose not to select any spn)
    all_lpn_sum = sum(large_prime_numbers)
    
    # Branch and bound with recursion
    # To avoid deep recursion issues, use iterative deepening or limit depth
    # Given n_spn is at most ~100, we need to be careful
    
    # Let's count how many small prime numbers there are
    # Numbers 2..256 with all prime factors <= 13
    # These are numbers of the form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f <= 256
    # This is a manageable set
    
    # Use a stack-based DFS to avoid recursion depth issues
    # State: (candidate_mask, selected_mask, spn_sum)
    # We branch on the highest-weight candidate
    
    # Precompute weight order for small prime numbers
    spn_order = sorted(range(n_spn), key=lambda i: -small_prime_numbers[i])
    
    # For branch and bound, we need an upper bound on the additional lpn sum
    # Given a partial selection, the best we can do for lpn is:
    # sum of lpn that don't conflict with selected spn
    # But we haven't decided on remaining spn, so the upper bound is:
    # sum of lpn that don't conflict with selected spn (ignoring future spn selections)
    # This is an upper bound because future spn selections can only reduce available lpn
    
    # Let's implement the branch and bound
    # To keep it tractable, we'll use a time limit and return the best found
    
    # Convert to a form where we can efficiently compute the lpn upper bound
    # For each spn index i, spn_to_lpn_conflict[i] is a set
    # We want: given selected_spn_mask, compute sum of lpn not in union of conflicts
    
    # Precompute lpn as a list with indices
    lpn_list = large_prime_numbers
    lpn_index = {v: i for i, v in enumerate(lpn_list)}
    n_lpn = len(lpn_list)
    
    # For each spn, create a bitmask of which lpn it conflicts with
    spn_to_lpn_mask = []
    for i in range(n_spn):
        mask = 0
        for v in spn_to_lpn_conflict[i]:
            mask |= (1 << lpn_index[v])
        spn_to_lpn_mask.append(mask)
    
    # For a given selected_spn_mask, the conflicting lpn mask is the OR of all
    # spn_to_lpn_mask[i] for selected i
    # Available lpn sum = sum of lpn_list[j] for j where bit j is not in conflict_mask
    
    # Precompute prefix sums of lpn weights sorted by index for fast sum computation?
    # Or just iterate. n_lpn is at most ~50, so iterating is fine.
    
    def lpn_available_sum(conflict_mask_lpn):
        total = 0
        for j in range(n_lpn):
            if not (conflict_mask_lpn & (1 << j)):
                total += lpn_list[j]
        return total
    
    # Branch and bound
    # We'll use a recursive function with a time check
    # To avoid Python recursion limits, we'll use an explicit stack
    
    # Actually, for n_spn up to ~100, even a well-optimized B&B might be too slow.
    # Let's limit the search to a reasonable depth or use a heuristic.
    
    # Alternative approach: since the problem is small enough, let's just do a
    # thorough local search from the greedy solution.
    
    # Local search: try to improve by:
    # 1. Adding an unselected number if it doesn't conflict
    # 2. Swapping: remove one selected, add one or more unselected
    # 3. For spn: try different assignments
    
    # Let's do a more systematic search: iterate over all possible subsets of
    # "critical" small prime numbers and greedily fill the rest.
    
    # The critical numbers are those with multiple small prime factors, as they
    # create the most conflicts.
    
    # Actually, let's just do a proper B&B with a time limit and good bounds.
    # The key is that the lpn upper bound is tight when few spn are selected.
    
    # Let's limit B&B to cases where n_spn is small enough, otherwise rely on greedy
    # with local search.
    
    if n_spn <= 30:
        # Do B&B
        # Stack: (candidate_mask, selected_mask, spn_sum, conflict_lpn_mask)
        # We process in order of decreasing spn weight
        
        # Initial state
        init_candidate = (1 << n_spn) - 1
        stack = [(init_candidate, 0, 0, 0)]
        # conflict_lpn_mask: bitmask of lpn that conflict with selected spn
        
        nodes_explored = 0
        max_nodes = 500000  # limit to avoid timeout
        
        while stack:
            if time.time() > deadline - 5:
                break
            if nodes_explored > max_nodes:
                break
            
            candidate_mask, selected_mask, spn_sum, conflict_lpn_mask = stack.pop()
            nodes_explored += 1
            
            if candidate_mask == 0:
                # All decided
                total = 1 + spn_sum + lpn_available_sum(conflict_lpn_mask)
                if total > best_total:
                    best_total = total
                    # Reconstruct solution
                    sel = [1]
                    for i in range(n_spn):
                        if selected_mask & (1 << i):
                            sel.append(small_prime_numbers[i])
                    # Add available lpn
                    for j in range(n_lpn):
                        if not (conflict_lpn_mask & (1 << j)):
                            sel.append(lpn_list[j])
                    best_selected = sel
                continue
            
            # Upper bound: spn_sum + sum of candidate spn weights + lpn_available_sum(conflict_lpn_mask)
            # The lpn sum is already an upper bound (ignoring future spn conflicts)
            # Sum of candidate spn weights:
            cand_spn_sum = 0
            temp_mask = candidate_mask
            while temp_mask:
                i = (temp_mask & -temp_mask).bit_length() - 1
                cand_spn_sum += small_prime_numbers[i]
                temp_mask &= temp_mask - 1
            
            ub = 1 + spn_sum + cand_spn_sum + lpn_available_sum(conflict_lpn_mask)
            if ub <= best_total:
                continue
            
            # Branch on the highest-weight candidate
            # Find the highest-weight vertex in candidate_mask
            best_i = -1
            best_val = -1
            temp_mask = candidate_mask
            while temp_mask:
                i = (temp_mask & -temp_mask).bit_length() - 1
                if small_prime_numbers[i] > best_val:
                    best_val = small_prime_numbers[i]
                    best_i = i
                temp_mask &= temp_mask - 1
            
            if best_i < 0:
                continue
            
            # Branch 1: include best_i
            new_selected_mask = selected_mask | (1 << best_i)
            new_conflict_lpn_mask = conflict_lpn_mask | spn_to_lpn_mask[best_i]
            # Remove best_i and its neighbors from candidates
            new_candidate_mask = candidate_mask & ~(1 << best_i)
            # Remove neighbors of best_i
            neighbors = spn_conflict[best_i]
            new_candidate_mask &= ~neighbors
            stack.append((new_candidate_mask, new_selected_mask, spn_sum + small_prime_numbers[best_i], new_conflict_lpn_mask))
            
            # Branch 2: exclude best_i
            new_candidate_mask2 = candidate_mask & ~(1 << best_i)
            stack.append((new_candidate_mask2, selected_mask, spn_sum, conflict_lpn_mask))
    
    # Verify the result
    for i in range(len(best_selected)):
        for j in range(i + 1, len(best_selected)):
            if math.gcd(best_selected[i], best_selected[j]) != 1:
                best_selected = [1]
                best_total = 1
                break
    
    return [int(x) for x in best_selected]
