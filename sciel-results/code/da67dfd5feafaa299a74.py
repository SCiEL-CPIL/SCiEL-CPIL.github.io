import itertools
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    all_nums = list(range(2, 41))
    exp_vectors = {}
    for n in all_nums:
        f = factorize(n)
        vec = [0] * num_primes
        for p, e in f.items():
            vec[prime_to_idx[p]] = e
        exp_vectors[n] = tuple(vec)

    def check_dissociated(selected):
        """Full check: all 2^k subset products distinct."""
        n = len(selected)
        if n == 0:
            return True
        vectors = [exp_vectors[num] for num in selected]
        seen = set()
        for mask in range(1 << n):
            sum_vec = [0] * num_primes
            for i in range(n):
                if mask & (1 << i):
                    for j in range(num_primes):
                        sum_vec[j] += vectors[i][j]
            key = tuple(sum_vec)
            if key in seen:
                return False
            seen.add(key)
        return True

    def is_linearly_independent(vecs):
        """Fast check: are vectors linearly independent over Q?"""
        if not vecs:
            return True
        m = len(vecs)
        d = num_primes
        mat = [[Fraction(v[j]) for j in range(d)] for v in vecs]
        rank = 0
        for col in range(d):
            pivot = -1
            for row in range(rank, m):
                if mat[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            pivot_val = mat[rank][col]
            for j in range(col, d):
                mat[rank][j] /= pivot_val
            for row in range(m):
                if row != rank and mat[row][col] != 0:
                    factor = mat[row][col]
                    for j in range(col, d):
                        mat[row][j] -= factor * mat[rank][j]
            rank += 1
            if rank == m:
                return True
        return rank == m

    # Precompute: for each pair, check if adding one to a set containing the other causes issues
    # Actually, let's build the search directly

    # First, get a good initial solution using the per-prime approach
    selected_numbers = []
    for p in primes:
        exponents = []
        exp = 1
        while p ** exp <= 40:
            exponents.append(exp)
            exp += 1
        
        best_subset = []
        for r in range(len(exponents), -1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in itertools.combinations(exponents, r):
                subset_sums = set()
                is_diss = True
                for mask in range(1 << len(subset)):
                    s = 0
                    for i in range(len(subset)):
                        if mask & (1 << i):
                            s += subset[i]
                    if s in subset_sums:
                        is_diss = False
                        break
                    subset_sums.add(s)
                if is_diss:
                    best_subset = list(subset)
                    found = True
                    break
            if found:
                break
        
        for e in best_subset:
            selected_numbers.append(p ** e)

    best_set = selected_numbers[:]
    if not check_dissociated(best_set):
        best_set = []

    # Branch and bound search
    # We'll search over the 39 numbers (2..40)
    # Use bitmasks for efficiency
    n_total = len(all_nums)
    num_to_idx = {n: i for i, n in enumerate(all_nums)}
    idx_to_num = {i: n for n, i in num_to_idx.items()}

    # Precompute which pairs are "compatible" in the sense that they don't immediately cause a linear dependency
    # Actually for branch and bound, we need a good upper bound

    # Upper bound: maximum size of linearly independent set from remaining candidates
    # Given current selected set S, the upper bound on total size is |S| + max additional
    # that can be linearly independent with S's vectors

    def compute_upper_bound(selected_idx, available_mask):
        """Upper bound: |selected| + max additional linearly independent vectors from available."""
        if not available_mask:
            return len(selected_idx)
        
        # Get vectors of selected
        sel_vecs = [exp_vectors[idx_to_num[i]] for i in selected_idx]
        
        # Try to find max additional from available that are linearly independent with sel_vecs
        # Use greedy: try to add as many as possible
        # This is an upper bound (may overestimate)
        
        # Better: compute rank of [sel_vecs + all available vecs] - rank(sel_vecs)
        # That's the max number of additional vectors we could add (upper bound)
        
        all_vecs = sel_vecs[:]
        for i in range(n_total):
            if available_mask & (1 << i):
                all_vecs.append(exp_vectors[idx_to_num[i]])
        
        if not all_vecs:
            return len(selected_idx)
        
        # Compute rank of all_vecs
        m = len(all_vecs)
        d = num_primes
        mat = [[Fraction(v[j]) for j in range(d)] for v in all_vecs]
        rank = 0
        for col in range(d):
            pivot = -1
            for row in range(rank, m):
                if mat[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            pivot_val = mat[rank][col]
            for j in range(col, d):
                mat[rank][j] /= pivot_val
            for row in range(m):
                if row != rank and mat[row][col] != 0:
                    factor = mat[row][col]
                    for j in range(col, d):
                        mat[row][j] -= factor * mat[rank][j]
            rank += 1
        
        # The max additional is rank - len(selected_idx) but capped by number available
        n_avail = bin(available_mask).count('1')
        additional = min(rank - len(selected_idx), n_avail)
        if additional < 0:
            additional = 0
        return len(selected_idx) + additional

    # Branch and bound
    best_found = len(best_set)
    best_solution = best_set[:]
    
    # Order candidates by some heuristic (e.g., numbers with fewer prime factors first, or by value)
    # Actually, let's order by: prefer numbers that are prime powers (more "independent")
    candidate_order = sorted(all_nums, key=lambda x: (len(factorize(x)), x))
    order_to_idx = [num_to_idx[n] for n in candidate_order]
    
    # Recursive branch and bound
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(pos, selected_idx, available_mask):
        nonlocal best_found, best_solution
        
        if time.time() > deadline:
            return
        
        # Update best
        if len(selected_idx) > best_found:
            best_found = len(selected_idx)
            best_solution = [idx_to_num[i] for i in selected_idx]
        
        if pos >= len(candidate_order):
            return
        
        # Upper bound check
        ub = compute_upper_bound(selected_idx, available_mask)
        if ub <= best_found:
            return
        
        idx = order_to_idx[pos]
        num = idx_to_num[idx]
        
        # Option 1: Include this number
        new_selected = selected_idx + [idx]
        new_vecs = [exp_vectors[idx_to_num[i]] for i in new_selected]
        if is_linearly_independent(new_vecs):
            if check_dissociated([idx_to_num[i] for i in new_selected]):
                new_avail = available_mask & ~(1 << idx)
                branch_and_bound(pos + 1, new_selected, new_avail)
        
        # Option 2: Exclude this number
        if time.time() > deadline:
            return
        branch_and_bound(pos + 1, selected_idx, available_mask)

    # Start branch and bound
    initial_available = 0
    for i in range(n_total):
        initial_available |= (1 << i)
    
    branch_and_bound(0, [], initial_available)

    # Also try: take best_set and try to improve via swaps
    if time.time() < deadline - 5 and best_set:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(best_set)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for candidate in all_nums:
                    if candidate in best_set:
                        continue
                    new_set = best_set[:i] + best_set[i+1:] + [candidate]
                    if len(set(new_set)) != len(new_set):
                        continue
                    new_vecs = [exp_vectors[x] for x in new_set]
                    if is_linearly_independent(new_vecs):
                        if check_dissociated(new_set):
                            best_set = new_set
                            best_found = len(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Final verification
    if best_solution and check_dissociated(best_solution):
        if all(2 <= x <= 40 for x in best_solution) and len(set(best_solution)) == len(best_solution):
            return [int(x) for x in best_solution]

    if best_set and check_dissociated(best_set):
        if all(2 <= x <= 40 for x in best_set) and len(set(best_set)) == len(best_set):
            return [int(x) for x in best_set]

    return [int(x) for x in selected_numbers]
