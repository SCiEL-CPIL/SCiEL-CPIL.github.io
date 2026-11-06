import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    all_numbers = list(range(2, 129))
    primes = [p for p in all_numbers if is_prime(p)]
    composites = [c for c in all_numbers if not is_prime(c)]

    def num_distinct_prime_factors(n):
        count = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count

    def compute_suffix_products(seq):
        """Compute suffix products: suffixes[i] = seq[i]*seq[i+1]*...*seq[-1], suffixes[-1]=1"""
        n = len(seq)
        suffixes = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffixes[i] = suffixes[i + 1] * seq[i]
        return suffixes

    def compute_all_products(seq):
        """Compute all consecutive block products as a set."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid_fast(seq):
        """Check if all consecutive block products are distinct."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    # Start with all primes (known valid, length 30)
    best_seq = sorted(primes)
    best_len = len(best_seq)

    # Fast incremental check: given a sequence and its product set, check if appending x is valid
    def can_append(seq, all_prods, suffixes, x):
        """Check if we can append x to seq. Returns (valid, new_all_prods, new_suffixes)"""
        n = len(seq)
        # New products are: x, x*seq[n-1], x*seq[n-1]*seq[n-2], ...
        # These are x * suffixes[i] for i in range(n+1) where suffixes[i] = seq[i]*...*seq[-1]
        # suffixes has length n+1: suffixes[0]=product of all, suffixes[n]=1
        # New products: x * suffixes[i] for i in 0..n
        new_prods = set()
        for i in range(n + 1):
            p = x * suffixes[i]
            if p in all_prods or p in new_prods:
                return False, None, None
            new_prods.add(p)
        return True, all_prods | new_prods, [x] + [x * s for s in suffixes]

    # Fast check for insertion at position pos
    def can_insert_at(seq, all_prods, pos, x):
        """Check if inserting x at position pos keeps validity. Returns (valid, new_all_prods)"""
        n = len(seq)
        # Products that change: those that include x
        # Before: products of blocks in seq
        # After: products of blocks in new_seq
        # New products: any consecutive block that includes x
        # These are: (product of seq[a:pos]) * x * (product of seq[pos:b]) for 0<=a<pos, pos<=b<=n
        # Plus: x alone (a=pos-1... wait, let me think more carefully)
        
        # Actually, let's just recompute all products for the new sequence
        # But we can be smart: only products that include the new element x are new
        # Products NOT including x are unchanged
        # Products including x: for each (i,j) where i <= pos <= j in the new sequence
        # In the new sequence, x is at index pos
        # Block from i to j includes x iff i <= pos <= j
        # Product = (prod seq[i:pos]) * x * (prod seq[pos:j])
        # where prod seq[i:pos] = seq[i]*...*seq[pos-1] (empty if i==pos)
        # and prod seq[pos:j] = seq[pos]*...*seq[j-1] (empty if j==pos)
        
        # Let's compute prefix products of seq and suffix products
        # prefix[k] = seq[0]*...*seq[k-1], prefix[0]=1
        # suffix[k] = seq[k]*...*seq[n-1], suffix[n]=1
        
        prefix = [1] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] * seq[i]
        
        suffix = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] * seq[i]
        
        # New products including x:
        # For i in 0..pos, j in pos..n (in new sequence coordinates, x is at pos)
        # Left part: seq[i:pos] product = prefix[pos] / prefix[i] if i < pos, else 1
        # Right part: seq[pos:j] product = suffix[pos] / suffix[j] if j > pos, else 1
        # But we need to be careful with division - use the products directly
        
        new_prods = set()
        for i in range(pos + 1):  # i from 0 to pos (start of block in old seq, or pos means x alone on left)
            left_prod = 1
            if i < pos:
                left_prod = prefix[pos] // prefix[i]
            # right part: from pos to j-1 in old seq, j from pos to n
            for j in range(pos, n + 1):
                right_prod = 1
                if j > pos:
                    right_prod = suffix[pos] // suffix[j]
                p = left_prod * x * right_prod
                if p in all_prods or p in new_prods:
                    return False, None
                new_prods.add(p)
        
        return True, all_prods | new_prods

    # Greedy insertion with fast checking
    def greedy_build(prime_list, composite_order, all_prods=None, suffixes=None):
        """Build sequence by greedily inserting composites."""
        seq = list(prime_list)
        if all_prods is None:
            all_prods = compute_all_products(seq)
        if suffixes is None:
            suffixes = compute_suffix_products(seq)
        
        for c in composite_order:
            if time.time() > deadline:
                break
            
            # Binary search for insertion point
            lo, hi = 0, len(seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if seq[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            insert_pos = lo
            
            if insert_pos < len(seq) and seq[insert_pos] == c:
                continue
            
            # Try to insert
            valid, new_prods = can_insert_at(seq, all_prods, insert_pos, c)
            if valid:
                seq = seq[:insert_pos] + [c] + seq[insert_pos:]
                all_prods = new_prods
                suffixes = compute_suffix_products(seq)
        
        return seq, all_prods, suffixes

    # Try multiple greedy passes with different orderings
    orders = []
    orders.append(sorted(composites, key=lambda x: num_distinct_prime_factors(x)))
    orders.append(sorted(composites, key=lambda x: -num_distinct_prime_factors(x)))
    orders.append(sorted(composites, key=lambda x: x))
    orders.append(sorted(composites, key=lambda x: -x))
    orders.append(sorted(composites, key=lambda x: (num_distinct_prime_factors(x), x)))
    orders.append(sorted(composites, key=lambda x: (num_distinct_prime_factors(x), -x)))
    orders.append(sorted(composites, key=lambda x: (x % 7, num_distinct_prime_factors(x))))
    orders.append(sorted(composites, key=lambda x: (x % 11, num_distinct_prime_factors(x))))

    for order in orders:
        if time.time() > deadline:
            break
        seq, _, _ = greedy_build(primes, order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    # Try removing one prime and adding composites
    if time.time() < deadline - 5.0:
        for i in range(len(primes)):
            if time.time() > deadline:
                break
            removed = primes[i]
            seq_without = primes[:i] + primes[i+1:]
            for order in orders[:4]:
                if time.time() > deadline:
                    break
                seq, _, _ = greedy_build(seq_without, order)
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq

    # Try a local search: swap elements or try different insertions
    if time.time() < deadline - 3.0:
        # Try to improve by trying all possible single insertions into current best
        current = list(best_seq)
        current_prods = compute_all_products(current)
        
        # Try inserting each composite at each valid position
        for c in composites:
            if time.time() > deadline:
                break
            if c in current:
                continue
            for pos in range(len(current) + 1):
                # Check if strictly increasing
                if pos > 0 and current[pos - 1] >= c:
                    continue
                if pos < len(current) and current[pos] <= c:
                    continue
                
                valid, new_prods = can_insert_at(current, current_prods, pos, c)
                if valid:
                    new_seq = current[:pos] + [c] + current[pos:]
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                    break  # Only try first valid position for this c

    return sorted(best_seq)
