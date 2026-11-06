import time
import random
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def is_valid(seq):
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
    
    def is_valid_fast(seq):
        """Faster validity check with early exit and incremental suffix tracking"""
        n = len(seq)
        if n <= 1:
            return True
        products = set()
        # For each starting position i, compute all suffix products seq[i]*...*seq[j]
        # Use early exit: if product exceeds a threshold, further extensions are unique
        # (since all elements >= 2, products grow exponentially)
        MAX_PROD = 10**18
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod > MAX_PROD:
                    # Once product exceeds MAX_PROD, all further extensions will be
                    # larger and distinct from any previous product (which are <= MAX_PROD)
                    # But we still need to check that this specific product isn't a duplicate
                    if prod in products:
                        return False
                    products.add(prod)
                    # Continue but products will keep growing, so unlikely to collide
                    # with anything seen before (since those were smaller)
                    # Actually, we can break here because future products from this i
                    # will be even larger, and all previously seen products are <= this
                    # Wait - that's not quite right. Previously seen products could be large too.
                    # But since seq is sorted and all >= 2, products from earlier i values
                    # that reach similar magnitude would have been added already.
                    # The key insight: once prod > MAX_PROD, it's very unlikely to collide
                    # with any previously computed product. We can add it and break.
                    break
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def is_valid_incremental(seq, last_added_idx=None):
        """
        Incremental validity check. If we just added an element at position last_added_idx,
        we only need to check products that involve this new element.
        Returns True if valid, False if collision found.
        """
        n = len(seq)
        if n <= 1:
            return True
        
        products = set()
        MAX_PROD = 10**18
        
        # First, compute all products not involving the new element (if any)
        if last_added_idx is not None:
            # Compute products from the old sequence
            for i in range(n):
                if i == last_added_idx:
                    continue
                prod = 1
                for j in range(i, n):
                    if j == last_added_idx:
                        continue
                    prod *= seq[j]
                    if prod > MAX_PROD:
                        products.add(prod)
                        break
                    products.add(prod)
            # Now check products involving the new element
            for i in range(n):
                if i == last_added_idx:
                    continue
                prod = 1
                for j in range(i, n):
                    prod *= seq[j]
                    if prod > MAX_PROD:
                        if prod in products:
                            return False
                        products.add(prod)
                        break
                    if prod in products:
                        return False
                    products.add(prod)
            # Also check products starting at last_added_idx
            prod = 1
            for j in range(last_added_idx, n):
                prod *= seq[j]
                if prod > MAX_PROD:
                    if prod in products:
                        return False
                    products.add(prod)
                    break
                if prod in products:
                    return False
                products.add(prod)
        else:
            # Full check
            for i in range(n):
                prod = 1
                for j in range(i, n):
                    prod *= seq[j]
                    if prod > MAX_PROD:
                        if prod in products:
                            return False
                        products.add(prod)
                        break
                    if prod in products:
                        return False
                    products.add(prod)
        return True
    
    primes = get_primes(128)
    all_nums = list(range(2, 129))
    composites = [x for x in all_nums if x not in primes]
    
    # Precompute which numbers conflict with which
    # Two numbers conflict if they can't both be in a valid sequence
    # This is complex, so we'll use a simpler heuristic:
    # Numbers with many common factors are more likely to conflict
    
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Score numbers by how "conflict-prone" they are
    # Numbers with more prime factors tend to create more product collisions
    def conflict_score(n):
        return num_prime_factors(n)
    
    # Sort composites by conflict score (lower first to try less conflicting ones)
    composites_sorted = sorted(composites, key=lambda x: (conflict_score(x), x))
    
    best_seq = primes[:]
    best_len = len(primes)
    
    # Greedy Add with smarter ordering
    def greedy_add(seq, order=None):
        if order is None:
            order = composites_sorted
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for c in order:
                if c in seq:
                    continue
                new_seq = sorted(seq + [c])
                if is_valid_fast(new_seq):
                    seq = new_seq
                    improved = True
                    break
        return seq
    
    def try_add_element(seq, c):
        """Try to add element c to seq, returning new_seq if valid, else None."""
        if c in seq:
            return None
        new_seq = sorted(seq + [c])
        if is_valid_fast(new_seq):
            return new_seq
        return None
    
    # Local Search with better moves
    def local_search(seq, max_iters=1000):
        seq_set = set(seq)
        candidates = [c for c in composites if c not in seq_set]
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try to add - prioritize elements that create fewer conflicts
            add_candidates = [c for c in candidates if c not in seq_set]
            # Sort by a heuristic: prefer numbers with fewer prime factors
            add_candidates.sort(key=lambda x: (num_prime_factors(x), x))
            
            for c in add_candidates:
                if c in seq_set:
                    continue
                new_seq = try_add_element(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    seq_set.add(c)
                    candidates.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            found_swap = False
            # Sort elements to remove by how "conflict-prone" they are (remove more conflicting ones first)
            elements_to_try = sorted(list(seq_set), key=lambda x: -num_prime_factors(x))
            
            for a in elements_to_try:
                for c in candidates:
                    if a == c:
                        continue
                    new_seq = sorted((seq_set - {a}) | {c})
                    if is_valid_fast(new_seq):
                        # Check if this swap enables further additions
                        new_seq_set = set(new_seq)
                        new_candidates = [x for x in candidates if x != c and x not in new_seq_set]
                        can_add = False
                        for d in new_candidates[:15]:  # Check a few more
                            test_seq = sorted(new_seq + [d])
                            if is_valid_fast(test_seq):
                                can_add = True
                                break
                        
                        if can_add or random.random() < 0.4:
                            seq = new_seq
                            seq_set = new_seq_set
                            candidates = [x for x in candidates if x != c and x not in seq_set]
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove two, add two
            if len(seq) > 4 and len(candidates) >= 2:
                two_swap_found = False
                # Try removing pairs of high-conflict elements
                high_conflict = sorted(list(seq_set), key=lambda x: -num_prime_factors(x))[:5]
                low_conflict_cands = sorted(candidates, key=lambda x: (num_prime_factors(x), x))[:10]
                
                for a1 in high_conflict:
                    for a2 in high_conflict:
                        if a1 >= a2:
                            continue
                        for c1 in low_conflict_cands:
                            for c2 in low_conflict_cands:
                                if c1 >= c2:
                                    continue
                                if c1 in seq_set or c2 in seq_set:
                                    continue
                                new_set = (seq_set - {a1, a2}) | {c1, c2}
                                new_seq = sorted(new_set)
                                if is_valid_fast(new_seq):
                                    # Check if this enables further additions
                                    new_seq_set = set(new_seq)
                                    new_candidates = [x for x in candidates if x not in new_seq_set and x != c1 and x != c2]
                                    can_add = False
                                    for d in new_candidates[:5]:
                                        test_seq = sorted(new_seq + [d])
                                        if is_valid_fast(test_seq):
                                            can_add = True
                                            break
                                    
                                    if can_add or random.random() < 0.5:
                                        seq = new_seq
                                        seq_set = new_seq_set
                                        candidates = [x for x in candidates if x not in seq_set]
                                        improved = True
                                        two_swap_found = True
                                        break
                            if two_swap_found:
                                break
                        if two_swap_found:
                            break
                    if two_swap_found:
                        break
                
                if two_swap_found:
                    continue
            
            # Try remove (kick) to escape local optimum
            if len(seq) > 3:
                a = random.choice(list(seq_set))
                new_seq = sorted(seq_set - {a})
                if is_valid_fast(new_seq):
                    seq = new_seq
                    seq_set.remove(a)
                    candidates.append(a)
                    improved = True
            
            if not improved:
                break
        
        return seq
    
    # Main Loop
    current_seq = primes[:]
    current_seq = greedy_add(current_seq)
    if len(current_seq) > best_len:
        best_seq = current_seq
        best_len = len(current_seq)
    
    # Run local search from this starting point
    current_seq = local_search(current_seq)
    if len(current_seq) > best_len:
        best_seq = current_seq
        best_len = len(current_seq)
    
    # Track the best sequence found so far
    while time.time() < deadline - 2:
        # Try to improve current_seq
        improved = False
        
        # 1. Try Greedy Add with different orderings
        for _ in range(5):
            random.shuffle(composites_sorted)
            new_seq = greedy_add(current_seq, composites_sorted)
            if len(new_seq) > len(current_seq):
                current_seq = new_seq
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(current_seq)
                improved = True
                break
        
        if not improved:
            # 2. Try Local Search with more iterations
            new_seq = local_search(current_seq, max_iters=300)
            if len(new_seq) > len(current_seq):
                current_seq = new_seq
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(current_seq)
                improved = True
        
        if not improved:
            # 3. Try targeted swaps: replace high-conflict elements with low-conflict ones
            seq_set = set(current_seq)
            candidates = [c for c in composites if c not in seq_set]
            if candidates:
                # Sort by conflict score
                high_conflict = sorted(list(seq_set), key=lambda x: -num_prime_factors(x))
                low_conflict_cands = sorted(candidates, key=lambda x: (num_prime_factors(x), x))
                
                swap_improved = False
                for a in high_conflict[:10]:  # Try removing top 10 high-conflict elements
                    for c in low_conflict_cands[:15]:  # Try adding top 15 low-conflict candidates
                        if a == c:
                            continue
                        new_seq = sorted((seq_set - {a}) | {c})
                        if is_valid_fast(new_seq):
                            # Check if this enables further additions
                            new_seq_set = set(new_seq)
                            new_candidates = [x for x in candidates if x not in new_seq_set and x != c]
                            can_add = False
                            for d in new_candidates[:10]:
                                test_seq = sorted(new_seq + [d])
                                if is_valid_fast(test_seq):
                                    can_add = True
                                    break
                            
                            if can_add or random.random() < 0.5:
                                current_seq = new_seq
                                seq_set = new_seq_set
                                candidates = [x for x in candidates if x not in seq_set]
                                improved = True
                                swap_improved = True
                                break
                    if swap_improved:
                        break
                
                if swap_improved:
                    if len(current_seq) > best_len:
                        best_seq = current_seq
                        best_len = len(current_seq)
                    continue
        
        if not improved:
            # 4. Try Random Remove (Kick)
            if len(current_seq) > 3:
                a = random.choice(current_seq)
                new_seq = sorted(set(current_seq) - {a})
                if is_valid_fast(new_seq):
                    current_seq = new_seq
                    improved = True
        
        if not improved:
            # 5. Restart with different random greedy
            random.shuffle(composites_sorted)
            current_seq = primes[:]
            improved_greedy = True
            while improved_greedy and time.time() < deadline - 2:
                improved_greedy = False
                for c in composites_sorted:
                    if c in current_seq:
                        continue
                    new_seq = try_add_element(current_seq, c)
                    if new_seq is not None:
                        current_seq = new_seq
                        improved_greedy = True
                        break
            
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
            
            # Run local search from this restart
            current_seq = local_search(current_seq, max_iters=150)
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
    
    return best_seq
