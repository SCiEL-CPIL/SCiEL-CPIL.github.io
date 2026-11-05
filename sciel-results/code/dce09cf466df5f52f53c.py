"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute prime factorizations for numbers 2..128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def factorize(n):
        exponents = [0] * num_primes
        for i, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                exponents[i] += 1
                n //= p
        if n > 1:
            if n in prime_to_idx:
                exponents[prime_to_idx[n]] += 1
        return tuple(exponents)
    
    factor_cache = {}
    for i in range(2, 129):
        factor_cache[i] = factorize(i)
    
    def add_vecs(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
    
    def compute_tails_from_tuple(seq_tuple, x):
        tails = []
        vec = factor_cache[x]
        tails.append(vec)
        for a in reversed(seq_tuple):
            vec = add_vecs(vec, factor_cache[a])
            tails.append(vec)
        return tails
    
    def check_collision(tails, product_set):
        for t in tails:
            if t in product_set:
                return True
        return False
    
    def compute_all_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            vec = ()
            for j in range(i, n):
                vec = add_vecs(vec, factor_cache[seq[j]])
                prods.add(vec)
        return prods
    
    def try_insert(seq, prods, x):
        """Try to insert x into seq at any valid position. Returns (new_seq, new_prods) or (None, None)."""
        n = len(seq)
        # Find valid insertion positions: seq[i-1] < x < seq[i] (or at ends)
        valid_positions = []
        for pos in range(n + 1):
            if pos > 0 and seq[pos - 1] >= x:
                continue
            if pos < n and seq[pos] <= x:
                continue
            valid_positions.append(pos)
        
        for pos in valid_positions:
            # Check if inserting x at pos creates any product collision
            # New products are: products ending at pos (new element) and products starting at pos
            # We need to check all consecutive blocks that include the new element at position pos
            
            # Build test sequence
            test_seq = seq[:pos] + [x] + seq[pos:]
            test_n = len(test_seq)
            
            # All new products involve index pos
            # Check for collisions with existing products and among themselves
            new_prods = set()
            valid = True
            
            # Products that start before or at pos and end at pos or after
            for i in range(pos + 1):  # start from 0 to pos
                vec = ()
                for j in range(i, test_n):
                    vec = add_vecs(vec, factor_cache[test_seq[j]])
                    if j >= pos:
                        # This product involves the new element
                        if vec in prods or vec in new_prods:
                            valid = False
                            break
                        new_prods.add(vec)
                if not valid:
                    break
            
            if valid:
                new_prods_full = prods | new_prods
                return test_seq, new_prods_full
        
        return None, None
    
    def remove_element(seq, prods, pos):
        """Remove element at pos, return (new_seq, new_prods)."""
        new_seq = seq[:pos] + seq[pos+1:]
        new_prods = compute_all_products(new_seq)
        return new_seq, new_prods
    
    # Multi-start greedy with different orderings
    def run_greedy(order):
        values = []
        product_set = set()
        for x in order:
            if time.time() > deadline:
                break
            seq_tuple = tuple(values)
            tails = compute_tails_from_tuple(seq_tuple, x)
            if not check_collision(tails, product_set):
                values.append(x)
                product_set.update(tails)
        return values, product_set
    
    # Generate diverse orderings
    orderings = []
    # Standard ascending
    orderings.append(list(range(2, 129)))
    # Primes first, then composites
    primes_list = [p for p in range(2, 129) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
    composites_list = [x for x in range(2, 129) if x not in primes_list]
    orderings.append(primes_list + composites_list)
    # Alternating small/large
    half = 65
    small = list(range(2, half+1))
    large = list(range(half+1, 129))
    alt = []
    si, li = 0, 0
    while si < len(small) or li < len(large):
        if si < len(small):
            alt.append(small[si])
            si += 1
        if li < len(large):
            alt.append(large[li])
            li += 1
    orderings.append(alt)
    # Descending within pairs
    pairs_order = []
    for i in range(2, 129, 2):
        if i + 1 <= 128:
            pairs_order.extend([i+1, i])
        else:
            pairs_order.append(i)
    orderings.append(pairs_order)
    # Even then odd
    evens = list(range(2, 129, 2))
    odds = list(range(3, 129, 2))
    orderings.append(evens + odds)
    # Odd then even
    orderings.append(odds + evens)
    # Multiples of 2 first, then 3, then others
    mod2 = [x for x in range(2, 129) if x % 2 == 0]
    mod3 = [x for x in range(2, 129) if x % 3 == 0 and x % 2 != 0]
    rest = [x for x in range(2, 129) if x % 2 != 0 and x % 3 != 0]
    orderings.append(mod2 + mod3 + rest)
    
    best_seq = []
    best_len = 0
    best_prods = set()
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        seq, prods = run_greedy(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            best_prods = prods
    
    # Local search: try insertions and removals
    if time.time() < deadline - 15:
        current = list(best_seq)
        current_prods = compute_all_products(current)
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(current)
            
            # Try inserting missing numbers
            current_set = set(current)
            missing = [x for x in range(2, 129) if x not in current_set]
            
            for x in missing:
                if time.time() > deadline - 3:
                    break
                new_seq, new_prods = try_insert(current, current_prods, x)
                if new_seq is not None:
                    current = new_seq
                    current_prods = new_prods
                    improved = True
                    # After insertion, try removing some elements to make room
                    break
            
            if not improved and time.time() < deadline - 8:
                # Try removing an element to potentially allow a better insertion
                for pos in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    old_val = current[pos]
                    new_seq, new_prods = remove_element(current, current_prods, pos)
                    
                    # Try to insert the removed value elsewhere or a different value
                    # Try inserting old_val at a different position
                    inserted = False
                    for x in [old_val] + [m for m in missing if m != old_val][:20]:
                        if time.time() > deadline - 4:
                            break
                        test_seq, test_prods = try_insert(new_seq, new_prods, x)
                        if test_seq is not None:
                            current = test_seq
                            current_prods = test_prods
                            improved = True
                            inserted = True
                            break
                    
                    if inserted:
                        break
    
    # Final extension pass
    if time.time() < deadline - 3:
        current = list(best_seq)
        current_prods = compute_all_products(current)
        last_val = current[-1] if current else 1
        for x in range(last_val + 1, 129):
            if time.time() > deadline - 2:
                break
            seq_tuple = tuple(current)
            tails = compute_tails_from_tuple(seq_tuple, x)
            if not check_collision(tails, current_prods):
                current.append(x)
                current_prods.update(tails)
        if len(current) > len(best_seq):
            best_seq = current
            best_len = len(current)
    
    return best_seq
# EVOLVE-BLOCK-END
