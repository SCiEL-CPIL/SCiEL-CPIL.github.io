import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    all_numbers = list(range(2, 129))
    
    # Precompute prime factorization for each number in [2, 128]
    # primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def factorize(n):
        """Return prime exponent vector as a tuple."""
        exponents = [0] * num_primes
        for i, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                exponents[i] += 1
                n //= p
        if n > 1:
            # n is prime
            exponents[prime_to_idx[n]] += 1
        return tuple(exponents)
    
    # Precompute factorizations
    factorizations = {}
    for n in range(2, 129):
        factorizations[n] = factorize(n)
    
    def add_vectors(a, b):
        """Add two exponent vectors."""
        return tuple(a[i] + b[i] for i in range(num_primes))
    
    def sub_vectors(a, b):
        """Subtract two exponent vectors."""
        return tuple(a[i] - b[i] for i in range(num_primes))
    
    def check_valid(seq):
        """
        Check if strictly increasing sequence 'seq' has distinct products
        for all non-empty consecutive blocks.
        """
        n = len(seq)
        if n == 0:
            return True
        
        # Compute prefix products as exponent vectors
        S = [tuple([0]*num_primes)] * (n + 1)
        for i in range(n):
            S[i+1] = add_vectors(S[i], factorizations[seq[i]])
            
        seen_products = set()
        # Products of blocks [i+1, j] are S[j] - S[i] for 0 <= i < j <= n
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod_vec = sub_vectors(S[j], S[i])
                if prod_vec in seen_products:
                    return False
                seen_products.add(prod_vec)
        return True

    def get_prefix_products(seq):
        n = len(seq)
        S = [tuple([0]*num_primes)] * (n + 1)
        for i in range(n):
            S[i+1] = add_vectors(S[i], factorizations[seq[i]])
        return S

    def get_product_set(seq):
        S = get_prefix_products(seq)
        n = len(seq)
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                seen_products.add(sub_vectors(S[j], S[i]))
        return seen_products

    def try_insert_fast(seq, c, S, prod_set):
        """
        Try to insert c into seq.
        Returns (new_seq, new_S, new_prod_set) if valid, else None.
        """
        n = len(seq)
        # Find insertion index
        idx = 0
        for i in range(n):
            if seq[i] < c:
                idx = i + 1
            else:
                break
        
        new_seq = seq[:idx] + [c] + seq[idx:]
        
        # Calculate new products involving c
        # New products are c * (product of seq[i..j-1]) for 0 <= i <= idx, idx <= j <= n
        # In exponent vector form: factorizations[c] + (S[j] - S[i])
        c_vec = factorizations[c]
        new_products_to_check = []
        for i in range(idx + 1):
            for j in range(idx, n + 1):
                # Product of seq[i..j-1] is S[j] - S[i]
                # Full product is c_vec + (S[j] - S[i])
                base = sub_vectors(S[j], S[i])
                val = add_vectors(c_vec, base)
                if val in prod_set:
                    return None
                new_products_to_check.append(val)
        
        # If all valid, update state
        new_prod_set = set(prod_set)
        new_prod_set.update(new_products_to_check)
        
        # Update prefix products
        new_S = [tuple([0]*num_primes)] * (n + 2)
        for i in range(idx + 1):
            new_S[i] = S[i]
        for i in range(idx + 1, n + 2):
            new_S[i] = add_vectors(S[i-1], c_vec)
            
        return new_seq, new_S, new_prod_set

    def greedy_build(order):
        seq = []
        S = get_prefix_products(seq)
        prod_set = get_product_set(seq)
        used = set(seq)
        
        for c in order:
            if c in used:
                continue
            res = try_insert_fast(seq, c, S, prod_set)
            if res is not None:
                seq, S, prod_set = res
                used = set(seq)
        
        return seq

    # CSP-style backtracking search with pruning
    def backtrack_search(seq, S, prod_set, remaining, best_seq_ref):
        """
        Chronological backtracking search.
        seq: current sequence (sorted)
        S: prefix products (exponent vectors)
        prod_set: set of all block products (exponent vectors)
        remaining: list of candidate numbers to try (sorted)
        best_seq_ref: list containing the best sequence found so far
        """
        if time.time() > deadline:
            return
        
        if len(seq) > len(best_seq_ref[0]):
            best_seq_ref[0] = seq[:]
        
        if not remaining:
            return
        
        # Try each candidate in remaining
        for i, c in enumerate(remaining):
            if time.time() > deadline:
                return
            
            # Try inserting c into seq
            res = try_insert_fast(seq, c, S, prod_set)
            if res is not None:
                new_seq, new_S, new_prod_set = res
                # Recurse with c removed from remaining
                new_remaining = remaining[:i] + remaining[i+1:]
                backtrack_search(new_seq, new_S, new_prod_set, new_remaining, best_seq_ref)
            
            # Pruning: if we can't beat the best even by adding all remaining, skip
            # (This is a weak bound; we could use better heuristics)

    best_seq = []
    best_len = 0
    
    # Try multiple orders to find a good starting point
    orders = [
        all_numbers,
        sorted(all_numbers, reverse=True),
        sorted(all_numbers, key=lambda x: math.gcd(x, 128)),
        sorted(all_numbers, key=lambda x: len([i for i in range(2, x) if x % i == 0]))
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        seq = greedy_build(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Local Search
    def local_search(seq):
        current_seq = seq[:]
        current_S = get_prefix_products(current_seq)
        current_prod_set = get_product_set(current_seq)
        used = set(current_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        improved = True
        while improved:
            improved = False
            
            # Try insertions
            for c in sorted(remaining):
                if time.time() > deadline:
                    break
                res = try_insert_fast(current_seq, c, current_S, current_prod_set)
                if res is not None:
                    current_seq, current_S, current_prod_set = res
                    used = set(current_seq)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            
            if time.time() > deadline:
                break

            # If no insertion worked, try swaps
            if remaining:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    if not remaining:
                        break
                    
                    i = random.randint(0, len(current_seq) - 1)
                    x_to_remove = current_seq[i]
                    c_to_add = random.choice(remaining)
                    
                    new_set = set(current_seq)
                    new_set.remove(x_to_remove)
                    new_set.add(c_to_add)
                    new_seq = sorted(new_set)
                    
                    if check_valid(new_seq):
                        current_seq = new_seq
                        current_S = get_prefix_products(new_seq)
                        current_prod_set = get_product_set(new_seq)
                        used = set(current_seq)
                        remaining = [x for x in all_numbers if x not in used]
                        improved = True
                        break

        return current_seq

    if best_seq:
        best_seq = local_search(best_seq)

    # CSP backtracking search from current best
    if time.time() < deadline - 5:
        best_ref = [best_seq[:]]
        current_S = get_prefix_products(best_seq)
        current_prod_set = get_product_set(best_seq)
        used = set(best_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        # Sort remaining by a heuristic: prefer numbers that are "compatible" with current seq
        # A simple heuristic: prefer numbers with fewer prime factors
        remaining.sort(key=lambda x: sum(factorizations[x]))
        
        backtrack_search(best_seq, current_S, current_prod_set, remaining, best_ref)
        if len(best_ref[0]) > len(best_seq):
            best_seq = best_ref[0][:]

    # Final Polish with multiple random restarts
    for _ in range(50):
        if time.time() > deadline:
            break
        random.shuffle(all_numbers)
        new_seq = greedy_build(all_numbers)
        new_seq = local_search(new_seq)
        
        if len(new_seq) > len(best_seq):
            best_seq = new_seq[:]

    # Final Verification
    if not check_valid(best_seq):
        best_seq = [2, 3, 5, 7]
        if not check_valid(best_seq):
            best_seq = [2]

    return best_seq
