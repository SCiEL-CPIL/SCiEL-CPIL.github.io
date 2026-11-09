import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    all_numbers = list(range(2, 129))
    
    # Precompute prime factorization for each number in [2, 128]
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
            exponents[prime_to_idx[n]] += 1
        return tuple(exponents)
    
    # Precompute factorizations
    factorizations = {}
    for n in range(2, 129):
        factorizations[n] = factorize(n)
    
    # Precompute "weight" for each number - sum of exponents (lower is lighter)
    number_weights = {}
    for n in range(2, 129):
        number_weights[n] = sum(factorizations[n])
    
    def add_vectors(a, b):
        """Add two exponent vectors."""
        return tuple(a[i] + b[i] for i in range(num_primes))
    
    def sub_vectors(a, b):
        """Subtract two exponent vectors."""
        return tuple(a[i] - b[i] for i in range(num_primes))
    
    def get_prefix_products(seq):
        n = len(seq)
        S = [tuple([0]*num_primes)] * (n + 1)
        for i in range(n):
            S[i+1] = add_vectors(S[i], factorizations[seq[i]])
        return S
    
    def get_product_set(seq, S=None):
        if S is None:
            S = get_prefix_products(seq)
        n = len(seq)
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                seen_products.add(sub_vectors(S[j], S[i]))
        return seen_products
    
    def check_valid(seq):
        """Check if strictly increasing sequence has distinct products for all blocks."""
        n = len(seq)
        if n == 0:
            return True
        
        S = get_prefix_products(seq)
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod_vec = sub_vectors(S[j], S[i])
                if prod_vec in seen_products:
                    return False
                seen_products.add(prod_vec)
        return True
    
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
        
        # Check if c is already in seq
        if idx < n and seq[idx] == c:
            return None
        
        new_seq = seq[:idx] + [c] + seq[idx:]
        
        # Calculate new products involving c
        # New products are c * (product of seq[i..j-1]) for 0 <= i <= idx, idx <= j <= n
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
    
    def local_search(seq, max_swaps=30):
        """
        Improve sequence by insertions and targeted swaps.
        """
        current_seq = seq[:]
        current_S = get_prefix_products(current_seq)
        current_prod_set = get_product_set(current_seq)
        used = set(current_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        improved = True
        swap_attempts = 0
        while improved and swap_attempts < max_swaps:
            improved = False
            
            # Try insertions with heuristic ordering
            # Score candidates by: (new product count, weight) - prefer minimal impact and lighter numbers
            scored_insertions = []
            for c in remaining:
                if time.time() > deadline:
                    break
                res = try_insert_fast(current_seq, c, current_S, current_prod_set)
                if res is not None:
                    new_seq, new_S, new_prod_set = res
                    new_count = len(new_prod_set) - len(current_prod_set)
                    # Prefer fewer new products, then lighter numbers
                    scored_insertions.append((new_count, number_weights[c], c, new_seq, new_S, new_prod_set))
            
            # Sort by (new product count, weight) ascending
            scored_insertions.sort(key=lambda x: (x[0], x[1]))
            
            for new_count, weight, c, new_seq, new_S, new_prod_set in scored_insertions:
                if time.time() > deadline:
                    break
                # Accept the first valid insertion
                current_seq, current_S, current_prod_set = new_seq, new_S, new_prod_set
                used = set(current_seq)
                remaining = [x for x in all_numbers if x not in used]
                improved = True
                break
            
            if time.time() > deadline:
                break

            # If no insertion worked, try targeted swaps
            if remaining and not improved:
                swap_attempts += 1
                
                # Strategy: try to remove elements that block many potential insertions
                # For each element in current_seq, count how many remaining numbers it blocks
                # (i.e., removing it would allow more insertions)
                
                best_swap = None
                best_score = float('inf')
                
                # Limit swap search to avoid timeout
                max_swap_candidates = min(len(current_seq), 10)
                for i in range(len(current_seq)):
                    if time.time() > deadline:
                        break
                    
                    x_to_remove = current_seq[i]
                    
                    # Try removing x_to_remove and see what becomes insertable
                    temp_seq = current_seq[:i] + current_seq[i+1:]
                    temp_S = get_prefix_products(temp_seq)
                    temp_prod_set = get_product_set(temp_seq)
                    
                    # Count how many remaining numbers can now be inserted
                    newly_insertable = []
                    for c in remaining:
                        res = try_insert_fast(temp_seq, c, temp_S, temp_prod_set)
                        if res is not None:
                            new_count = len(res[2]) - len(temp_prod_set)
                            newly_insertable.append((new_count, number_weights[c], c))
                    
                    # If we can insert at least one number, this is a candidate
                    if newly_insertable:
                        # Score: prefer swaps that allow inserting lighter numbers with fewer new products
                        newly_insertable.sort(key=lambda x: (x[0], x[1]))
                        best_new_count, best_weight, best_c = newly_insertable[0]
                        
                        # Overall score for this swap
                        swap_score = best_new_count + best_weight * 0.1
                        
                        if swap_score < best_score:
                            best_score = swap_score
                            best_swap = (i, x_to_remove, best_c)
                
                if best_swap is not None:
                    i, x_to_remove, c_to_add = best_swap
                    
                    # Perform the swap
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
        
        return current_seq
    
    # Main Search
    best_seq = []
    best_len = 0
    
    # 1. Multiple greedy starts with different orderings
    orders = [
        all_numbers,  # Natural order
        sorted(all_numbers, key=lambda x: number_weights[x]),  # Lightest first
        sorted(all_numbers, reverse=True),  # Heaviest first
        sorted(all_numbers, key=lambda x: (number_weights[x], x)),  # Weight then value
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        seq = greedy_build(order)
        seq = local_search(seq, max_swaps=20)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # 2. Perturbation-based restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Perturb best_seq: remove 1-2 random elements
        perturbed = best_seq[:]
        num_removals = random.randint(1, 2)
        for _ in range(num_removals):
            if not perturbed:
                break
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
        
        # Re-greedy from perturbed state with random ordering
        random.shuffle(all_numbers)
        new_seq = greedy_build(all_numbers)
        
        # Also try building from perturbed as seed
        perturbed_seq = perturbed[:]
        perturbed_S = get_prefix_products(perturbed_seq)
        perturbed_prod_set = get_product_set(perturbed_seq)
        perturbed_used = set(perturbed_seq)
        
        for c in all_numbers:
            if c in perturbed_used:
                continue
            res = try_insert_fast(perturbed_seq, c, perturbed_S, perturbed_prod_set)
            if res is not None:
                perturbed_seq, perturbed_S, perturbed_prod_set = res
                perturbed_used = set(perturbed_seq)
        
        # Take the better of the two
        candidate = new_seq if len(new_seq) >= len(perturbed_seq) else perturbed_seq
        candidate = local_search(candidate, max_swaps=15)
        
        if len(candidate) > best_len:
            best_len = len(candidate)
            best_seq = candidate[:]

    # 3. Final intensive local search
    if time.time() < deadline - 5:
        best_seq = local_search(best_seq, max_swaps=50)

    # Final Verification
    if not check_valid(best_seq):
        # Fallback to a known valid sequence
        best_seq = [2, 3, 5, 7]
        if not check_valid(best_seq):
            best_seq = [2]

    return best_seq
