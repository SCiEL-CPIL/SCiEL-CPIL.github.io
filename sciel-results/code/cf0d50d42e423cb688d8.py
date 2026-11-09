import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    
    # Precompute prime exponent vectors for all numbers 2..128
    prime_list = primes
    num_primes = len(prime_list)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    
    # Exponent vector for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
        
    def add_vecs(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))

    best_seq = []
    best_len = 0
    
    def backtrack_include_first(idx, seq, all_products, suffix_products):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
            
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = seq[:]
            
        if idx == len(all_numbers):
            return
            
        # Pruning: even if we take all remaining, can we beat best?
        remaining_count = len(all_numbers) - idx
        if current_len + remaining_count <= best_len:
            return
            
        num = all_numbers[idx]
        
        # Option 1: Include num (if valid)
        vec_num = exp_vecs[num]
        new_suffixes = [vec_num]
        for v in suffix_products:
            new_suffixes.append(add_vecs(vec_num, v))
            
        valid = True
        for p in new_suffixes:
            if p in all_products:
                valid = False
                break
                
        if valid:
            new_all_products = set(all_products)
            new_all_products.update(new_suffixes)
            new_suffix_products = new_suffixes
            new_seq = seq + [num]
            
            backtrack_include_first(idx + 1, new_seq, new_all_products, new_suffix_products)
            
        # Option 2: Skip num
        backtrack_include_first(idx + 1, seq, all_products, suffix_products)

    # Run the include-first backtracking
    backtrack_include_first(0, [], set(), [])
    
    # If time remains, try to improve via local search on the best found
    if time.time() < deadline - 5:
         def check_sequence_valid(seq):
            n = len(seq)
            if n == 0:
                return True
            prods = set()
            for i in range(n):
                vec = [0] * num_primes
                for j in range(i, n):
                    ej = exp_vecs[seq[j]]
                    for k in range(num_primes):
                        vec[k] += ej[k]
                    tv = tuple(vec)
                    if tv in prods:
                        return False
                    prods.add(tv)
            return True

         def try_insert(seq, c):
            n = len(seq)
            valid_positions = []
            for i in range(n + 1):
                if i > 0 and seq[i-1] >= c:
                    continue
                if i < n and c >= seq[i]:
                    continue
                valid_positions.append(i)
            
            if not valid_positions:
                return None
            
            # Sort by centrality
            mid = n / 2.0
            valid_positions.sort(key=lambda i: abs(i - mid))
            
            for i in valid_positions:
                new_seq = seq[:i] + [c] + seq[i:]
                if check_sequence_valid(new_seq):
                    return new_seq
            return None

         def local_search(seq, max_iter=100):
            current = seq[:]
            used = set(current)
            remaining = [x for x in all_numbers if x not in used]
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                random.shuffle(remaining)
                for c in remaining:
                    new_seq = try_insert(current, c)
                    if new_seq is not None:
                        current = new_seq
                        used = set(current)
                        remaining = [x for x in all_numbers if x not in used]
                        improved = True
                        break
                if not improved:
                    break
            return current

         improved_seq = local_search(best_seq)
         if len(improved_seq) > len(best_seq):
             best_seq = improved_seq

    return best_seq
