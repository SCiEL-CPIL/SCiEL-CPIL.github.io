import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin

    all_numbers = list(range(2, 129))
    
    def check_valid(seq):
        """
        Check if strictly increasing sequence 'seq' has distinct products
        for all non-empty consecutive blocks.
        """
        n = len(seq)
        if n == 0:
            return True
        
        # Compute prefix products S[0]=1, S[i]=seq[0]*...*seq[i-1]
        S = [1] * (n + 1)
        for i in range(n):
            S[i+1] = S[i] * seq[i]
            
        seen_products = set()
        # Products of blocks [i+1, j] are S[j] // S[i] for 0 <= i < j <= n
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod = S[j] // S[i]
                if prod in seen_products:
                    return False
                seen_products.add(prod)
        return True

    def get_prefix_products(seq):
        n = len(seq)
        S = [1] * (n + 1)
        for i in range(n):
            S[i+1] = S[i] * seq[i]
        return S

    def get_product_set(seq):
        S = get_prefix_products(seq)
        n = len(seq)
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                seen_products.add(S[j] // S[i])
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
        # Formula: x * (P[j] // P[i]) for 0 <= i <= idx, idx <= j <= n
        # Note: S is prefix of old seq (length n+1)
        new_products_to_check = []
        for i in range(idx + 1):
            for j in range(idx, n + 1):
                # P[j] // P[i] is product of seq[i..j-1]
                # If i==idx and j==idx, product is 1 (empty), so val = c
                # If i < idx, left part exists.
                # If j > idx, right part exists.
                val = c * (S[j] // S[i])
                if val in prod_set:
                    return None
                new_products_to_check.append(val)
        
        # If all valid, update state
        new_prod_set = set(prod_set)
        new_prod_set.update(new_products_to_check)
        
        # Update prefix products
        new_S = [1] * (n + 2)
        for i in range(idx + 1):
            new_S[i] = S[i]
        for i in range(idx + 1, n + 2):
            new_S[i] = S[i-1] * c
            
        return new_seq, new_S, new_prod_set

    def local_search(seq, S, prod_set):
        """
        Improve sequence by insertions.
        """
        current_seq = seq[:]
        current_S = S[:]
        current_prod_set = set(prod_set)
        used = set(current_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        improved = True
        while improved:
            improved = False
            
            # Try insertions
            # Sort remaining by some heuristic? Smallest first?
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

            # If no insertion worked, try swaps (slower, use check_valid)
            # We only do this if stuck.
            for _ in range(20): # Reduced swaps as they are expensive
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

        return current_seq, current_S, current_prod_set

    # Main Search Loop
    best_seq = []
    best_len = 0
    
    # 1. Greedy Start
    def greedy_build(start_list=None):
        if start_list is None:
            start_list = []
        seq = start_list[:]
        S = get_prefix_products(seq)
        prod_set = get_product_set(seq)
        used = set(seq)
        
        # Try to insert all numbers in order
        for c in all_numbers:
            if c in used:
                continue
            res = try_insert_fast(seq, c, S, prod_set)
            if res is not None:
                seq, S, prod_set = res
                used = set(seq)
        
        return seq, S, prod_set

    # Run greedy
    seq, S, prod_set = greedy_build()
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]
        
    # Local Search on greedy result
    seq, S, prod_set = local_search(seq, S, prod_set)
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]
        
    # 2. Randomized Restart / Perturbation
    for _ in range(500):
        if time.time() > deadline:
            break
            
        # Perturb best_seq
        perturbed = best_seq[:]
        num_removals = random.randint(1, 3)
        for _ in range(num_removals):
            if not perturbed:
                break
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Re-greedy from perturbed state
        new_seq, new_S, new_prod_set = greedy_build(perturbed)
        
        # Local search on new_seq
        new_seq, new_S, new_prod_set = local_search(new_seq, new_S, new_prod_set)
        
        if len(new_seq) > best_len:
            best_len = len(new_seq)
            best_seq = new_seq[:]
            
    # Final Polish
    S = get_prefix_products(best_seq)
    prod_set = get_product_set(best_seq)
    best_seq, S, prod_set = local_search(best_seq, S, prod_set)
    
    return best_seq
