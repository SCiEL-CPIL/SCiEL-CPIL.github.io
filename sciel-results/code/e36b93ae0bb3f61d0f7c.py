import time
import itertools
import random
from collections import deque

def construct(n):
    start_time = time.time()
    timeout = 110.0  # Safety margin before 150s kill
    
    # For n=1, return "1"
    if n == 1:
        return "1"
    
    # For small n, we can use known optimal values or fast construction
    # n=2: "121" (len 3)
    # n=3: "123121321" (len 9) - actually optimal is 9
    # n=4: optimal is 33
    # n=5: optimal is 153
    # n=6: recursive gives 873, optimal known? LB 872. Recursive is 873.
    # n=7: recursive gives 5913. Known best is 5905. LB 5884.
    
    # The recursive construction S_n:
    # S_1 = "1"
    # S_n = for each perm P in S_{n-1} (in order of appearance):
    #         append P, n, P  (with maximal overlap)
    # This gives length 1! + 2! + ... + n!
    # For n=7: 1+2+6+24+120+720+5040 = 5913
    
    def get_recursive_superpermutation(k):
        if k == 1:
            return [1]
        
        prev = get_recursive_superpermutation(k - 1)
        
        # Extract unique permutations of length k-1 from prev in order
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 1) + 1):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            # Overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return new_word

    # Get the recursive solution
    recursive_word = get_recursive_superpermutation(n)
    best_word = recursive_word
    best_len = len(best_word)
    
    # For n <= 5, recursive is optimal, return it
    if n <= 5:
        return "".join(map(str, best_word))
    
    # For n=6, recursive gives 873. Let's try to improve.
    # For n=7, recursive gives 5913. Known best is 5905.
    
    # Strategy: The recursive construction produces a specific ordering of permutations.
    # We can try to find a better ordering using local search on the permutation sequence.
    # The cost of a sequence of permutations p1, p2, ..., p_{n!} is:
    #   n + sum_{i=1}^{n!-1} (n - overlap(p_i, p_{i+1}))
    # where overlap(a, b) = max{k < n : a[n-k:] == b[:k]}
    
    # For n=7, n! = 5040. We have 5040 permutations.
    # The recursive ordering gives total length 5913.
    # We want to minimize the sum of (n - overlap) between consecutive permutations.
    
    # Let's extract the permutation order from the recursive word
    def extract_perm_order(word, n):
        perms = []
        seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n and window not in seen:
                seen.add(window)
                perms.append(window)
        return perms
    
    if n <= 7 and time.time() - start_time < timeout - 5:
        perm_order = extract_perm_order(best_word, n)
        num_perms = len(perm_order)
        
        if num_perms == n ** 0 + 1:  # sanity check
            pass
        
        # Verify we have all permutations
        all_perms = set(itertools.permutations(range(1, n+1)))
        if set(perm_order) != all_perms:
            # Fallback to recursive
            pass
        else:
            # Local search: try to swap adjacent permutations to reduce total length
            # Total length = n + sum_{i=0}^{num_perms-2} (n - overlap(perm_order[i], perm_order[i+1]))
            
            def calc_overlap(a, b):
                # max k < n such that a[n-k:] == b[:k]
                for k in range(n-1, 0, -1):
                    if a[n-k:] == b[:k]:
                        return k
                return 0
            
            def total_length(order):
                total = n
                for i in range(len(order)-1):
                    total += n - calc_overlap(order[i], order[i+1])
                return total
            
            current_order = list(perm_order)
            current_len = total_length(current_order)
            
            # Optimized Hill Climbing
            # We use full passes without immediate restart to reduce overhead.
            # If stuck, we apply a random perturbation.
            
            current_order = list(perm_order)
            current_len = total_length(current_order)
            
            def apply_swap(i):
                # Swap i and i+1 in current_order and update current_len
                nonlocal current_len
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    old_cost += n - calc_overlap(current_order[i-1], current_order[i])
                    new_cost += n - calc_overlap(current_order[i-1], current_order[i+1])
                
                old_cost += n - calc_overlap(current_order[i], current_order[i+1])
                new_cost += n - calc_overlap(current_order[i+1], current_order[i])
                
                if i+2 < num_perms:
                    old_cost += n - calc_overlap(current_order[i+1], current_order[i+2])
                    new_cost += n - calc_overlap(current_order[i], current_order[i+2])
                
                delta = new_cost - old_cost
                if delta < 0:
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    current_len += delta
                    return True
                return False

            no_improvement_passes = 0
            max_no_improvement_passes = 50 # Allow some random restarts
            
            while time.time() - start_time < timeout - 10:
                if no_improvement_passes >= max_no_improvement_passes:
                    break
                
                improved_in_pass = False
                for i in range(num_perms - 1):
                    if time.time() - start_time > timeout - 10:
                        break
                    if apply_swap(i):
                        improved_in_pass = True
                
                if not improved_in_pass:
                    no_improvement_passes += 1
                    # Perturbation: Swap two random elements
                    if no_improvement_passes < max_no_improvement_passes:
                        # Pick two random indices
                        idx1 = random.randint(0, num_perms - 1)
                        idx2 = random.randint(0, num_perms - 1)
                        if idx1 != idx2:
                            current_order[idx1], current_order[idx2] = current_order[idx2], current_order[idx1]
                            # Recalculate length? 
                            # This is expensive. 
                            # Better to just accept the perturbation and let hill climbing fix it?
                            # But we need current_len to be accurate for delta checks.
                            # We can recalculate total_length once.
                            current_len = total_length(current_order)
                            # Reset counter? No, let it accumulate.
                            # But if we keep perturbing without improvement, we waste time.
                            # Let's just break after a few failed passes.
                else:
                    no_improvement_passes = 0
            
            # Reconstruct word from best order
            if current_len < best_len:
                # Build the word
                new_word = list(current_order[0])
                for i in range(1, num_perms):
                    a = current_order[i-1]
                    b = current_order[i]
                    ov = calc_overlap(a, b)
                    new_word.extend(b[ov:])
                
                best_word = new_word
                best_len = len(best_word)
    
    # Check time
    if time.time() - start_time > timeout:
        return "".join(map(str, best_word))
    
    return "".join(map(str, best_word))
