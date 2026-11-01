import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate the standard recursive superpermutation
    # This gives a baseline valid solution
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap_len = 0
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap_len = o
                    break
            new_word += block[overlap_len:]
        word = new_word
        
    # If n is small, the recursive is optimal
    if n <= 4:
        return "".join(map(str, word))
        
    # Extract the order of permutations from the recursive word
    # The recursive word contains all permutations of 1..n
    # We need to identify the sequence of permutations used to build it
    # Actually, it's easier to just generate the order recursively.
    
    def get_recursive_order(n):
        if n == 1:
            return [(1,)]
        order_n_minus_1 = get_recursive_order(n - 1)
        current_order = []
        seen = set()
        # The recursive construction iterates through the word of n-1
        # and for each new permutation of n-1, adds P, n, P
        # We need to replicate this logic to get the order of n-permutations
        # The order of n-permutations is:
        # For each permutation P of n-1 in the order they appear in W_{n-1}:
        #   The permutation formed is P + [n] + P? No.
        #   The block is P, n, P.
        #   The permutations contained in P, n, P are not just one.
        #   Wait, the standard construction says:
        #   "each permutation P of the previous word ... becomes P m P"
        #   This implies that the *new* permutations introduced are related to P.
        #   Actually, the set of permutations of 1..n is partitioned into sets,
        #   one for each permutation P of 1..n-1.
        #   The permutations in the set for P are those that contain P as a subsequence?
        #   No, the construction is:
        #   W_n is formed by concatenating blocks B_P for each P in W_{n-1} (in order).
        #   B_P = P, n, P.
        #   The permutations of 1..n that appear *first* in B_P are the ones we care about for the next step?
        #   No, the next step uses ALL permutations of 1..n.
        
        # Let's just extract the permutations from the generated word in order of first appearance.
        return [] # Placeholder

    # Actually, let's just use the word we generated.
    # We can extract the sequence of permutations by scanning the word.
    # But we need the *order* in which they were "added" or appear?
    # The local search works on ANY permutation of the set of all n! permutations.
    # The recursive word provides a valid ordering.
    # We can extract the ordering by finding the first occurrence of each permutation in the word?
    # No, that's not the order used to build it.
    # But it IS a valid ordering that produces a valid superpermutation.
    # And it's a good starting point for local search.
    
    # Extract all permutations from the word in order of first appearance
    all_perms = set()
    for i in range(n, 8): # 1..7
        pass # Hardcode for 7? No, general n.
        
    # General extraction
    seen_perms = []
    seen_set = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p not in seen_set:
            seen_set.add(p)
            seen_perms.append(p)
            
    # Now seen_perms is an ordering of all n! permutations.
    # We will run local search on this ordering.
    
    N = len(seen_perms)
    
    # Precompute overlap function
    def get_overlap(a, b):
        # a, b are tuples
        # max overlap k
        # a[-k:] == b[:k]
        # k can be up to n-1
        for k in range(n-1, 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    # Calculate total overlap
    def total_overlap(order):
        s = 0
        for i in range(len(order) - 1):
            s += get_overlap(order[i], order[i+1])
        return s

    current_order = seen_perms
    current_score = total_overlap(current_order)
    
    best_order = current_order.copy()
    best_score = current_score
    
    # Local Search with Simulated Annealing
    start_time = time.time()
    deadline = 140 # 10s margin
    
    T = 1.0
    min_T = 0.01
    
    while time.time() - start_time < deadline:
        # Random swap
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Calculate delta
        # Affected overlaps:
        # (i-1, i), (i, i+1)
        # (j-1, j), (j, j+1)
        
        old_score = 0
        new_score = 0
        
        # Helper to get overlap at index k (between k and k+1)
        def o_at(k, ord):
            if k < 0 or k >= len(ord) - 1:
                return 0
            return get_overlap(ord[k], ord[k+1])
        
        # Old overlaps
        if i > 0: old_score += o_at(i-1, current_order)
        old_score += o_at(i, current_order)
        if j > 0: old_score += o_at(j-1, current_order)
        old_score += o_at(j, current_order)
        
        # Perform swap in a temporary list? Or just calculate new values?
        # Let's create a modified view or just swap in place and revert?
        # In-place swap is faster.
        
        # Save old values
        pi_i = current_order[i]
        pi_j = current_order[j]
        
        # Swap
        current_order[i] = pi_j
        current_order[j] = pi_i
        
        # New overlaps
        if i > 0: new_score += o_at(i-1, current_order)
        new_score += o_at(i, current_order)
        if j > 0: new_score += o_at(j-1, current_order)
        new_score += o_at(j, current_order)
        
        delta = new_score - old_score
        
        # Acceptance criterion
        if delta > 0 or (T > min_T and random.random() < math.exp(delta / T)):
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            # Revert
            current_order[i] = pi_i
            current_order[j] = pi_j
            
        # Cool down
        T *= 0.999
        if T < min_T:
            # Restart or stop?
            # If we restart, we might lose progress.
            # Let's just stop cooling and keep exploring at min_T or break.
            # Given the small gap, we might find it quickly.
            break
            
    # Construct final word from best_order
    final_word = list(best_order[0])
    for i in range(1, len(best_order)):
        prev = best_order[i-1]
        curr = best_order[i]
        # Find max overlap
        max_ov = 0
        for k in range(n-1, 0, -1):
            if prev[-k:] == curr[:k]:
                max_ov = k
                break
        final_word += curr[max_ov:]
        
    return "".join(map(str, final_word))
