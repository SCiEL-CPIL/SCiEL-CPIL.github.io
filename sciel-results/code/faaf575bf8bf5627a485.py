import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate standard superpermutation for n-1
    # For n=7, we need W_6.
    # W_6 standard is 873.
    # We can generate W_6 using the same recursive method.
    
    def generate_standard(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            # Extract unique permutations of length m-1 from word
            # Optimization: use a set to track seen
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap - optimized: only need to check up to 2*(m-1)
                # because block is p + [m] + p, so the prefix is p followed by m
                # The suffix of new_word must match a prefix of block
                # Since block starts with p (length m-1), overlap can be at most 2*(m-1)
                # but realistically much less. Let's cap at 2*(m-1) for efficiency.
                max_k = min(len(new_word), 2 * (m - 1))
                overlap = 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        overlap = 0
                        break
                    if new_word[len(new_word) - k:] == block[:k]:
                        overlap = k
                        break
                new_word += block[overlap:]
            word = new_word
        return word

    # Get W_{n-1}
    w_prev = generate_standard(n - 1)
    
    # Now extend to n
    # We extract permutations of 1..n-1 from w_prev
    perms, seen = [], set()
    for i in range(len(w_prev) - (n - 1) + 1):
        window = tuple(w_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # Standard order is the order of appearance in w_prev
    # For n=7, len(perms) is 720.
    
    # Precompute blocks for all permutations
    # block for permutation p is: p + [n] + p
    # We'll work with tuples for efficiency
    
    def build_word(perms_order, n, precomputed_blocks=None):
        """Build superpermutation from given order of permutations."""
        if precomputed_blocks is None:
            precomputed_blocks = {}
        
        new_word = []
        for p in perms_order:
            block = precomputed_blocks[p]
            # Find max overlap - optimized
            # block = p + [n] + p, length = 2*(n-1) + 1
            # Max useful overlap is 2*(n-1)
            max_k = min(len(new_word), 2 * (n - 1))
            overlap = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            new_word += block[overlap:]
        return new_word
    
    # Precompute blocks for all perms
    precomputed_blocks = {}
    for p in perms:
        block = list(p) + [n] + list(p)
        precomputed_blocks[p] = block
    
    # 1. Try standard order
    word_std = build_word(perms, n, precomputed_blocks)
    best_word = word_std
    best_len = len(word_std)
    best_order = perms[:]
    
    # Check time
    if time.time() > deadline:
        return "".join(map(str, best_word))
        
    # 2. Try lexicographic order
    perms_lex = sorted(perms)
    if perms_lex != perms:
        word_lex = build_word(perms_lex, n, precomputed_blocks)
        if len(word_lex) < best_len:
            best_word = word_lex
            best_len = len(word_lex)
            best_order = perms_lex[:]
            
    # Check time
    if time.time() > deadline:
        return "".join(map(str, best_word))
        
    # 3. Try reverse lexicographic
    perms_rev = sorted(perms, reverse=True)
    if perms_rev != perms:
        word_rev = build_word(perms_rev, n, precomputed_blocks)
        if len(word_rev) < best_len:
            best_word = word_rev
            best_len = len(word_rev)
            best_order = perms_rev[:]
            
    # Check time
    if time.time() > deadline:
        return "".join(map(str, best_word))
        
    # 4. Local search: try swapping adjacent permutations
    # This is more effective than random shuffles
    import random
    random.seed(42)
    
    current_order = best_order[:]
    current_len = best_len
    
    # Try adjacent swaps - multiple passes
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(len(current_order) - 1):
            if time.time() > deadline - 5:
                break
            
            # Try swapping i and i+1
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            
            new_word = build_word(current_order, n, precomputed_blocks)
            new_len = len(new_word)
            
            if new_len < current_len:
                current_len = new_len
                improved = True
            else:
                # Swap back
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
        
        if not improved:
            break
    
    if current_len < best_len:
        best_word = build_word(current_order, n, precomputed_blocks)
        best_len = current_len
        best_order = current_order[:]
    
    # 5. Try some random shuffles if time permits
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        perms_rand = perms[:]
        random.shuffle(perms_rand)
        word_rand = build_word(perms_rand, n, precomputed_blocks)
        if len(word_rand) < best_len:
            best_word = word_rand
            best_len = len(word_rand)
            best_order = perms_rand[:]
    
    return "".join(map(str, best_word))
