import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    n = 512
    random.seed(12345)
    
    # Bitset representation:
    # We need 513 bits (indices 0 to 512).
    # 513 / 64 = 8.015, so we need 9 integers.
    num_words = (n + 64) // 64 + 1
    
    # Precompute word and bit for each index
    idx_word = [i // 64 for i in range(n + 1)]
    idx_bit = [i % 64 for i in range(n + 1)]
    
    # Precompute conflict triples as (word1, bit1, word2, bit2, word3, bit3)
    # This avoids computing word/bit inside can_add
    conflict_triples = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Store precomputed word/bit indices for the other 3 elements
            wb = (idx_word[b], idx_bit[b], idx_word[c], idx_bit[c], idx_word[e], idx_bit[e])
            wc = (idx_word[c], idx_bit[c], idx_word[e], idx_bit[e], idx_word[a], idx_bit[a])
            we = (idx_word[e], idx_bit[e], idx_word[a], idx_bit[a], idx_word[b], idx_bit[b])
            wa = (idx_word[a], idx_bit[a], idx_word[b], idx_bit[b], idx_word[c], idx_bit[c])
            
            conflict_triples[a].append(wb)
            conflict_triples[b].append(wa)
            conflict_triples[c].append(wc)
            conflict_triples[e].append(we)
            
    def can_add(x, bits):
        """Check if adding x to S (represented by bits) creates a 4-AP."""
        for w1, b1, w2, b2, w3, b3 in conflict_triples[x]:
            # Check if all three are present
            if (bits[w1] >> b1) & 1 and (bits[w2] >> b2) & 1 and (bits[w3] >> b3) & 1:
                return False
        return True

    def has_4ap(bits):
        """Verify if bits contains a 4-AP."""
        # Iterate over all possible 4-APs
        for a in range(1, n + 1):
            if not ((bits[idx_word[a]] >> idx_bit[a]) & 1):
                continue
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if ((bits[idx_word[b]] >> idx_bit[b]) & 1 and 
                    (bits[idx_word[c]] >> idx_bit[c]) & 1 and 
                    (bits[idx_word[e]] >> idx_bit[e]) & 1):
                    return True
        return False

    def bits_to_list(bits):
        """Convert bitset to list of indices."""
        res = []
        for i in range(1, n + 1):
            if (bits[idx_word[i]] >> idx_bit[i]) & 1:
                res.append(i)
        return res

    def ternary_01_set(n):
        """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
        result = set()
        for mask in range(1, 1 << 6):
            val = 0
            for bit in range(6):
                if mask & (1 << bit):
                    val += 3 ** bit
            if 1 <= val <= n:
                result.add(val)
        return result

    def greedy_construct(S_init, order):
        """Greedy construction starting from S_init, adding elements in 'order'."""
        bits = [0] * num_words
        for x in S_init:
            bits[idx_word[x]] |= (1 << idx_bit[x])
        
        S_size = len(S_init)
        for x in order:
            if (bits[idx_word[x]] >> idx_bit[x]) & 1:
                continue
            if can_add(x, bits):
                bits[idx_word[x]] |= (1 << idx_bit[x])
                S_size += 1
        return bits, S_size

    def local_search(bits, deadline, max_rounds=2000):
        """
        Local search: try to remove one element and add two others (1-2 move) 
        or remove one and add one (1-1 move) to maintain/increase size.
        """
        bits = bits.copy()
        current_size = sum(1 for i in range(1, n+1) if (bits[idx_word[i]] >> idx_bit[i]) & 1)
        no_improvement_count = 0
        
        # Precompute list of present elements for fast sampling
        present_list = [i for i in range(1, n+1) if (bits[idx_word[i]] >> idx_bit[i]) & 1]
        absent_list = [i for i in range(1, n+1) if not ((bits[idx_word[i]] >> idx_bit[i]) & 1)]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Try 1-2 move: Remove 1, Add 2
            if present_list:
                to_remove = random.choice(present_list)
                bits[idx_word[to_remove]] &= ~(1 << idx_bit[to_remove])
                present_list.remove(to_remove)
                absent_list.append(to_remove)
                
                # Try to find 2 candidates to add
                # Shuffle a subset of absent_list to avoid full shuffle cost
                candidates = absent_list.copy()
                random.shuffle(candidates)
                
                added = []
                # We need to check against the modified bits
                for c in candidates:
                    if len(added) == 2:
                        break
                    if can_add(c, bits):
                        bits[idx_word[c]] |= (1 << idx_bit[c])
                        added.append(c)
                        absent_list.remove(c)
                        present_list.append(c)
                        
                if len(added) == 2:
                    current_size += 1
                    no_improvement_count = 0
                else:
                    # Revert if not improved
                    for a in added:
                        bits[idx_word[a]] &= ~(1 << idx_bit[a])
                        present_list.remove(a)
                        absent_list.append(a)
                    # Restore removed element
                    bits[idx_word[to_remove]] |= (1 << idx_bit[to_remove])
                    present_list.append(to_remove)
                    absent_list.remove(to_remove)
                    
                    no_improvement_count += 1
                    if no_improvement_count > 100:
                        # Try 1-1 swap to diversify
                        if present_list:
                            to_remove = random.choice(present_list)
                            bits[idx_word[to_remove]] &= ~(1 << idx_bit[to_remove])
                            present_list.remove(to_remove)
                            absent_list.append(to_remove)
                            
                            candidates = absent_list.copy()
                            random.shuffle(candidates)
                            found_swap = False
                            for c in candidates:
                                if can_add(c, bits):
                                    bits[idx_word[c]] |= (1 << idx_bit[c])
                                    absent_list.remove(c)
                                    present_list.append(c)
                                    found_swap = True
                                    break
                                    
                            if not found_swap:
                                # Revert swap
                                bits[idx_word[to_remove]] |= (1 << idx_bit[to_remove])
                                present_list.append(to_remove)
                                absent_list.remove(to_remove)
                                break
                            else:
                                no_improvement_count = 0
                        else:
                            break
            else:
                break
                
        return bits, current_size

    # Initial sets
    best_bits = [0] * num_words
    best_size = 0
    
    # 1. Greedy from empty
    order = list(range(1, n + 1))
    random.shuffle(order)
    greedy_bits, greedy_size = greedy_construct(set(), order)
    if greedy_size > best_size:
        best_bits = greedy_bits
        best_size = greedy_size
        
    # 2. Greedy from ternary
    ternary = ternary_01_set(n)
    order2 = list(range(1, n + 1))
    random.shuffle(order2)
    ternary_greedy, ternary_size = greedy_construct(ternary, order2)
    if ternary_size > best_size:
        best_bits = ternary_greedy
        best_size = ternary_size
        
    # 3. Local search on best
    if time.time() < deadline:
        best_bits, best_size = local_search(best_bits, deadline, max_rounds=3000)
        
    # 4. Multiple restarts with kicks
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Random start
        if attempt % 3 == 0:
            S_init = set()
        elif attempt % 3 == 1:
            S_init = ternary_01_set(n)
        else:
            S_init = set(random.sample(range(1, n + 1), 30))
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        current, current_size = greedy_construct(S_init, order)
        
        # Local search
        if current_size >= best_size - 5: # Allow slightly worse to explore
            current, current_size = local_search(current, deadline, max_rounds=1000)
            
        if current_size > best_size:
            best_bits = current
            best_size = current_size
            
    # 5. Final local search on the best set
    if time.time() < deadline:
        best_bits, best_size = local_search(best_bits, deadline, max_rounds=2000)
        
    # Final verification
    if has_4ap(best_bits):
        # Fallback to safe ternary set
        ternary = ternary_01_set(n)
        best_bits = [0] * num_words
        for x in ternary:
            best_bits[idx_word[x]] |= (1 << idx_bit[x])
        best_size = len(ternary)
        
    return bits_to_list(best_bits)
