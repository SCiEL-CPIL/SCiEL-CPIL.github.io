"""Greedy set-cover with randomized removal passes for the ternary covering code K_3(7,1)."""
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fixed task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    
    deadline = time.time() + 110.0  # safety margin before 120s kill
    rng = random.Random(42)
    
    # Total words in space
    total = q ** n  # 2187
    
    # Precompute neighbor lists for all words (each word + words at distance 1)
    # A word w is covered by codeword c if dist(w, c) <= radius
    # For radius 1: w is in N(c) = {c} ∪ {words at dist 1 from c}
    # We need to cover all 2187 words.
    
    # Precompute: for each word index, which codeword candidates cover it
    # But 3^7 = 2187 candidates, each covers 1 + 7*2 = 15 words
    # Total coverage capacity: 2187 * 15 = 32805, need to cover 2187
    
    # Strategy: Greedy set cover
    # 1. Generate all candidate codewords (all 2187 words)
    # 2. Greedily pick the codeword that covers the most uncovered words
    # 3. After greedy, try removal passes to see if any codeword is redundant
    
    # For speed, use numpy
    # Represent each word as an index 0..2186
    # Precompute for each candidate codeword, the set of words it covers
    
    # Generate all words
    all_words = []
    for idx in range(total):
        w = []
        x = idx
        for pos in range(n):
            w.append(x % q)
            x //= q
        all_words.append(w)
    
    # Precompute neighbor sets for each word
    # neighbors[i] = set of word indices covered by codeword i
    # For radius 1: word i itself + all words differing in exactly one position
    
    # Use a boolean array for coverage tracking
    # covered[j] = True if word j is covered
    
    # For efficiency, precompute the list of neighbor indices for each codeword
    # neighbor_list[i] = list of word indices covered by codeword i
    
    # Precompute
    neighbor_list = []
    for i in range(total):
        w = all_words[i]
        covered_set = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_w = w[:]
                    new_w[pos] = new_val
                    # Convert to index
                    idx = 0
                    for p in range(n):
                        idx = idx * q + new_w[p]
                    covered_set.append(idx)
        neighbor_list.append(covered_set)
    
    # Greedy set cover
    # uncovered_count[j] = number of times word j is still uncovered (initially 1)
    # We track which words are uncovered
    uncovered = [True] * total
    uncovered_count = [1] * total
    num_uncovered = total
    
    chosen = []
    chosen_set = set()
    
    # For greedy: at each step, find the codeword that covers the most uncovered words
    # To speed up, we can maintain a count of how many uncovered words each candidate covers
    
    # Initial marginal coverage for each candidate
    marginal = [0] * total
    for i in range(total):
        for j in neighbor_list[i]:
            marginal[i] += 1
    
    # Greedy selection
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find candidate with max marginal coverage among uncovered
        best_idx = -1
        best_val = -1
        for i in range(total):
            if i in chosen_set:
                continue
            if marginal[i] > best_val:
                best_val = marginal[i]
                best_idx = i
        
        if best_idx == -1 or best_val == 0:
            break
        
        chosen.append(best_idx)
        chosen_set.add(best_idx)
        
        # Update marginal coverage for all other candidates
        for j in neighbor_list[best_idx]:
            if uncovered[j]:
                num_uncovered -= 1
                uncovered[j] = False
                # Decrease marginal for all candidates that cover j
                for i in range(total):
                    if i != best_idx and j in neighbor_list[i]:
                        marginal[i] -= 1
    
    # Convert chosen indices to codewords
    codewords = [all_words[i] for i in chosen]
    
    # Now try removal passes: see if any codeword is redundant
    # A codeword is redundant if all words it covers are also covered by other chosen codewords
    
    def is_redundant(cw_idx, chosen_list, all_words, neighbor_list, q, n):
        """Check if codeword at position cw_idx in chosen_list is redundant."""
        # Get the words covered by this codeword
        cw = all_words[chosen_list[cw_idx]]
        # For each word covered by this codeword, check if it's covered by another chosen codeword
        for pos in range(n):
            for val in range(q):
                # Check word differing at position pos with value val
                new_w = cw[:]
                new_w[pos] = val
                # Convert to index
                idx = 0
                for p in range(n):
                    idx = idx * q + new_w[p]
                # Check if idx is covered by any other chosen codeword
                found = False
                for other_idx in chosen_list:
                    if other_idx == chosen_list[cw_idx]:
                        continue
                    if idx in neighbor_list[other_idx]:
                        found = True
                        break
                if not found:
                    return False
        return True
    
    # Try removal passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_chosen = chosen[:]
        for i in range(len(new_chosen) - 1, -1, -1):
            if is_redundant(i, new_chosen, all_words, neighbor_list, q, n):
                new_chosen.pop(i)
                improved = True
        chosen = new_chosen
    
    codewords = [all_words[i] for i in chosen]
    
    # Try randomized removal passes with different orders
    for attempt in range(20):
        if time.time() > deadline:
            break
        rng.shuffle(chosen)
        new_chosen = chosen[:]
        for i in range(len(new_chosen) - 1, -1, -1):
            if is_redundant(i, new_chosen, all_words, neighbor_list, q, n):
                new_chosen.pop(i)
        if len(new_chosen) < len(chosen):
            chosen = new_chosen
            break
    
    codewords = [all_words[i] for i in chosen]
    
    # Verify validity
    covered = [False] * total
    for cw_idx in chosen:
        for j in neighbor_list[cw_idx]:
            covered[j] = True
    if not all(covered):
        # Fallback to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    
    return codewords
