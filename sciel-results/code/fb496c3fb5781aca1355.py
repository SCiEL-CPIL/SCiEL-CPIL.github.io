"""Improved program: greedy set-cover with efficient bitset coverage, plus structural shortcuts."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

# Deadline guard: return best found before this many seconds have elapsed.
_DEADLINE_SECONDS = 110.0


def _cover_radius1_tertiary(n):
    """
    Greedy set-cover for ternary radius-1 covering code of length n.
    Each codeword covers 1 + 2*n words (itself and all single-symbol changes).
    Total words = 3**n.
    We represent the universe as a boolean array indexed by base-3 number.
    """
    total = 3 ** n
    # Precompute the coverage of each possible codeword as a list of indices it covers.
    # For memory efficiency with n=8 (6561 words), each codeword covers 1+16=17 words.
    # We'll use a bytearray for the "covered" flag.
    covered = bytearray(total)
    # Precompute the list of covered indices for each candidate codeword.
    # candidate c (0..total-1) covers c and c +- 3^i (mod 3**n) for each position i,
    # but only if the digit at position i is not wrapped around in a way that changes other digits.
    # Actually, in base-3, changing digit i by +1 or -1 (mod 3) affects only that digit.
    # So the covered set for codeword c is: {c} union {c + 3^i * d : i in 0..n-1, d in {+1,-1}, digit_i(c+d*3^i) is valid}
    # But since we're working mod 3**n, we need to be careful. Actually, the Hamming ball of radius 1
    # around c consists of c and all words that differ in exactly one position. In base-3 representation,
    # changing position i by +1 or -1 (mod 3) gives a valid word. So:
    # covered_indices(c) = [c] + [ (c + 3**i) % total if digit_i(c) != 2 else c - 3**i, 
    #                             (c - 3**i) % total if digit_i(c) != 0 else c + 3**i for i in range(n) ]
    # Wait, that's not quite right. Let me think again.
    # For position i, the digit is d_i = (c // 3**i) % 3.
    # Changing it to (d_i+1)%3 gives index: c - d_i*3**i + ((d_i+1)%3)*3**i
    # Changing it to (d_i-1)%3 gives index: c - d_i*3**i + ((d_i-1)%3)*3**i
    # So we can precompute for each c the 1+2n covered indices.
    
    # Precompute powers of 3
    pow3 = [1]
    for i in range(1, n):
        pow3.append(pow3[-1] * 3)
    
    # Precompute covered indices for all candidates
    # This is a list of lists, each of length 1+2n
    covered_sets = []
    for c in range(total):
        cs = [c]
        for i in range(n):
            d = (c // pow3[i]) % 3
            # +1 change
            new_d_plus = (d + 1) % 3
            idx_plus = c - d * pow3[i] + new_d_plus * pow3[i]
            # -1 change
            new_d_minus = (d - 1) % 3
            idx_minus = c - d * pow3[i] + new_d_minus * pow3[i]
            cs.append(idx_plus)
            cs.append(idx_minus)
        covered_sets.append(cs)
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words.
    # For efficiency, we maintain a count of how many uncovered words each candidate covers.
    # Initially, every candidate covers 1+2n words.
    # When we pick a codeword, we mark its covered words as covered, and for each newly covered word,
    # we decrement the count of all candidates that cover that word.
    
    # Precompute reverse mapping: for each word w, which candidates cover it?
    # A candidate c covers w if w is in covered_sets[c]. Equivalently, c is in the radius-1 ball of w.
    # The radius-1 ball of w has 1+2n elements. So we can precompute this.
    reverse_cover = [None] * total
    for w in range(total):
        ball = [w]
        for i in range(n):
            d = (w // pow3[i]) % 3
            new_d_plus = (d + 1) % 3
            idx_plus = w - d * pow3[i] + new_d_plus * pow3[i]
            new_d_minus = (d - 1) % 3
            idx_minus = w - d * pow3[i] + new_d_minus * pow3[i]
            ball.append(idx_plus)
            ball.append(idx_minus)
        reverse_cover[w] = ball
    
    # uncovered_count[c] = number of currently uncovered words in covered_sets[c]
    uncovered_count = [len(covered_sets[0])] * total  # all start at 1+2n
    
    # Greedy selection
    selected = []
    uncovered_remaining = total
    start_time = time.time()
    
    while uncovered_remaining > 0:
        if time.time() - start_time > _DEADLINE_SECONDS - 5:
            break
        # Find the candidate with max uncovered_count
        best_c = -1
        best_count = -1
        for c in range(total):
            if uncovered_count[c] > best_count:
                best_count = uncovered_count[c]
                best_c = c
                if best_count == 0:
                    break
        if best_c == -1 or best_count == 0:
            break
        selected.append(best_c)
        # Mark covered words
        for w in covered_sets[best_c]:
            if not covered[w]:
                covered[w] = 1
                uncovered_remaining -= 1
                # For each candidate that covers w, decrement its uncovered_count
                for c2 in reverse_cover[w]:
                    uncovered_count[c2] -= 1
    
    # Convert selected indices to codeword lists
    result = []
    for c in selected:
        word = []
        tmp = c
        for i in range(n):
            word.append(tmp % 3)
            tmp //= 3
        result.append(word)
    return result


def _direct_sum_tetracode(n):
    """Direct sum of ternary tetracode with all words on remaining coordinates."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    
    # Handle the specific case we know well: q=3, radius=1
    if q == 3 and radius == 1:
        # For small n, use the tetracode direct sum as a baseline
        if n <= 4:
            # Perfect code exists for n=4 (tetracode). For n<4, just enumerate or use greedy.
            if n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            # For n < 4, use greedy
            return _cover_radius1_tertiary(n)
        
        # For n >= 5, try greedy set-cover
        try:
            result = _cover_radius1_tertiary(n)
            return result
        except Exception:
            pass
    
    # Fallback: direct sum construction or brute force for small cases
    if q == 3 and n >= 4:
        try:
            return _direct_sum_tetracode(n)
        except Exception:
            pass
    
    # General fallback: use a simple greedy approach
    total = q ** n
    covered = bytearray(total)
    selected = []
    uncovered_remaining = total
    
    # Precompute coverage for each candidate
    # For radius r, a codeword covers all words within Hamming distance r.
    # For small n and q, we can enumerate.
    
    # Generate all candidates
    candidates = list(itertools.product(range(q), repeat=n))
    
    # Map from tuple to index
    # We'll use a dict for indexing
    def tuple_to_idx(t):
        idx = 0
        for i, v in enumerate(t):
            idx += v * (q ** i)
        return idx
    
    def idx_to_tuple(idx):
        t = []
        for i in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(t)
    
    # Precompute covered sets for each candidate
    covered_sets = []
    for cand in candidates:
        cs = set()
        # Enumerate all words within Hamming distance radius
        # For small n and radius, this is feasible
        for offset in itertools.product(range(q), repeat=n):
            dist = sum(1 for a, b in zip(cand, offset) if a != b)
            if dist <= radius:
                cs.add(offset)
        covered_sets.append(cs)
    
    # Reverse cover
    reverse_cover = defaultdict(list)
    for c_idx, cs in enumerate(covered_sets):
        for w in cs:
            reverse_cover[w].append(c_idx)
    
    uncovered_count = [len(cs) for cs in covered_sets]
    
    while uncovered_remaining > 0:
        if time.time() - start_time > _DEADLINE_SECONDS - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(len(candidates)):
            if uncovered_count[c] > best_count:
                best_count = uncovered_count[c]
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        selected.append(candidates[best_c])
        for w in covered_sets[best_c]:
            if not covered[tuple_to_idx(w)]:
                covered[tuple_to_idx(w)] = 1
                uncovered_remaining -= 1
                for c2 in reverse_cover[w]:
                    uncovered_count[c2] -= 1
    
    return selected


# EVOLVE-BLOCK-END
