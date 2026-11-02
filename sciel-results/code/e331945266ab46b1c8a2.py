"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """Construct a covering code using greedy + local search with numpy bitsets."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    total_words = q ** n

    # Precompute all words as integers 0..total_words-1
    # Each word is a tuple of n digits in 0..q-1
    # Map: word_index -> tuple, and precompute neighbor sets

    # For efficiency with n=6, q=3: 729 words, each has 1+6*2=13 neighbors (radius 1)
    # Use numpy bitsets: 729 bits = 12 uint64s

    # Build neighbor lists: for each codeword, which words does it cover?
    # A word w is covered by codeword c if Hamming distance <= radius

    # For radius=1: covered = {w} ∪ {words differing in exactly 1 position}

    # Precompute: for each codeword index, the list of covered word indices
    # This is expensive for large spaces but fine for 729 words

    # Convert word index to tuple
    def idx_to_tuple(idx):
        digits = []
        for i in range(n):
            digits.append(idx % q)
            idx //= q
        return tuple(digits)

    def tuple_to_idx(tup):
        idx = 0
        for i in range(n):
            idx = idx * q + tup[i]
        return idx

    # Precompute coverage: for each possible codeword, which words it covers
    # coverage[c] = set of word indices within radius of c
    # For n=6, q=3: 729 codewords × 13 covered words = manageable

    # Build coverage as list of lists of word indices
    coverage = []
    for c_idx in range(total_words):
        c_tup = idx_to_tuple(c_idx)
        covered = [c_idx]
        for pos in range(n):
            for alt in range(q):
                if alt == c_tup[pos]:
                    continue
                new_tup = list(c_tup)
                new_tup[pos] = alt
                new_idx = tuple_to_idx(tuple(new_tup))
                covered.append(new_idx)
        coverage.append(covered)

    # Use numpy bitsets for fast coverage tracking
    # 729 bits needs ceil(729/64) = 12 uint64 words
    n_bits = (total_words + 63) // 64

    def make_bitset(indices):
        bs = np.zeros(n_bits, dtype=np.uint64)
        for idx in indices:
            bs[idx // 64] |= (1 << (idx % 64))
        return bs

    # Precompute coverage bitsets
    coverage_bits = [make_bitset(covered) for covered in coverage]

    def bitset_popcount(bs):
        return int(np.sum(np.unpackbits(bs.view(np.uint8), axis=1), axis=0).sum())

    def bitset_and_popcount(a, b):
        return bitset_popcount(a & b)

    def bitset_or(a, b):
        return a | b

    def bitset_not(bs, total):
        # Invert only the relevant bits
        result = bs.copy()
        for i in range(n_bits):
            if i == n_bits - 1:
                # Mask to only valid bits
                valid_bits = total - i * 64
                mask = (1 << valid_bits) - 1
                result[i] = (~result[i]) & mask
            else:
                result[i] = ~result[i]
        return result

    # Greedy set cover
    def greedy_cover():
        uncovered = np.ones(n_bits, dtype=np.uint64)
        # Set all bits to 1 for valid word indices
        for i in range(n_bits):
            if i == n_bits - 1:
                valid_bits = total_words - i * 64
                uncovered[i] = (1 << valid_bits) - 1
            else:
                uncovered[i] = (1 << 64) - 1

        selected = []
        while True:
            if time.time() > deadline:
                break
            # Find codeword that covers most uncovered words
            best_c = -1
            best_count = -1
            for c in range(total_words):
                count = bitset_and_popcount(uncovered, coverage_bits[c])
                if count > best_count:
                    best_count = count
                    best_c = c
                if count == 0:
                    continue
            if best_c == -1 or best_count == 0:
                break
            selected.append(best_c)
            uncovered = uncovered & ~coverage_bits[best_c]
            # Check if all covered
            if np.all(uncovered == 0):
                break
        return selected

    # Local search: try to delete codewords
    def local_search_delete(codewords):
        # Build coverage of selected codewords
        covered = np.zeros(n_bits, dtype=np.uint64)
        for c in codewords:
            covered = covered | coverage_bits[c]

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                c = codewords[i]
                # Check if removing c still covers everything
                new_covered = covered & ~coverage_bits[c]
                # Check if new_covered still covers all words
                if np.all(new_covered == covered):
                    # Can remove c
                    codewords.pop(i)
                    covered = new_covered
                    changed = True
        return codewords

    # Better local search: for each codeword, check if its covered words are all covered by others
    def local_search_delete_v2(codewords):
        if len(codewords) == 0:
            return codewords

        # Build coverage matrix: for each selected codeword, what it covers
        # For each word, which selected codewords cover it?
        # Simpler: for each selected codeword c, check if all words in coverage_bits[c]
        # are also covered by at least one other selected codeword

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                c = codewords[i]
                # For each word that c covers, check if another selected codeword covers it
                c_covered = coverage_bits[c]
                # Build coverage of all other selected codewords
                others_covered = np.zeros(n_bits, dtype=np.uint64)
                for j in range(len(codewords)):
                    if j != i:
                        others_covered = others_covered | coverage_bits[codewords[j]]
                # Check if c_covered is subset of others_covered
                if np.all((c_covered & ~others_covered) == 0):
                    codewords.pop(i)
                    changed = True
        return codewords

    # Even better: precompute and use a smarter check
    def local_search_delete_v3(codewords):
        if len(codewords) == 0:
            return codewords

        # For each codeword, compute what it uniquely covers
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                c = codewords[i]
                # Check if any word in c's coverage is NOT covered by any other selected codeword
                unique_covered = np.zeros(n_bits, dtype=np.uint64)
                for j in range(len(codewords)):
                    if j != i:
                        unique_covered = unique_covered | coverage_bits[codewords[j]]
                # Words covered by c but not by others
                only_c = c_covered_check(coverage_bits[c], unique_covered)
                if np.all(only_c == 0):
                    codewords.pop(i)
                    changed = True
        return codewords

    def c_covered_check(c_bits, others_bits):
        return c_bits & ~others_bits

    # Run greedy + local search
    best_codewords = greedy_cover()
    best_codewords = local_search_delete_v3(best_codewords)
    best_count = len(best_codewords)

    # Try multiple restarts with random perturbation if time allows
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Perturb: start greedy but with a random bias or different ordering
        # Simple approach: shuffle the order in which we consider codewords
        # Or: start with a random subset and fill in

        # Alternative: take current best, remove a random codeword, and re-greedy
        if len(best_codewords) > 1:
            perturbed = best_codewords[:]
            # Remove 1-3 random codewords
            num_remove = min(random.randint(1, 3), len(perturbed) - 1)
            for _ in range(num_remove):
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)

            # Fill in: find uncovered words, greedily add
            covered = np.zeros(n_bits, dtype=np.uint64)
            for c in perturbed:
                covered = covered | coverage_bits[c]

            # Find uncovered
            uncovered_mask = np.zeros(n_bits, dtype=np.uint64)
            for i in range(n_bits):
                if i == n_bits - 1:
                    valid_bits = total_words - i * 64
                    uncovered_mask[i] = (1 << valid_bits) - 1
                else:
                    uncovered_mask[i] = (1 << 64) - 1
            uncovered_mask = uncovered_mask & ~covered

            # Greedily add to cover uncovered
            while True:
                if time.time() > deadline:
                    break
                best_c = -1
                best_count_new = -1
                for c in range(total_words):
                    if c in perturbed:
                        continue
                    count = bitset_and_popcount(uncovered_mask, coverage_bits[c])
                    if count > best_count_new:
                        best_count_new = count
                        best_c = c
                if best_c == -1 or best_count_new == 0:
                    break
                perturbed.append(best_c)
                uncovered_mask = uncovered_mask & ~coverage_bits[best_c]
                if np.all(uncovered_mask == 0):
                    break

            # Local search
            perturbed = local_search_delete_v3(perturbed)

            if len(perturbed) < best_count:
                best_count = len(perturbed)
                best_codewords = perturbed
                # Try more aggressive local search
                best_codewords = local_search_delete_v3(best_codewords)
                best_count = len(best_codewords)

    # Convert back to list of lists
    result = []
    for c_idx in best_codewords:
        result.append(list(idx_to_tuple(c_idx)))

    # Ensure distinctness (should be by construction but verify)
    seen = set()
    final = []
    for word in result:
        t = tuple(word)
        if t not in seen:
            seen.add(t)
            final.append(word)

    return final

# EVOLVE-BLOCK-END
