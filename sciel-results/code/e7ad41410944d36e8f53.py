"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for i, v in enumerate(word):
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_ball_indices(word_idx, q, n, radius):
    """Generate all word indices within Hamming distance `radius` from `word_idx`."""
    word = _int_to_word(word_idx, q, n)
    ball = set()
    ball.add(word_idx)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(_word_to_int(new_word, q, n))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave safety margin before 150s kill
    
    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
    # Precompute balls as bitmasks (using Python's arbitrary-precision ints)
    balls = [0] * total_words
    for idx in range(total_words):
        ball_indices = _generate_ball_indices(idx, q, n, radius)
        mask = 0
        for i in ball_indices:
            mask |= (1 << i)
        balls[idx] = mask
    
    all_masks = balls  # alias
    
    best_solution = None
    best_size = float('inf')
    
    def _build_coverage_count(sol):
        """Build coverage count array for given solution."""
        cov = [0] * total_words
        for idx in sol:
            m = all_masks[idx]
            # Iterate through set bits in m
            while m:
                bit = m & (-m)
                pos = bit.bit_length() - 1
                cov[pos] += 1
                m ^= bit
        return cov
    
    def _try_remove_redundant_fast(sol):
        """Remove redundant codewords using coverage counts. O(k * ball_size) total."""
        if not sol:
            return sol
        cov = _build_coverage_count(sol)
        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            for i in range(len(sol)):
                idx = sol[i]
                # Check if any word has coverage count == 1 (only covered by this codeword)
                m = all_masks[idx]
                has_unique = False
                tmp = m
                while tmp:
                    bit = tmp & (-tmp)
                    pos = bit.bit_length() - 1
                    if cov[pos] == 1:
                        has_unique = True
                        break
                    tmp ^= bit
                if not has_unique:
                    # This codeword is redundant, remove it
                    # Update coverage counts
                    tmp2 = all_masks[idx]
                    while tmp2:
                        bit = tmp2 & (-tmp2)
                        pos = bit.bit_length() - 1
                        cov[pos] -= 1
                        tmp2 ^= bit
                    sol.pop(i)
                    improved = True
                    break
        return sol
    
    def _local_search_move(sol, rng, max_iters=100):
        """Move-based local search with coverage count tracking."""
        if len(sol) < 2:
            return sol
        cov = _build_coverage_count(sol)
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            i = rng.randint(0, len(sol) - 1)
            old_idx = sol[i]
            
            # Temporarily remove old_idx from coverage
            tmp = all_masks[old_idx]
            while tmp:
                bit = tmp & (-tmp)
                pos = bit.bit_length() - 1
                cov[pos] -= 1
                tmp ^= bit
            
            # Find words with coverage count 0 (uncovered)
            uncovered_mask = 0
            for pos in range(total_words):
                if cov[pos] == 0:
                    uncovered_mask |= (1 << pos)
            
            if uncovered_mask == 0:
                # old_idx was redundant, keep it removed
                sol.pop(i)
                sol = _try_remove_redundant_fast(sol)
                cov = _build_coverage_count(sol)
                continue
            
            # Try to find a replacement that covers all uncovered words
            found = False
            for _ in range(50):
                if time.time() > deadline - 20:
                    break
                new_idx = rng.randint(0, total_words - 1)
                if new_idx in sol:
                    continue
                if (all_masks[new_idx] & uncovered_mask) == uncovered_mask:
                    # Accept replacement
                    sol[i] = new_idx
                    # Update coverage: add new_idx
                    tmp2 = all_masks[new_idx]
                    while tmp2:
                        bit = tmp2 & (-tmp2)
                        pos = bit.bit_length() - 1
                        cov[pos] += 1
                        tmp2 ^= bit
                    found = True
                    break
            
            if not found:
                # Restore old_idx coverage
                tmp3 = all_masks[old_idx]
                while tmp3:
                    bit = tmp3 & (-tmp3)
                    pos = bit.bit_length() - 1
                    cov[pos] += 1
                    tmp3 ^= bit
        
        return sol
    
    def _add_remove_search(sol, rng, max_iters=50):
        """Try adding a word then removing up to 2 to achieve net reduction."""
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            new_idx = rng.randint(0, total_words - 1)
            if new_idx in sol:
                continue
            
            sol.append(new_idx)
            cov = _build_coverage_count(sol)
            
            removed = 0
            for i in range(len(sol) - 1, -1, -1):
                if removed >= 2:
                    break
                idx = sol[i]
                m = all_masks[idx]
                has_unique = False
                tmp = m
                while tmp:
                    bit = tmp & (-tmp)
                    pos = bit.bit_length() - 1
                    if cov[pos] == 1:
                        has_unique = True
                        break
                    tmp ^= bit
                if not has_unique:
                    tmp2 = all_masks[idx]
                    while tmp2:
                        bit = tmp2 & (-tmp2)
                        pos = bit.bit_length() - 1
                        cov[pos] -= 1
                        tmp2 ^= bit
                    sol.pop(i)
                    removed += 1
            
            if removed < 1:
                # Revert the addition
                if new_idx in sol:
                    sol.remove(new_idx)
        
        return sol
    
    # Try multiple random restarts with greedy + local search
    num_restarts = 80
    for attempt in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(12345 + attempt)
        
        # Greedy set cover using bitmasks with bit_count for speed
        current_solution = []
        covered_mask = 0
        in_solution = set()
        
        # Start with a random word
        first_idx = rng.randint(0, total_words - 1)
        current_solution.append(first_idx)
        in_solution.add(first_idx)
        covered_mask |= all_masks[first_idx]
        
        # Greedy: repeatedly add the word that covers the most uncovered words
        while covered_mask != full_mask:
            if time.time() > deadline - 20:
                break
            
            uncovered_mask = full_mask ^ covered_mask
            best_gain = -1
            best_idx = -1
            
            for i in range(total_words):
                if i in in_solution:
                    continue
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(best_idx)
            in_solution.add(best_idx)
            covered_mask |= all_masks[best_idx]
        
        # Local search with fast coverage-count-based methods
        current_solution = _try_remove_redundant_fast(current_solution)
        current_solution = _local_search_move(current_solution, rng, max_iters=150)
        current_solution = _add_remove_search(current_solution, rng, max_iters=80)
        current_solution = _try_remove_redundant_fast(current_solution)
        
        # Validate and update best
        final_mask = 0
        for idx in current_solution:
            final_mask |= all_masks[idx]
        
        if final_mask == full_mask:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [_int_to_word(idx, q, n) for idx in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
