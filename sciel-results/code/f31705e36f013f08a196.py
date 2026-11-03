import numpy as np
import itertools
import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, aim to finish by 145s
    deadline = start_time + 145
    
    N = q ** n
    
    # Fast index conversions
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbor masks and lists
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    for c_idx in range(N):
        c_word = all_words[c_idx]
        mask = 1 << c_idx
        neighbor_lists[c_idx].append(c_idx)
        
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
        else:
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
                        
        neighbor_masks[c_idx] = mask

    # Precompute: for each word, which codewords cover it (inverse mapping)
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)

    def greedy_cover_criticality():
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # Track coverage count for each word
        cover_count = np.zeros(N, dtype=np.int32)
        
        # Precompute criticality for all words
        criticality = np.array([len(word_to_coders[w]) for w in range(N)], dtype=np.int32)
        
        while uncovered:
            if time.time() > deadline - 20:
                break
            
            # Get list of uncovered word indices
            uncovered_words = []
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                uncovered_words.append(w_idx)
                temp ^= lsb
            
            if not uncovered_words:
                break
            
            # Find word with minimum criticality among uncovered
            min_crit = N + 1
            target_word = -1
            
            for w_idx in uncovered_words:
                crit = criticality[w_idx]
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:  # Can't get better
                    break
            
            if target_word == -1:
                break
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                
                # Count how many new words this codeword would cover
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                # Fallback: pick any codeword that covers target_word
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
            
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords - vectorized check"""
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Vectorized check: any word in c's neighborhood with cover_count == 1?
                if not np.any(cover_counts[neighbor_lists[c]] == 1):
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def simulated_annealing(code, time_budget):
        """
        Simulated Annealing to minimize code size while maintaining coverage.
        Uses bitmask for fast delta calculations.
        """
        code = list(code)
        code_set = set(code)
        
        # Initial state
        covered_mask = 0
        for c in code:
            covered_mask |= neighbor_masks[c]
        
        uncovered_count = N - covered_mask.bit_count()
        P = 10000  # Penalty for uncovered words
        
        current_energy = len(code) + P * uncovered_count
        best_energy = current_energy
        best_code = code[:]
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < start_time + time_budget and T > T_min:
            r = random.random()
            
            if r < 0.4:
                # Move: Remove a random codeword
                if len(code) == 0:
                    continue
                c = random.choice(code)
                
                # Calculate new coverage
                # If c is the only one covering some words, they become uncovered
                # We need to check if removing c increases uncovered_count
                # new_uncovered = words in ball(c) that are NOT covered by any other c' in code\{c}
                
                # Efficient check:
                # Let M_c be mask of ball(c).
                # Let M_others be mask of union of balls of all other codewords.
                # If (M_c & ~M_others) is non-zero, then we lose coverage.
                
                # To avoid computing M_others every time, we can use cover_counts?
                # No, cover_counts is slow to update.
                # Let's stick to bitmask union for others? That's O(|code|) bit ops.
                # |code| ~ 500, bit ops on 6561 bits are fast.
                
                # Alternative: Just compute the change in uncovered_count directly.
                # delta_uncovered = count of words w in ball(c) such that cover_count[w] == 1.
                # We don't have cover_count here. Let's maintain it?
                # Maintaining cover_count in SA is expensive (update on every move).
                
                # Let's try a different approach for SA moves:
                # 1. Remove c.
                # 2. Check if coverage is lost.
                #    If yes, energy increases by P * lost_coverage.
                #    If no, energy decreases by 1.
                
                # To check coverage loss efficiently:
                # We can iterate through ball(c) and check if any other codeword covers it.
                # This is O(|ball(c)| * |code|) which is 17 * 500 = 8500 checks. Too slow?
                # Maybe.
                
                # Let's use the bitmask approach but optimize the "others" mask.
                # It's hard to maintain "others" mask dynamically.
                
                # Let's go back to cover_counts but update them efficiently?
                # Actually, for N=6561, bit operations are very fast in Python integers.
                # Let's compute the union of all other masks.
                
                others_mask = 0
                for c2 in code:
                    if c2 != c:
                        others_mask |= neighbor_masks[c2]
                
                # Words in ball(c) that are NOT in others_mask
                lost_words_mask = neighbor_masks[c] & ~others_mask
                lost_count = lost_words_mask.bit_count()
                
                delta_energy = 1 + P * lost_count # Wait, removing reduces size by 1, so -1.
                # Energy = Size + P * Uncovered
                # New Size = Size - 1
                # New Uncovered = Uncovered + lost_count
                # Delta E = -1 + P * lost_count
                
                if lost_count == 0:
                    delta_energy = -1
                else:
                    delta_energy = -1 + P * lost_count
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    # Accept
                    code.remove(c)
                    code_set.discard(c)
                    covered_mask &= ~neighbor_masks[c] # This is wrong, covered_mask is union.
                    # We need to recalculate covered_mask or track it differently.
                    # Recalculating covered_mask is O(|code|).
                    # Let's just recalculate uncovered_count?
                    # No, let's just track energy and best code.
                    # We don't strictly need covered_mask for acceptance, just delta_energy.
                    # But we need to know if we found a better valid solution.
                    
                    current_energy += delta_energy
                    
                    if lost_count == 0 and current_energy < best_energy:
                        best_energy = current_energy
                        best_code = code[:]
                        
            elif r < 0.7:
                # Move: Add a random codeword
                # Prefer adding near uncovered areas?
                # For now, random.
                w = random.randint(0, N - 1)
                if w in code_set:
                    continue
                
                # Calculate gain in coverage
                # Words in ball(w) that are currently uncovered
                # We need to know which words are uncovered.
                # This requires maintaining a global uncovered mask or similar.
                
                # Let's maintain `uncovered_mask` in the SA loop.
                # Initial uncovered_mask = ~covered_mask
                # When adding w:
                # new_covered = neighbor_masks[w] & uncovered_mask
                # gain = new_covered.bit_count()
                # Delta E = 1 - P * gain
                
                # We need to maintain uncovered_mask.
                # Let's refactor to maintain `uncovered_mask` as a Python int.
                
                pass # This branch is complex without state. Let's implement stateful SA.
            
            T *= cooling
            
        return best_code

    # Stateful Simulated Annealing
    def stateful_simulated_annealing(code, time_budget):
        code = list(code)
        code_set = set(code)
        
        # Calculate initial covered mask
        covered_mask = 0
        for c in code:
            covered_mask |= neighbor_masks[c]
        
        uncovered_mask = ((1 << N) - 1) & ~covered_mask
        uncovered_count = uncovered_mask.bit_count()
        
        P = 10000
        current_energy = len(code) + P * uncovered_count
        best_energy = current_energy
        best_code = code[:]
        
        T = 5.0
        T_min = 0.05
        cooling = 0.9999
        
        while time.time() < start_time + time_budget and T > T_min:
            r = random.random()
            
            if r < 0.5:
                # Remove
                if len(code) == 0:
                    continue
                c = random.choice(code)
                
                # Check if removing c leaves any words uncovered
                # Words in ball(c) that are ONLY covered by c
                # These are words w in ball(c) such that w is not in union(ball(c') for c' in code\{c})
                
                # Efficient check:
                # Let M_c = neighbor_masks[c]
                # Let M_others = 0
                # for c2 in code: if c2 != c: M_others |= neighbor_masks[c2]
                # Lost = M_c & ~M_others
                
                # This is O(|code|) big-int ORs.
                # |code| ~ 500. 500 ORs of 6561-bit ints.
                # Python big-int OR is fast (C level). 500 ops is ~microseconds.
                
                M_others = 0
                for c2 in code:
                    if c2 != c:
                        M_others |= neighbor_masks[c2]
                
                lost_mask = neighbor_masks[c] & ~M_others
                lost_count = lost_mask.bit_count()
                
                delta_energy = -1 + P * lost_count
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    code.remove(c)
                    code_set.discard(c)
                    
                    # Update masks
                    # covered_mask changes?
                    # The words in lost_mask become uncovered.
                    uncovered_mask |= lost_mask
                    # The words in (neighbor_masks[c] & ~lost_mask) remain covered by others, so covered_mask stays same for them?
                    # Actually, covered_mask is the union. Removing c might remove some bits from covered_mask if they were unique to c.
                    # Those bits are exactly lost_mask.
                    # So covered_mask &= ~lost_mask is correct?
                    # Yes, because if a bit was in covered_mask and in ball(c), and it's in lost_mask, it means no other codeword covers it.
                    # If it's NOT in lost_mask, some other codeword covers it, so it remains in covered_mask.
                    
                    covered_mask &= ~lost_mask
                    uncovered_count += lost_count
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = code[:]
                        
            elif r < 0.8:
                # Add
                # Pick a random word, or a word near uncovered?
                # If we pick random, we might add useless codewords.
                # Let's pick a random uncovered word, and add a random neighbor of it.
                if uncovered_count > 0:
                    # Pick random bit from uncovered_mask
                    # This is tricky. Let's just pick a random index from 0..N-1 and check if it's uncovered?
                    # Or iterate?
                    # Let's pick a random codeword candidate w.
                    w = random.randint(0, N - 1)
                    
                    if w in code_set:
                        continue
                    
                    # Calculate gain
                    # Words in ball(w) that are currently uncovered
                    gain_mask = neighbor_masks[w] & uncovered_mask
                    gain_count = gain_mask.bit_count()
                    
                    if gain_count == 0:
                        # Adding this doesn't help coverage, only increases size
                        # Delta E = 1
                        # Only accept if T is high enough
                        delta_energy = 1
                    else:
                        delta_energy = 1 - P * gain_count
                    
                    if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                        code.append(w)
                        code_set.add(w)
                        
                        # Update masks
                        # gain_mask becomes covered
                        covered_mask |= gain_mask
                        uncovered_mask &= ~gain_mask
                        uncovered_count -= gain_count
                        current_energy += delta_energy
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = code[:]
                        
            else:
                # Swap: Remove one, Add one
                if len(code) == 0:
                    continue
                c_remove = random.choice(code)
                w_add = random.randint(0, N - 1)
                
                if w_add in code_set and w_add != c_remove:
                    continue
                if w_add == c_remove:
                    continue
                
                # Calculate combined effect
                # 1. Remove c_remove
                # 2. Add w_add
                
                # This is complex to calculate delta in one go.
                # Let's simulate:
                # New code = code - {c_remove} + {w_add}
                
                # M_others_remove = union(ball(c) for c in code\{c_remove})
                M_others_remove = 0
                for c2 in code:
                    if c2 != c_remove:
                        M_others_remove |= neighbor_masks[c2]
                
                lost_mask_remove = neighbor_masks[c_remove] & ~M_others_remove
                lost_count_remove = lost_mask_remove.bit_count()
                
                # New covered mask after removal
                temp_covered = covered_mask & ~lost_mask_remove
                temp_uncovered = ((1 << N) - 1) & ~temp_covered
                
                # Now add w_add
                # Gain is words in ball(w_add) that are in temp_uncovered
                gain_mask = neighbor_masks[w_add] & temp_uncovered
                gain_count = gain_mask.bit_count()
                
                # Net change in uncovered count
                # Uncovered increases by lost_count_remove
                # Uncovered decreases by gain_count
                net_delta_uncovered = lost_count_remove - gain_count
                
                # Size change: -1 + 1 = 0
                delta_energy = P * net_delta_uncovered
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    # Accept
                    code.remove(c_remove)
                    code_set.discard(c_remove)
                    code.append(w_add)
                    code_set.add(w_add)
                    
                    # Update masks
                    # Covered mask: remove lost_mask_remove, add gain_mask
                    # Note: gain_mask is calculated based on temp_uncovered, which is correct.
                    # But we need to be careful with overlaps.
                    
                    # Let's just recalculate covered_mask from scratch? No, too slow.
                    # Let's update incrementally.
                    
                    # 1. Remove c_remove's contribution that is unique
                    covered_mask &= ~lost_mask_remove
                    # 2. Add w_add's contribution that is new
                    covered_mask |= gain_mask
                    
                    # Update uncovered
                    uncovered_mask = ((1 << N) - 1) & ~covered_mask
                    uncovered_count = uncovered_mask.bit_count()
                    
                    current_energy = len(code) + P * uncovered_count
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = code[:]
                        
            T *= cooling
            
        return best_code

    # Run criticality-based greedy
    best_code = greedy_cover_criticality()
    best_code = local_search_remove(best_code)
    
    best_size = len(best_code)
    
    # Run Simulated Annealing
    # Allocate remaining time to SA
    elapsed = time.time() - start_time
    time_left = deadline - elapsed
    
    if time_left > 10:
        # Use most of the time for SA
        sa_time = time_left - 5 # 5s buffer for final checks
        polished_code = stateful_simulated_annealing(best_code, sa_time)
        
        if len(polished_code) < best_size:
            best_code = polished_code
            best_size = len(best_code)
            
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            # Try to find a codeword that covers the most uncovered words
            # Sample candidates to speed up
            candidates = random.sample(range(N), min(100, N))
            for c in candidates:
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            
            if best_c is None or best_count == 0:
                # Fallback: exhaustive search (slow)
                for c in range(N):
                    if c in best_code:
                        continue
                    count = (uncovered & neighbor_masks[c]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_c = c
            
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [all_words[i] for i in best_code]
    return result
