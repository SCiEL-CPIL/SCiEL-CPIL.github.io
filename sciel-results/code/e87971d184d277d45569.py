import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    total_space = q ** n

    # Precompute words and indices
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute cover_bits: for each word i, the bitmask of all words within radius
    # Also precompute coverers: for each word w, the bitmask of all codewords covering w
    # (for radius 1, a word w is covered by codeword i iff i is within radius of w,
    #  which is symmetric, so coverers[w] = cover_bits[w])
    cover_bits = [0] * total_space

    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Compute index of modified word
                    # word_to_idx: idx = sum(v[k] * q^(n-1-k))
                    # For position pos, the contribution is (val - orig_val) * q^(n-1-pos)
                    delta = (val - orig_val) * (q ** (n - 1 - pos))
                    j = i + delta
                    mask |= (1 << j)
        cover_bits[i] = mask

    # coverers[w] = cover_bits[w] by symmetry of Hamming distance
    coverers = cover_bits  # same array, just different interpretation

    total_mask = (1 << total_space) - 1

    # Precompute popcount via numpy for speed on big ints
    def popcount_big(x):
        if x == 0:
            return 0
        # Use int.bit_count() (Python 3.10+)
        return x.bit_count()

    def greedy_cover(randomize=False):
        uncovered = total_mask
        code = []
        # For speed, maintain a list of candidate indices
        candidates = list(range(total_space))
        
        while uncovered and time.time() < deadline - 5.0:
            if not candidates:
                break
                
            if randomize:
                # Randomized greedy: pick from top-k by gain
                # For speed, sample a subset if too many
                if len(candidates) > 200:
                    sample = random.sample(candidates, 200)
                else:
                    sample = candidates
                
                best_gain = -1
                best_idx = -1
                for i in sample:
                    gain = popcount_big(cover_bits[i] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            else:
                best_gain = -1
                best_idx = -1
                for i in candidates:
                    gain = popcount_big(cover_bits[i] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            
            if best_gain <= 0:
                break
            
            code.append(best_idx)
            uncovered &= ~cover_bits[best_idx]
            
            # Remove dominated candidates (those whose cover is subset of best's)
            # This is expensive, so skip for now; just remove the chosen one
            candidates.remove(best_idx)
            # Also remove any candidate whose cover is subset of the new codeword's cover
            # This helps a lot but is O(n * bit_ops) per step
            if len(candidates) < 500:
                new_cover = cover_bits[best_idx]
                to_remove = []
                for i in candidates:
                    if (cover_bits[i] & uncovered) | (cover_bits[i] & ~uncovered) == 0:
                        pass  # can't happen
                    # If cover_bits[i] is subset of (new_cover | already_covered), skip
                    # Simpler: if cover_bits[i] & uncovered is subset of new_cover & uncovered
                    if (cover_bits[i] & uncovered) & ~new_cover == 0:
                        to_remove.append(i)
                for i in to_remove:
                    candidates.remove(i)
        
        return code, uncovered

    def local_search_reduce(code, uncovered):
        if not code:
            return code, uncovered

        # Build cover_count using numpy for speed
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code:
            m = cover_bits[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] += 1
                m ^= lsb

        improved = True
        passes = 0
        while improved and time.time() < deadline - 3.0:
            improved = False
            passes += 1
            if passes > 50:
                break
            random.shuffle(code)
            n_cw = len(code)
            
            for i in range(n_cw):
                idx = code[i]
                mask = cover_bits[idx]
                
                # Check if removable
                is_removable = True
                m = mask
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        is_removable = False
                        break
                    m ^= lsb
                
                if is_removable:
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Find unique words (covered only by this codeword)
                    unique_mask = 0
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        if cover_count[bit_idx] == 1:
                            unique_mask |= lsb
                        m ^= lsb
                    
                    if unique_mask:
                        # Try to find a replacement codeword that covers all unique words
                        # For each unique word w, find codewords that cover w
                        # The replacement must cover ALL unique words
                        # Start with the set of candidates = intersection of coverers[w] for each unique w
                        # But this is expensive. Instead, try swapping with a codeword
                        # that covers at least one unique word, then check if all are covered.
                        
                        # Get the first unique word
                        lsb = unique_mask & -unique_mask
                        first_unique = lsb.bit_length() - 1
                        
                        # Candidates are codewords that cover first_unique
                        candidate_mask = coverers[first_unique]
                        # Remove current code from candidates
                        # For each candidate, check if it covers all unique words
                        found_swap = False
                        cm = candidate_mask
                        while cm and not found_swap:
                            lsb2 = cm & -cm
                            cand_idx = lsb2.bit_length() - 1
                            cm ^= lsb2
                            
                            if cand_idx in code:
                                continue
                            
                            # Check if cand covers all unique words
                            cand_cover = cover_bits[cand_idx]
                            if (unique_mask & ~cand_cover) == 0:
                                # Swap!
                                # Remove old
                                m = mask
                                while m:
                                    lsb3 = m & -m
                                    bit_idx = lsb3.bit_length() - 1
                                    cover_count[bit_idx] -= 1
                                    m ^= lsb3
                                # Add new
                                m = cand_cover
                                while m:
                                    lsb4 = m & -m
                                    bit_idx = lsb4.bit_length() - 1
                                    cover_count[bit_idx] += 1
                                    m ^= lsb4
                                code[i] = cand_idx
                                improved = True
                                found_swap = True
                                break
                        
                        if found_swap:
                            break
                            
        return code, uncovered

    def simulated_annealing(code, uncovered):
        if not code:
            return code, uncovered

        current_code = code[:]
        current_uncovered = uncovered

        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            m = cover_bits[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] += 1
                m ^= lsb

        def get_energy(c_uncovered, c_code):
            return len(c_code) + 1000 * popcount_big(c_uncovered)

        current_energy = get_energy(current_uncovered, current_code)
        best_code = current_code[:]
        best_uncovered = current_uncovered
        best_energy = current_energy

        T_start = 100.0
        T_end = 0.5
        cooling_rate = 0.9995

        T = T_start
        steps = 0

        while T > T_end and time.time() < deadline - 2.0:
            steps += 1
            move_type = random.random()

            if move_type < 0.3:
                # Add a random codeword
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= cooling_rate
                    continue

                new_uncovered = current_uncovered & ~cover_bits[new_idx]
                new_code = current_code + [new_idx]
                new_energy = get_energy(new_uncovered, new_code)

                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / max(T, 0.01)):
                    m = cover_bits[new_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] += 1
                        m ^= lsb

                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered

            elif move_type < 0.6:
                # Remove a random codeword
                if len(current_code) <= 1:
                    T *= cooling_rate
                    continue

                remove_pos = random.randint(0, len(current_code) - 1)
                idx = current_code[remove_pos]

                unique_mask = 0
                m = cover_bits[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb

                new_uncovered = current_uncovered | unique_mask
                new_code = current_code[:remove_pos] + current_code[remove_pos+1:]
                new_energy = get_energy(new_uncovered, new_code)

                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / max(T, 0.01)):
                    m = cover_bits[idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb

                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered

            else:
                # Swap
                if not current_code:
                    T *= cooling_rate
                    continue

                remove_pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[remove_pos]
                new_idx = random.randint(0, total_space - 1)

                if new_idx == old_idx:
                    T *= cooling_rate
                    continue

                unique_old = 0
                m = cover_bits[old_idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_old |= lsb
                    m ^= lsb

                temp_uncovered = current_uncovered | unique_old
                new_uncovered = temp_uncovered & ~cover_bits[new_idx]

                new_code = current_code[:]
                new_code[remove_pos] = new_idx
                new_energy = get_energy(new_uncovered, new_code)

                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / max(T, 0.01)):
                    m = cover_bits[old_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb

                    m = cover_bits[new_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] += 1
                        m ^= lsb

                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered

            T *= cooling_rate

        if best_uncovered == 0:
            return best_code, best_uncovered
        else:
            if current_uncovered == 0:
                return current_code, current_uncovered
            else:
                return code, uncovered

    # Main search loop
    best_code = None
    best_len = total_space + 1

    # First: deterministic greedy
    code, unc = greedy_cover(randomize=False)
    code, _ = local_search_reduce(code, unc)
    if unc == 0:
        best_code = code
        best_len = len(code)

    # Multiple randomized restarts
    while time.time() < deadline - 25.0:
        code, unc = greedy_cover(randomize=True)
        code, _ = local_search_reduce(code, unc)
        
        if unc == 0:
            if len(code) < best_len:
                best_code = code
                best_len = len(code)
                # Try to improve with SA
                if time.time() < deadline - 10.0:
                    sa_code, sa_unc = simulated_annealing(best_code, 0)
                    if sa_unc == 0 and len(sa_code) < best_len:
                        best_code = sa_code
                        best_len = len(sa_code)
        else:
            # Try SA to fix uncovered
            if time.time() < deadline - 10.0:
                sa_code, sa_unc = simulated_annealing(code, unc)
                if sa_unc == 0 and len(sa_code) < best_len:
                    best_code = sa_code
                    best_len = len(sa_code)

    # Final polish
    if best_code is not None:
        best_code, _ = local_search_reduce(best_code, 0)
        if time.time() < deadline - 5.0:
            sa_code, sa_unc = simulated_annealing(best_code, 0)
            if sa_unc == 0 and len(sa_code) < len(best_code):
                best_code = sa_code
        best_code, _ = local_search_reduce(best_code, 0)

    if best_code is None:
        # Fallback: return a trivial covering (all words)
        best_code = list(range(total_space))

    return [all_words[i] for i in best_code]
