import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    if total_words == 1:
        return [[0] * n]

    # Precompute neighbors as a flat numpy array for vectorized access
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    q_powers = [q ** i for i in range(n)]

    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets for swap logic
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]

    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words

        while remaining > 0:
            if time.time() > deadline - 40.0:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]

            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])

            codewords.append(best_word)

            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1

        return codewords

    def ca_reorganize(count, in_code, cw, max_steps=20):
        """Synchronous cellular automaton pass for global reorganization."""
        nonlocal best_size, best_cw
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            
            # Compute coverage deficit
            critical = count == 1
            n_critical = np.sum(critical)
            
            if n_critical == 0:
                break
            
            # For each codeword, check if it can be removed (all neighbors have count > 1)
            removable = []
            for i, ci in enumerate(cw):
                if np.min(count[neighbors[ci]]) > 1:
                    removable.append(i)
            
            if removable:
                # Remove a random redundant codeword
                i = random.choice(removable)
                ci = cw[i]
                for u in neighbors[ci]:
                    count[u] -= 1
                in_code[ci] = False
                cw.pop(i)
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
                continue
            
            # No redundant codewords; try swaps for critical positions
            critical_indices = np.where(critical)[0]
            if len(critical_indices) == 0:
                break
                
            # Pick a random critical position
            crit_pos = int(random.choice(critical_indices))
            
            # Find which codeword covers crit_pos
            covering_idx = None
            for i, ci in enumerate(cw):
                if np.any(neighbors[ci] == crit_pos):
                    covering_idx = i
                    break
            
            if covering_idx is None:
                continue
            
            ci = cw[covering_idx]
            # Try to find a replacement codeword that covers crit_pos and has slack
            valid_replacements = []
            for cand in neighbor_sets[crit_pos]:
                if cand == ci or in_code[cand]:
                    continue
                # Check if replacing ci with cand maintains coverage
                temp_count = count.copy()
                for u in neighbors[ci]:
                    temp_count[u] -= 1
                for u in neighbors[cand]:
                    temp_count[u] += 1
                if np.min(temp_count) >= 1:
                    valid_replacements.append(cand)
            
            if valid_replacements:
                cand = random.choice(valid_replacements)
                for u in neighbors[ci]:
                    count[u] -= 1
                for u in neighbors[cand]:
                    count[u] += 1
                in_code[ci] = False
                in_code[cand] = True
                cw[covering_idx] = cand
        
        return count, in_code, cw

    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)

        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1

        stall_count = 0
        last_improvement_size = best_size

        while time.time() < time_limit:
            improved = False

            # Pass 1: Remove redundant codewords (vectorized check)
            indices = list(range(len(cw)))
            random.shuffle(indices)

            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]

                # Check if all neighbors have count > 1
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True

                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Pass 2: Swaps
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)

                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]

                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]

                    if len(critical) == 0:
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        break

                    if len(critical) > 0:
                        first_crit = critical[0]

                        if len(critical) == 1:
                            valid_cands = []
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                valid_cands.append(cand)

                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[ci]:
                                    count[u] -= 1
                                for u in neighbors[cand]:
                                    count[u] += 1

                                in_code[ci] = False
                                in_code[cand] = True
                                cw[i] = cand
                                improved = True
                                break
                        else:
                            # Multiple critical: need candidate in intersection
                            candidate_set = neighbor_sets[first_crit].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break

                            if candidate_set:
                                valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                                if valid_cands:
                                    cand = random.choice(valid_cands)
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1

                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    break

            if improved:
                stall_count = 0
                if len(cw) < last_improvement_size:
                    last_improvement_size = len(cw)
            else:
                stall_count += 1
                
                # Try CA reorganization when stalled
                if stall_count >= 3 and len(cw) > 2:
                    count, in_code, cw = ca_reorganize(count, in_code, cw, max_steps=30)
                    stall_count = 0
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
                elif stall_count >= 5:
                    # Aggressive perturbation: remove and re-add random codewords
                    if len(cw) > 2:
                        idx1 = random.randint(0, len(cw) - 1)
                        c1 = cw[idx1]
                        
                        # Find a replacement
                        critical_mask = count[neighbors[c1]] == 1
                        critical = neighbors[c1][critical_mask]
                        
                        if len(critical) == 0:
                            for u in neighbors[c1]:
                                count[u] -= 1
                            in_code[c1] = False
                            cw.pop(idx1)
                            improved = True
                        else:
                            # Try random swaps
                            for _ in range(10):
                                idx2 = random.randint(0, len(cw) - 1)
                                c2 = cw[idx2]
                                temp_count = count.copy()
                                for u in neighbors[c1]:
                                    temp_count[u] -= 1
                                for u in neighbors[c2]:
                                    temp_count[u] += 1
                                if np.min(temp_count) >= 1:
                                    for u in neighbors[c1]:
                                        count[u] -= 1
                                    for u in neighbors[c2]:
                                        count[u] += 1
                                    in_code[c1] = False
                                    in_code[c2] = True
                                    cw[idx1] = c2
                                    improved = True
                                    break
                    stall_count = 0

        return best_cw

    best_code = None
    best_size = total_words + 1

    # Run multiple restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        code = greedy_init(i)
        if code is None or len(code) == 0:
            continue
            
        # Only run local search if we have time
        remaining_time = deadline - 15.0 - time.time()
        if remaining_time > 5.0:
            code = local_search(code, time.time() + remaining_time)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code

    if best_code is None:
        # Fallback: Greedy without local search
        code = greedy_init(0)
        if code:
            best_code = code
        else:
            # Very basic fallback
            import itertools
            # Tetracode construction for n=8, q=3
            if q == 3 and n == 8:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                # Simple greedy
                covered = set()
                best_code = []
                for w in range(total_words):
                    if w not in covered:
                        best_code.append(w)
                        covered.update(neighbors[w])
                        
    # Convert indices back to words
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    return [idx_to_word(c) for c in best_code]
