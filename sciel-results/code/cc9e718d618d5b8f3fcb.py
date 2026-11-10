import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Handle general case fallback if not the specific task
    if not (q == 3 and n == 8 and radius == 1):
        # Simple greedy fallback for robustness
        N = q ** n
        if N > 100000:
            # Too large, return all words (trivial upper bound)
            return [[i % q for i in range(n)] for i in range(N)]
        
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    # Specific implementation for q=3, n=8, radius=1
    N = q ** n  # 6561
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors
    neighbors = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                diff = new_d - d
                new_w = w + diff * powers[i]
                neighbors[w, idx] = new_w
                idx += 1

    def compute_coverage(in_code):
        """Compute coverage count for each word given a boolean in_code array."""
        cov = np.zeros(N, dtype=np.int32)
        for w in np.where(in_code)[0]:
            cov[neighbors[w, :]] += 1
        return cov

    def local_search(in_code, rng, deadline):
        """
        Local search on a valid code (in_code boolean array).
        Uses deletion and swap moves to minimize size while maintaining validity.
        Returns (best_in_code, best_size).
        """
        current_in_code = in_code.copy()
        current_coverage = compute_coverage(current_in_code)
        current_size = int(np.sum(current_in_code))
        
        best_in_code = current_in_code.copy()
        best_size = current_size
        
        # Parameters
        T = 5.0
        alpha = 0.998
        min_T = 0.01
        M = 100  # Penalty for uncovered words (lower to allow more exploration)
        
        sa_deadline = deadline - 2.0
        
        # Maintain a list of code words for faster selection
        code_list = np.where(current_in_code)[0].tolist()
        
        while time.time() < sa_deadline:
            if T < min_T:
                if rng.random() < 0.1:
                    T = 1.0
                else:
                    break
            
            if len(code_list) == 0:
                break
            
            # Random move selection
            move_type = rng.random()
            
            if move_type < 0.65:
                # Deletion - most important for minimization
                w = code_list[rng.integers(len(code_list))]
                
                nbs = neighbors[w, :]
                min_cov = np.min(current_coverage[nbs])
                
                if min_cov > 1:
                    # Safe deletion - always accept
                    current_in_code[w] = False
                    code_list.remove(w)
                    current_coverage[nbs] -= 1
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_in_code = current_in_code.copy()
                else:
                    # Check if deletion causes uncovered words
                    newly_uncovered = np.sum(current_coverage[nbs] == 1)
                    if newly_uncovered == 0:
                        current_in_code[w] = False
                        code_list.remove(w)
                        current_coverage[nbs] -= 1
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_in_code = current_in_code.copy()
                    else:
                        # Try SA acceptance for invalid deletion
                        delta_cost = M * newly_uncovered - 1
                        if rng.random() < np.exp(-delta_cost / T):
                            current_in_code[w] = False
                            code_list.remove(w)
                            current_coverage[nbs] -= 1
                            current_size -= 1
                        # else: don't change
                    
            elif move_type < 0.85:
                # Swap: Remove w1, add w2
                if len(code_list) == 0:
                    break
                w1 = code_list[rng.integers(len(code_list))]
                
                # Find candidate w2 that covers essential words of w1
                nbs1 = neighbors[w1, :]
                essential_mask = (current_coverage[nbs1] == 1)
                essential_words = nbs1[essential_mask]
                
                if len(essential_words) > 0:
                    # Candidate must cover all essential words
                    cand_set = set(neighbors[essential_words[0], :].tolist())
                    for ew in essential_words[1:]:
                        cand_set.intersection_update(neighbors[ew, :].tolist())
                    candidates = [c for c in cand_set if not current_in_code[c]]
                    
                    if not candidates:
                        T *= alpha
                        continue
                    w2 = candidates[rng.integers(len(candidates))]
                else:
                    # No essential words, just pick a random non-code word
                    non_code_mask = ~current_in_code
                    if not np.any(non_code_mask):
                        T *= alpha
                        continue
                    non_code = np.where(non_code_mask)[0]
                    w2 = non_code[rng.integers(len(non_code))]
                
                nbs1 = neighbors[w1, :]
                nbs2 = neighbors[w2, :]
                
                prev_cov1 = current_coverage[nbs1].copy()
                prev_cov2 = current_coverage[nbs2].copy()
                
                # Apply swap
                current_coverage[nbs1] -= 1
                current_coverage[nbs2] += 1
                
                # Check validity
                if np.any(current_coverage == 0):
                    # Invalid swap, revert
                    current_coverage[nbs1] = prev_cov1
                    current_coverage[nbs2] = prev_cov2
                    T *= alpha
                    continue
                
                current_in_code[w1] = False
                current_in_code[w2] = True
                code_list.remove(w1)
                code_list.append(w2)
                # Size doesn't change, but structure does.
                # Accept to explore - this helps escape local minima
                
            else:
                # Double swap: Remove w1, w2, add w3
                if len(code_list) < 2:
                    T *= alpha
                    continue
                
                idx1 = rng.integers(len(code_list))
                idx2 = rng.integers(len(code_list) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                w1 = int(code_list[idx1])
                w2 = int(code_list[idx2])
                
                nbs1 = neighbors[w1, :]
                nbs2 = neighbors[w2, :]
                
                prev_cov1 = current_coverage[nbs1].copy()
                prev_cov2 = current_coverage[nbs2].copy()
                
                # Apply removals
                current_coverage[nbs1] -= 1
                current_coverage[nbs2] -= 1
                
                # Find newly uncovered words
                newly_uncovered_mask = (current_coverage == 0)
                newly_uncovered_words = np.where(newly_uncovered_mask)[0]
                
                if len(newly_uncovered_words) == 0:
                    # Both removals safe
                    current_in_code[w1] = False
                    current_in_code[w2] = False
                    code_list.remove(w1)
                    code_list.remove(w2)
                    current_size -= 2
                    if current_size < best_size:
                        best_size = current_size
                        best_in_code = current_in_code.copy()
                else:
                    # Need to add a word that covers all newly uncovered
                    if len(newly_uncovered_words) <= 5:
                        cand_set = set(neighbors[newly_uncovered_words[0], :].tolist())
                        for uw in newly_uncovered_words[1:]:
                            cand_set.intersection_update(neighbors[uw, :].tolist())
                        candidates = [c for c in cand_set if not current_in_code[c]]
                        
                        if candidates:
                            w3 = candidates[rng.integers(len(candidates))]
                            nbs3 = neighbors[w3, :]
                            prev_cov3 = current_coverage[nbs3].copy()
                            current_coverage[nbs3] += 1
                            
                            if np.any(current_coverage == 0):
                                current_coverage[nbs1] = prev_cov1
                                current_coverage[nbs2] = prev_cov2
                                current_coverage[nbs3] = prev_cov3
                            else:
                                current_in_code[w1] = False
                                current_in_code[w2] = False
                                current_in_code[w3] = True
                                code_list.remove(w1)
                                code_list.remove(w2)
                                code_list.append(w3)
                                current_size -= 1
                                if current_size < best_size:
                                    best_size = current_size
                                    best_in_code = current_in_code.copy()
                        else:
                            current_coverage[nbs1] = prev_cov1
                            current_coverage[nbs2] = prev_cov2
                    else:
                        current_coverage[nbs1] = prev_cov1
                        current_coverage[nbs2] = prev_cov2
                
            T *= alpha

        # Final aggressive deletion pass on best_in_code
        final_in_code = best_in_code.copy()
        final_coverage = compute_coverage(final_in_code)
        for _ in range(50):
            changed = False
            # Try multiple passes with different orderings
            for w in np.where(final_in_code)[0]:
                if np.min(final_coverage[neighbors[w, :]]) > 1:
                    final_in_code[w] = False
                    final_coverage[neighbors[w, :]] -= 1
                    changed = True
            if not changed:
                break
        
        return final_in_code, int(np.sum(final_in_code))

    # Main Loop: Multiple restarts
    best_valid_code = None
    best_size = N

    # Greedy Initialization with bucket-based max selection
    def greedy_init(rng):
        in_code = np.zeros(N, dtype=np.bool_)
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_count = N
        gains = np.full(N, neigh_size, dtype=np.int32)
        
        buckets = [[] for _ in range(neigh_size + 1)]
        for w in range(N):
            buckets[gains[w]].append(w)
        max_gain = neigh_size
        
        while uncovered_count > 0 and time.time() < deadline - 5:
            while max_gain > 0 and len(buckets[max_gain]) == 0:
                max_gain -= 1
            if max_gain == 0:
                break
            candidates = buckets[max_gain]
            if not candidates:
                continue
            w = int(rng.choice(candidates))
            
            if in_code[w]:
                buckets[max_gain].remove(w)
                continue
                
            in_code[w] = True
            buckets[max_gain].remove(w)
            nbs = neighbors[w, :]
            
            for u in nbs:
                old_cov = coverage[u]
                coverage[u] = old_cov + 1
                
                if old_cov == 0:
                    for v in neighbors[u, :]:
                        old_g = gains[v]
                        new_g = old_g - 1
                        if new_g < 0:
                            new_g = 0
                        gains[v] = new_g
                        if in_code[v]:
                            continue
                        if new_g != old_g:
                            buckets[old_g].remove(v)
                            buckets[new_g].append(v)
                    uncovered_count -= 1
            
            if uncovered_count == 0:
                break
        
        return in_code, np.sum(in_code)

    # Run multiple restarts
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
            
        # Use different seeds for diversity
        rng = np.random.default_rng(12345 + i * 7919)
        
        # Greedy Init
        in_code, size = greedy_init(rng)
        
        # Local Search
        best_code, best_sz = local_search(in_code, rng, deadline)
        
        if best_sz < best_size:
            best_size = best_sz
            best_valid_code = best_code.copy()
            
        if best_size <= 486:
            break

    # Fallback if no valid code found (should not happen with greedy)
    if best_valid_code is None:
        best_valid_code = np.ones(N, dtype=np.bool_)
        best_size = N

    # Final verification and polish
    # Ensure the code is valid by checking coverage
    final_coverage = compute_coverage(best_valid_code)
    if np.any(final_coverage == 0):
        # Add back any missing words (should not happen)
        for w in range(N):
            if final_coverage[w] == 0:
                best_valid_code[w] = True
                final_coverage[neighbors[w, :]] += 1

    # Additional polish: try to remove words in a smart order
    for _ in range(20):
        changed = False
        # Sort code words by minimum coverage of their neighbors (remove most redundant first)
        code_words = np.where(best_valid_code)[0]
        if len(code_words) == 0:
            break
        min_covs = np.array([np.min(final_coverage[neighbors[w, :]]) for w in code_words])
        order = np.argsort(min_covs)  # Remove words with highest min coverage first
        for idx in order:
            w = code_words[idx]
            if best_valid_code[w] and np.min(final_coverage[neighbors[w, :]]) > 1:
                best_valid_code[w] = False
                final_coverage[neighbors[w, :]] -= 1
                changed = True
        if not changed:
            break

    # Construct output
    final_code = []
    for w in range(N):
        if best_valid_code[w]:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
            
    return final_code
