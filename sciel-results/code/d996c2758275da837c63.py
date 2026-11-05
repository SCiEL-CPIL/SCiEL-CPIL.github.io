import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate all words and their indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # 2. Precompute neighbors for radius 1
    # Each word has 1 (itself) + n*(q-1) neighbors
    num_neighbors = 1 + n * (q - 1)

    neighbors = []
    neighbor_array = np.zeros((N, num_neighbors), dtype=np.int32)
    
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_array[i, :] = neigh

    # 3. Precompute reverse neighbors: for each point, which words cover it
    # This is the "hypergraph" structure. We store lists of codewords that cover each point.
    covered_by = [[] for _ in range(N)]
    for i in range(N):
        for nb in neighbors[i]:
            covered_by[nb].append(i)
    
    # Convert to sets for faster membership testing? 
    # Actually, for the peeling heuristic, we need to check if a point is covered by *other* nodes.
    # We will maintain counts dynamically.

    # 4. Precompute neighbor sets for fast membership testing (used in local search)
    neighbor_sets = [set(neigh) for neigh in neighbors]

    # 5. Helper functions
    
    def get_counts(code_list):
        """Calculate coverage counts for all points given a code."""
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        for c in code_list:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    def is_valid(code_list):
        """Check if code covers all points."""
        if not code_list:
            return N == 0
        counts = get_counts(code_list)
        return np.min(counts) >= 1

    def prune_code(code_list):
        """
        Hypergraph 2-core peeling heuristic.
        Iteratively remove codewords whose removal does not uncover any point.
        This finds a minimal subcode (inclusion-wise) by exploiting redundancy.
        """
        if not code_list:
            return []
        
        # Initialize counts
        counts = np.zeros(N, dtype=np.int32)
        for c in code_list:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        code_set = set(code_list)
        # We use a queue of "active" candidates to check.
        # Initially, all codewords are candidates.
        # We check if a codeword is "removable": all its neighbors have count > 1.
        
        # To be efficient, we can just iterate until no more changes.
        # Since N is small (2187) and code size is ~200, this is fast.
        changed = True
        while changed:
            changed = False
            # Iterate over a copy because we might modify the list
            for c in list(code_list):
                # Check if all points covered by c are covered by at least one other codeword
                # i.e., counts[nb] > 1 for all nb in neighbors[c]
                # Vectorized check:
                if np.all(counts[neighbor_array[c]] > 1):
                    # Remove c
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        
        return code_list

    def greedy_set_cover(time_limit):
        """Greedy set cover with overlap-aware scoring."""
        code = []
        covered = np.zeros(N, dtype=bool)
        remaining = N
        counts = np.zeros(N, dtype=np.int32)
        
        while remaining > 0 and time.time() < deadline - time_limit:
            # Score: prefer points that cover many uncovered, penalize overlap
            gains = (~covered)[neighbor_array].sum(axis=1)
            overlap = (counts >= 1)[neighbor_array].sum(axis=1)
            # Weighted score: maximize new coverage, minimize overlap
            scores = gains.astype(np.float64) - 0.1 * overlap
            max_score = np.max(scores)
            if max_score <= 0:
                # Fall back to pure gain
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                candidates = np.where(scores == max_score)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            best_idx = int(best_idx)
            code.append(best_idx)
            for nb in neighbors[best_idx]:
                if not covered[nb]:
                    covered[nb] = True
                    remaining -= 1
                counts[nb] += 1
        
        return code

    def local_search(code_list, max_time=5.0):
        """Local search with 1-exchange, 2-for-1, and 2-for-2 moves."""
        if not code_list:
            return []
        
        code_set = set(code_list)
        counts = get_counts(code_list)
        improved = True
        ls_start = time.time()
        max_time = min(max_time, deadline - 5 - time.time())
        
        if max_time <= 0:
            return prune_code(code_list)
        
        while improved and time.time() - ls_start < max_time and time.time() < deadline - 5:
            improved = False
            # Use the new peeling prune
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            # Try 1-for-1 swaps
            code_order = list(code_list)
            random.shuffle(code_order)
            
            for c in code_order:
                if time.time() >= deadline - 5:
                    break
                if c not in code_set:
                    continue
                
                # Find points only covered by c
                crit = [nb for nb in neighbors[c] if counts[nb] == 1]
                
                if not crit:
                    # c is redundant
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    improved = True
                    break
                
                # Try to find a replacement that covers all critical points
                p = crit[0]
                candidates = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    if not all(k in neighbor_sets[w] for k in crit):
                        continue
                    # Score: prefer replacements that create more redundancy (easier to remove later)
                    score = sum(1 for nb in neighbors[w] if counts[nb] >= 2)
                    # Bonus for covering points with high count
                    score += 0.5 * sum(max(0, counts[nb] - 2) for nb in neighbors[w])
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w != -1:
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    code_list.append(best_w)
                    code_set.add(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-1 exchanges
            if len(code_list) >= 2 and time.time() < deadline - 5:
                code_copy = list(code_list)
                random.shuffle(code_copy)
                
                found = False
                for i in range(len(code_copy)):
                    if found:
                        break
                    c1 = code_copy[i]
                    if c1 not in code_set:
                        continue
                    crit1 = [nb for nb in neighbors[c1] if counts[nb] == 1]
                    
                    for j in range(i + 1, len(code_copy)):
                        c2 = code_copy[j]
                        if c2 not in code_set:
                            continue
                        crit2 = [nb for nb in neighbors[c2] if counts[nb] == 1]
                        
                        all_crit = list(set(crit1 + crit2))
                        
                        if not all_crit:
                            # Both redundant
                            code_list.remove(c1)
                            code_set.remove(c1)
                            for nb in neighbors[c1]:
                                counts[nb] -= 1
                            code_list.remove(c2)
                            code_set.remove(c2)
                            for nb in neighbors[c2]:
                                counts[nb] -= 1
                            improved = True
                            found = True
                            break
                        
                        p = all_crit[0]
                        candidates = [x for x in neighbors[p] if x not in code_set and x != c1 and x != c2]
                        
                        best_w = -1
                        best_score = -1
                        
                        for w in candidates:
                            if not all(k in neighbor_sets[w] for k in all_crit):
                                continue
                            score = sum(1 for nb in neighbors[w] if counts[nb] >= 2)
                            score += 0.5 * sum(max(0, counts[nb] - 2) for nb in neighbors[w])
                            if score > best_score:
                                best_score = score
                                best_w = w
                        
                        if best_w != -1:
                            code_list.remove(c1)
                            code_set.remove(c1)
                            for nb in neighbors[c1]:
                                counts[nb] -= 1
                            code_list.remove(c2)
                            code_set.remove(c2)
                            for nb in neighbors[c2]:
                                counts[nb] -= 1
                            code_list.append(best_w)
                            code_set.add(best_w)
                            for nb in neighbors[best_w]:
                                counts[nb] += 1
                            improved = True
                            found = True
                            break
                        
                        if time.time() >= deadline - 5:
                            break
                
                if found:
                    continue
            
            # Try 2-for-2 exchanges: replace c1,c2 with w1,w2 (net zero change in size, but may enable further deletions)
            if len(code_list) >= 2 and time.time() < deadline - 5:
                code_copy = list(code_list)
                random.shuffle(code_copy)
                
                found_22 = False
                # Limit pairs to avoid O(n^2) blowup
                max_pairs = min(len(code_copy), 30)
                
                for i in range(max_pairs):
                    if found_22:
                        break
                    c1 = code_copy[i]
                    if c1 not in code_set:
                        continue
                    crit1 = [nb for nb in neighbors[c1] if counts[nb] == 1]
                    if not crit1:
                        continue
                    
                    for j in range(i + 1, min(i + 10, len(code_copy))):
                        c2 = code_copy[j]
                        if c2 not in code_set:
                            continue
                        crit2 = [nb for nb in neighbors[c2] if counts[nb] == 1]
                        
                        all_crit = list(set(crit1 + crit2))
                        
                        # Find two replacements w1, w2 that together cover all_crit
                        p = all_crit[0]
                        cand1 = [x for x in neighbors[p] if x not in code_set and x != c1 and x != c2]
                        
                        best_pair = None
                        best_pair_score = -1
                        
                        for w1 in cand1[:20]:  # Limit candidates
                            if not all(k in neighbor_sets[w1] for k in crit1):
                                continue
                            remaining_crit = [k for k in all_crit if k not in neighbor_sets[w1]]
                            if not remaining_crit:
                                # w1 alone covers all - just do 1-for-2
                                score = sum(1 for nb in neighbors[w1] if counts[nb] >= 2)
                                if score > best_pair_score:
                                    best_pair_score = score
                                    best_pair = (w1, None)
                                continue
                            p2 = remaining_crit[0]
                            cand2 = [x for x in neighbors[p2] if x not in code_set and x != c1 and x != c2 and x != w1]
                            for w2 in cand2[:10]:
                                if not all(k in neighbor_sets[w2] for k in remaining_crit):
                                    continue
                                score = sum(1 for nb in neighbors[w1] if counts[nb] >= 2) + sum(1 for nb in neighbors[w2] if counts[nb] >= 2)
                                if score > best_pair_score:
                                    best_pair_score = score
                                    best_pair = (w1, w2)
                        
                        if best_pair is not None:
                            w1, w2 = best_pair
                            code_list.remove(c1)
                            code_set.remove(c1)
                            for nb in neighbors[c1]:
                                counts[nb] -= 1
                            code_list.remove(c2)
                            code_set.remove(c2)
                            for nb in neighbors[c2]:
                                counts[nb] -= 1
                            code_list.append(w1)
                            code_set.add(w1)
                            for nb in neighbors[w1]:
                                counts[nb] += 1
                            if w2 is not None:
                                code_list.append(w2)
                                code_set.add(w2)
                                for nb in neighbors[w2]:
                                    counts[nb] += 1
                            improved = True
                            found_22 = True
                            break
                        
                        if time.time() >= deadline - 5:
                            break
                
                if found_22:
                    continue
        
        return code_list

    def perturb(code_list, strength=0.1):
        """Perturb by removing low-contribution codewords and repairing."""
        if not code_list:
            return []
        
        code_set = set(code_list)
        code_list = list(code_list)
        counts = get_counts(code_list)
        
        # Score each codeword by its "essentiality": how many points it uniquely covers
        essentiality = []
        for c in code_list:
            uniq = sum(1 for nb in neighbors[c] if counts[nb] == 1)
            essentiality.append((uniq, c))
        
        # Remove codewords with lowest essentiality first
        essentiality.sort(key=lambda x: x[0])
        num_remove = max(1, int(len(code_list) * strength))
        
        for _, c in essentiality[:num_remove]:
            if c in code_set:
                code_list.remove(c)
                code_set.discard(c)
        
        # Repair
        counts = get_counts(code_list)
        uncovered = counts < 1
        uncovered_count = int(np.sum(uncovered))
        
        while uncovered_count > 0 and time.time() < deadline - 5:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            code_list.append(best_idx)
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                if counts[nb] < 1:
                    uncovered_count -= 1
                counts[nb] += 1
            uncovered = counts < 1
        
        return code_list

    def run_ils(seed):
        """Iterated Local Search with adaptive perturbation."""
        random.seed(seed)
        
        # Initial greedy solution
        code = greedy_set_cover(time_limit=15.0)
        code = prune_code(code)
        
        best_local = list(code)
        best_size = len(best_local)
        no_improve_count = 0
        
        # ILS loop
        ils_start = time.time()
        max_ils_time = min(10.0, deadline - 12 - time.time())
        
        while time.time() - ils_start < max_ils_time and time.time() < deadline - 8:
            # Adaptive perturbation strength
            if no_improve_count < 3:
                strength = random.uniform(0.03, 0.08)
            elif no_improve_count < 6:
                strength = random.uniform(0.08, 0.15)
            else:
                strength = random.uniform(0.15, 0.30)
            
            perturbed = perturb(best_local, strength)
            
            # Local search
            ls_time = min(4.0, max(1.0, (deadline - 8 - time.time()) * 0.4))
            optimized = local_search(perturbed, max_time=ls_time)
            optimized = prune_code(optimized)
            
            if len(optimized) < best_size:
                best_local = optimized
                best_size = len(optimized)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Occasional stronger restart
            if no_improve_count >= 5 and random.random() < 0.5:
                # Fresh greedy with different random seed
                random.seed(seed + no_improve_count * 1000)
                fresh = greedy_set_cover(time_limit=5.0)
                fresh = prune_code(fresh)
                if len(fresh) < best_size:
                    best_local = fresh
                    best_size = len(fresh)
                    no_improve_count = 0
        
        # Final validity check
        counts = get_counts(best_local)
        if np.min(counts) < 1:
            return None
        
        return best_local

    # Main loop: multiple restarts with time-aware scheduling
    best_code = None
    best_size = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 7919 + 13
        current_code = run_ils(seed)
        num_restarts += 1
        
        if current_code is not None:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
        
        # Adaptive stopping: allow more restarts early, fewer when time is low
        remaining_time = deadline - time.time()
        if num_restarts >= 3 and remaining_time < 30:
            break
        if num_restarts >= 8:
            break
    
    if best_code is None:
        # Fallback: greedy only
        code = greedy_set_cover(time_limit=10.0)
        best_code = prune_code(code)
        if not best_code:
            return []
    
    best_code = prune_code(best_code)
    
    # Final polish: try aggressive local search on the best solution
    if best_code and time.time() < deadline - 5:
        polish_time = min(3.0, deadline - 5 - time.time())
        polished = local_search(best_code, max_time=polish_time)
        polished = prune_code(polished)
        if len(polished) <= len(best_code):
            best_code = polished
    
    # Final validity check
    if best_code:
        counts = get_counts(best_code)
        if np.min(counts) < 1:
            # Emergency repair
            uncovered = counts < 1
            while np.any(uncovered) and time.time() < deadline - 1:
                gains = uncovered[neighbor_array].sum(axis=1)
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = int(candidates[0])
                best_code.append(best_idx)
                for nb in neighbors[best_idx]:
                    counts[nb] += 1
                uncovered = counts < 1
            best_code = prune_code(best_code)
    
    return [list(words[i]) for i in best_code]
