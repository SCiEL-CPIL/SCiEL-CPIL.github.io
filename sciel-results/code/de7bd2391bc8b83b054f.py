import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        
        # Precompute distances and neighbors
        # dist[i][j] is Hamming distance
        # We need neighbors[i]: indices j such that dist(i,j) <= radius
        
        # Efficient distance calculation
        # all_words is (729, 6)
        # diff = all_words[:, None, :] - all_words[None, :, :]
        # dist = (diff != 0).sum(axis=2)
        
        # To save memory and time, we can compute neighbors directly.
        # But 729 is small. Let's compute the full distance matrix once.
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # neighbors[i] is a list of indices
        neighbors = [np.where(dist[i] <= radius)[0] for i in range(total_words)]
        
        # Precompute reverse neighbors (who covers me?)
        # rev_neighbors[i] is list of codewords c such that i in neighbors[c]
        # Since distance is symmetric, rev_neighbors[i] == neighbors[i]
        
        def greedy_cover(uncovered_indices, initial_codes, rng):
            uncovered = set(uncovered_indices)
            codes = list(initial_codes)
            
            while uncovered:
                best_score = -1
                best_candidates = []
                
                # Vectorized coverage count?
                # For each candidate c, count how many words in uncovered are in neighbors[c]
                # This is slow in Python loops.
                
                # Optimization: 
                # If we maintain an array `uncovered_mask` (bool), we can do:
                # counts = (neighbors[c] in uncovered_mask).sum()
                
                # But iterating all 729 candidates every step is 729 * |uncovered| operations.
                # 729 * 729 * 73 ~ 38M ops. Might be too slow in Python.
                
                # Better: Randomized Greedy
                # Pick a random word w from uncovered.
                # Pick a random codeword c that covers w.
                # This is "Random Greedy" which is often faster and good enough.
                
                # Let's use a hybrid: 
                # 1. Find the word w in uncovered with the fewest candidates (most constrained).
                # 2. Pick a random candidate c covering w.
                
                # To find "fewest candidates", we need degrees.
                # degree[w] = len(neighbors[w])
                # But we only care about candidates *not yet in codes*? No, we can re-pick.
                # Actually, standard greedy picks max coverage.
                # Let's stick to max coverage but optimize the loop.
                
                # Use NumPy for the max coverage step?
                # Create a boolean matrix `cover_matrix` where cover_matrix[c][w] = 1 if w in neighbors[c].
                # This is 729x729.
                
                # Let's precompute cover_matrix as a numpy array of bools.
                
                best_idx = -1
                best_count = -1
                # Iterate candidates
                # To speed up, we can sample candidates if the set is large?
                # No, 729 is small.
                
                # We can vectorize the count if we have an array of uncovered indices.
                # But uncovered changes.
                
                # Let's use the "Most Constrained Variable" heuristic (MRV)
                # Find w in uncovered with min len(neighbors[w] \ codes)?
                # No, just min len(neighbors[w]).
                
                # Precompute static degrees
                degrees = [len(nb) for nb in neighbors]
                
                # Find w in uncovered with min degree
                # This is O(|uncovered|)
                min_deg = float('inf')
                target_w = -1
                for w in uncovered:
                    if degrees[w] < min_deg:
                        min_deg = degrees[w]
                        target_w = w
                        
                if target_w == -1:
                    break
                    
                # Pick a random codeword from neighbors[target_w]
                # Prefer one that covers many other uncovered words?
                # Let's pick randomly to add diversity, or pick best coverage.
                # Let's pick best coverage among neighbors of target_w.
                
                cand_ids = neighbors[target_w]
                best_c = -1
                best_cov = -1
                for c in cand_ids:
                    # Count coverage of c in uncovered
                    # neighbors[c] is a numpy array.
                    # We can do: len(np.intersect1d(neighbors[c], list(uncovered)))
                    # This is slow.
                    
                    # Alternative: 
                    # Just pick randomly from cand_ids.
                    # "Random Greedy" is a valid heuristic.
                    pass
                
                # Let's use a simpler, faster greedy:
                # Pick random word w from uncovered.
                # Pick random c from neighbors[w].
                # Add c.
                
                # This is very fast.
                w = rng.choice(list(uncovered))
                c = rng.choice(neighbors[w].tolist())
                
                codes.append(c)
                # Remove all words covered by c from uncovered
                for nw in neighbors[c]:
                    if nw in uncovered:
                        uncovered.remove(nw)
                        
            return codes

        def local_search(codes):
            codes = set(codes)
            # counts[i] = number of codewords covering i
            counts = np.zeros(total_words, dtype=np.int32)
            for c in codes:
                counts[neighbors[c]] += 1
                
            improved = True
            while improved:
                improved = False
                
                # 1. Prune
                for c in list(codes):
                    # If all neighbors of c are covered by >1 codeword, remove c
                    if np.min(counts[neighbors[c]]) > 1:
                        codes.remove(c)
                        counts[neighbors[c]] -= 1
                        improved = True
                        break # Restart loop to recalculate? Or continue?
                        # Continue is fine, counts are updated.
                
                if improved:
                    continue # Restart pruning/swapping after a removal
                
                # 2. Swap
                # Try to remove each c and replace with d
                c_list = list(codes)
                rng.shuffle(c_list)
                
                for c in c_list:
                    # Find private words
                    private_mask = (counts == 1)
                    # Private words are those in neighbors[c] where counts == 1
                    private_words = [w for w in neighbors[c] if private_mask[w]]
                    
                    if not private_words:
                        # Safe to remove
                        codes.remove(c)
                        counts[neighbors[c]] -= 1
                        improved = True
                        break # Restart loop
                    else:
                        # Need to find d that covers all private_words
                        # d must be in intersection of neighbors[p] for all p in private_words
                        # And d not in codes (or d in codes but we are replacing c, so if d==c it's no change, if d in codes it's a move)
                        
                        # Start with neighbors of first private word
                        common = set(neighbors[private_words[0]].tolist())
                        for p in private_words[1:]:
                            common &= set(neighbors[p].tolist())
                            
                        # Filter out existing codes? 
                        # If d is in codes, we are moving c to d.
                        # If d is not in codes, we are adding d.
                        # We want to minimize size, so we prefer d in codes (size unchanged) or d not in codes (size +1, but we removed c so size unchanged).
                        # So any d in common is valid.
                        
                        # Pick a random d from common
                        if common:
                            d = rng.choice(list(common))
                            # Remove c, add d
                            codes.remove(c)
                            counts[neighbors[c]] -= 1
                            codes.add(d)
                            counts[neighbors[d]] += 1
                            improved = True
                            break # Restart loop
                            
            return codes

        best_code = None
        best_size = float('inf')
        start_time = time.time()
        time_limit = 110
        
        # Initial solution
        rng = random.Random(42)
        initial = greedy_cover(set(range(total_words)), [], rng)
        ls_code = local_search(initial)
        best_code = ls_code
        best_size = len(ls_code)
        
        # ILS
        while time.time() - start_time < time_limit:
            # Perturb
            perturbed = list(best_code)
            k_remove = rng.randint(1, 3)
            if len(perturbed) > k_remove:
                to_remove = rng.sample(perturbed, k_remove)
                for r in to_remove:
                    perturbed.remove(r)
            
            # Repair
            # Find uncovered
            covered = set()
            for c in perturbed:
                covered.update(neighbors[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_cover(uncovered, perturbed, rng)
            
            # Local Search
            ls_code = local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
                
        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
