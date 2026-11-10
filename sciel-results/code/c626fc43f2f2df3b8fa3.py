import numpy as np
import time
import sys

def construct(q, n, radius):
    deadline = time.time() + 115.0

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                neighbors[w, idx] = w + diff1 * powers[i]
                idx += 1
                diff2 = ((d + 2) % q) - d
                neighbors[w, idx] = w + diff2 * powers[i]
                idx += 1

        def greedy_construct(rng, alpha=1.0):
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                threshold = max_gain * alpha
                candidates = np.where(gains >= threshold)[0]
                
                if alpha < 1.0 and len(candidates) > 1:
                    weights = gains[candidates]
                    w = rng.choice(candidates, p=weights / weights.sum())
                else:
                    w = rng.choice(candidates)
                
                code.add(w)
                uncovered[neighbors[w, :]] = False
            return code

        def reduce_code(code_set):
            if not code_set:
                return code_set
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def local_search(code_set, max_iters=1000):
            """
            Optimized local search with Grow-Shrink and Swaps.
            """
            if not code_set:
                return code_set
            
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
                
            rng = np.random.default_rng()
            code_list = list(code_set)
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # 1. Try 1-for-1 Swap (Reduction or Neutral)
                # Pick a random codeword c
                idx = rng.integers(0, len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c, :]
                
                # Find words only covered by c
                essential = c_neigh[coverage[c_neigh] == 1]
                
                if len(essential) == 0:
                    # c is redundant, remove it
                    code_set.remove(c)
                    coverage[neighbors[c, :]] -= 1
                    code_list.pop(idx)
                    continue
                
                # Find candidates d that cover all essential
                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    # Intersection of neighborhoods
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)
                
                # Try to find a valid swap
                found_swap = False
                for d in candidates:
                    if d in code_set:
                        continue
                    # Check if removing c and adding d is valid
                    # c_neigh words with coverage 1 must be in d_neigh
                    # We already know essential are in d_neigh.
                    # But what about other words in c_neigh with coverage 1?
                    # Essential IS the set of words in c_neigh with coverage 1.
                    # So if d covers essential, and d is not c, it's a valid swap?
                    # Wait, d might not cover some word in c_neigh that has coverage > 1?
                    # No, if coverage > 1, it's fine if d doesn't cover it, as long as coverage stays >= 1.
                    # Removing c decreases coverage of c_neigh by 1.
                    # Adding d increases coverage of d_neigh by 1.
                    # Words in c_neigh \ d_neigh: coverage decreases by 1.
                    # If any had coverage 1, they become 0.
                    # These are exactly the 'essential' words.
                    # So if d covers all essential, the swap is valid.
                    
                    # Perform swap
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    code_list = list(code_set)
                    found_swap = True
                    break
                
                if found_swap:
                    continue
                
                # 2. Try Grow-Shrink (Add 1, Remove 2)
                # Add a random candidate
                new_c = None
                for _ in range(10):
                    candidate = rng.integers(0, N)
                    if candidate not in code_set:
                        new_c = candidate
                        break
                if new_c is None:
                    continue

                code_set.add(new_c)
                coverage[neighbors[new_c, :]] += 1
                code_list.append(new_c)
                
                # Now try to remove 2 codewords
                # We need to find 2 codewords c1, c2 such that removing them leaves all words covered.
                # This is equivalent to: The union of their "essential" sets (after adding new_c) must be coverable by... 
                # Actually, simpler: 
                # Remove c1. Check if valid. If not, revert.
                # This is just a sequence of 1-for-0 removals? No, we want to reduce size.
                # If we remove 2 and added 1, net size -1.
                
                # Heuristic: Find a codeword c1 that is "redundant" in the new set?
                # If min(coverage[neighbors[c1]]) > 1, it's redundant.
                # If we remove 2 redundant ones, we shrink by 2.
                # But we only added 1. So we shrink by 1 net.
                
                removed_any = False
                # Try to find 2 redundant codewords
                redundant = []
                for c in code_list:
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        redundant.append(c)
                
                if len(redundant) >= 2:
                    # Remove two
                    c1 = redundant[0]
                    c2 = redundant[1]
                    code_set.remove(c1)
                    coverage[neighbors[c1, :]] -= 1
                    code_set.remove(c2)
                    coverage[neighbors[c2, :]] -= 1
                    code_list = list(code_set)
                    removed_any = True
                elif len(redundant) == 1:
                    # Remove one (Net size 0)
                    c1 = redundant[0]
                    code_set.remove(c1)
                    coverage[neighbors[c1, :]] -= 1
                    code_list = list(code_set)
                    removed_any = True
                else:
                    # No redundant codewords. Revert the addition.
                    code_set.remove(new_c)
                    coverage[neighbors[new_c, :]] -= 1
                    code_list = list(code_set)
                
                if not removed_any:
                    continue
            
            return code_set

        best_code = None
        best_size = float('inf')

        configs = [
            (42, 1.0), (123, 1.0), (456, 0.95), (789, 0.9),
            (101, 1.0), (202, 0.85), (303, 0.95), (404, 1.0),
            (505, 0.9), (606, 0.8), (707, 1.0), (808, 0.95)
        ]

        for seed, alpha in configs:
            if time.time() > deadline - 10:
                break

            rng = np.random.default_rng(seed)
            code_set = greedy_construct(rng, alpha)
            code_set = reduce_code(code_set)
            code_set = local_search(code_set, max_iters=500)
            code_set = reduce_code(code_set)
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

        if best_code is None:
            rng = np.random.default_rng(0)
            code_set = greedy_construct(rng, 1.0)
            best_code = reduce_code(code_set)

        # Final polish
        best_code = local_search(best_code, max_iters=200)
        best_code = reduce_code(best_code)

        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)

        return final_code
    else:
        # Fallback
        N = q ** n
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                neighbors[w, idx] = w + diff1 * powers[i]
                idx += 1
                diff2 = ((d + 2) % q) - d
                neighbors[w, idx] = w + diff2 * powers[i]
                idx += 1

        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False

        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
