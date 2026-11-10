import numpy as np
import time
import itertools
import random
from collections import deque

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        def greedy_init(rng, use_randomized=False):
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            while np.any(uncovered) and time.time() < deadline:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                if use_randomized and len(candidates) > 1:
                    top_k = min(5, len(candidates))
                    chosen = rng.choice(candidates[:top_k])
                    w = chosen
                else:
                    w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w, :]] = False
            return list(code)
        
        def reduce_code(code, neighbors, deadline):
            code_set = set(code)
            if not code_set:
                return code_set
            
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            
            return code_set
        
        def try_swap(code_set, coverage, neighbors, deadline, rng):
            if not code_set:
                return code_set, coverage, False
            
            c = rng.choice(list(code_set))
            c_neighbors = neighbors[c, :]
            
            essential_words = c_neighbors[coverage[c_neighbors] == 1]
            
            if len(essential_words) == 0:
                code_set.remove(c)
                coverage[neighbors[c, :]] -= 1
                return code_set, coverage, True
            
            w = rng.choice(essential_words)
            candidates = neighbors[w, :]
            
            best_new_set = None
            best_size = len(code_set)
            
            cand_list = candidates.tolist()
            random.shuffle(cand_list)
            
            for d in cand_list[:15]:
                if d in code_set:
                    continue
                
                new_set = set(code_set)
                new_set.remove(c)
                new_set.add(d)
                
                for ew in essential_words:
                    if not np.any(neighbors[d, :] == ew):
                        break
                else:
                    new_coverage = coverage.copy()
                    new_coverage[neighbors[c, :]] -= 1
                    new_coverage[neighbors[d, :]] += 1
                    
                    new_set = reduce_code(list(new_set), neighbors, deadline)
                    
                    if len(new_set) < best_size:
                        best_new_set = new_set
                        best_size = len(new_set)
            
            if best_new_set is not None:
                new_cov = np.zeros(N, dtype=np.int32)
                for cc in best_new_set:
                    new_cov[neighbors[cc, :]] += 1
                return best_new_set, new_cov, True
            
            return code_set, coverage, False
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts with different strategies
        for i in range(25):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.default_rng(42 + i * 7)
            
            use_randomized = (i % 3 == 0)
            
            code_list = greedy_init(rng, use_randomized)
            code_set = set(code_list)
            
            code_set = reduce_code(list(code_set), neighbors, deadline)
            
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for _ in range(80):
                    if time.time() > deadline - 5:
                        break
                    new_set, new_coverage, was_improved = try_swap(
                        code_set, coverage, neighbors, deadline, rng
                    )
                    if was_improved:
                        code_set = new_set
                        coverage = new_coverage
                        improved = True
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set
                
            if best_size <= 486:
                break

        if best_code is None:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
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
        N = q ** n
        deadline = time.time() + 110.0
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
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
