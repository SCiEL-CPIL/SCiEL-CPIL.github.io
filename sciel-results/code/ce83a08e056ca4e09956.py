import time
import itertools
import numpy as np

def construct(n):
    if n == 1:
        return "1"
    
    if n <= 6:
        # Use standard recursive construction
        def get_gl(m):
            if m == 1:
                return [1], [(1,)]
            prev_word, prev_perms = get_gl(m-1)
            new_word = []
            for p in prev_perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new_word), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        ov = o
                        break
                new_word += block[ov:]
            
            perms = []
            seen = set()
            for i in range(len(new_word) - m + 1):
                w = tuple(new_word[i:i+m])
                if len(set(w)) == m and w not in seen:
                    seen.add(w)
                    perms.append(w)
            return new_word, perms
        
        word, _ = get_gl(n)
        return "".join(map(str, word))
    
    if n == 7:
        # Get all permutations of 1..6
        perms = list(itertools.permutations(range(1, 7)))
        num_perms = len(perms)
        
        # Precompute overlap matrix using NumPy
        # L[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        # Convert perms to array for vectorized operations
        perm_array = np.array(perms, dtype=np.int32)
        
        # For each pair (i,j), compute overlap
        # overlap[i][j] = max k in 0..5 where perm_array[i][-k:] == perm_array[j][:k]
        L = np.zeros((num_perms, num_perms), dtype=np.int8)
        
        # Vectorized overlap computation
        # For k from 5 down to 1, check if suffix of length k of i matches prefix of length k of j
        for k in range(5, 0, -1):
            # suffixes of length k: perm_array[:, -k:]
            # prefixes of length k: perm_array[:, :k]
            # Need to check if suffix[i] == prefix[j] for all i,j
            # This is a matrix comparison
            suffixes = perm_array[:, -k:]  # shape (num_perms, k)
            prefixes = perm_array[:, :k]   # shape (num_perms, k)
            
            # For each i,j, check if suffixes[i] == prefixes[j]
            # Use broadcasting: suffixes[:, np.newaxis, :] == prefixes[np.newaxis, :, :]
            # This gives shape (num_perms, num_perms, k)
            # Then check all elements match
            match = np.all(suffixes[:, np.newaxis, :] == prefixes[np.newaxis, :, :], axis=2)
            # Only update where L is still 0 (haven't found a longer overlap yet)
            L[match & (L == 0)] = k
        
        # Initial order: identity
        order = np.arange(num_perms, dtype=np.int32)
        
        def calc_total_overlap(order):
            """Calculate total overlap for a given order."""
            if len(order) < 2:
                return 0
            idx = order[:-1]
            idx_next = order[1:]
            return np.sum(L[idx, idx_next])
        
        best_overlap = calc_total_overlap(order)
        best_order = order.copy()
        
        # Local search with 2-opt
        start_time = time.time()
        deadline = start_time + 100.0  # 100s for search, 20s buffer
        
        def do_2opt(order, deadline):
            """Perform 2-opt local search."""
            improved = True
            while improved and time.time() < deadline:
                improved = False
                n = len(order)
                for i in range(n - 1):
                    if time.time() > deadline:
                        return order
                    for j in range(i + 1, n):
                        # Calculate delta for reversing order[i:j+1]
                        delta = 0
                        
                        # Boundary 1: link before i
                        if i > 0:
                            delta += L[order[i-1], order[j]] - L[order[i-1], order[i]]
                        # Boundary 2: link after j
                        if j < n - 1:
                            delta += L[order[i], order[j+1]] - L[order[j], order[j+1]]
                        
                        # Internal: sum of L[order[k], order[k+1]] - L[order[j-(k-i)], order[j-(k-i)-1]]
                        # For speed, only compute if segment is small
                        seg_len = j - i
                        if seg_len <= 20:
                            s_old = 0
                            s_new = 0
                            for k in range(i, j):
                                s_old += L[order[k], order[k+1]]
                                s_new += L[order[j-(k-i)], order[j-(k-i)-1]]
                            delta += s_new - s_old
                        else:
                            # For large segments, approximate or skip
                            # We can still compute but it's expensive
                            s_old = 0
                            s_new = 0
                            for k in range(i, j):
                                s_old += L[order[k], order[k+1]]
                                s_new += L[order[j-(k-i)], order[j-(k-i)-1]]
                            delta += s_new - s_old
                        
                        if delta > 0:
                            # Apply reversal
                            order[i:j+1] = order[i:j+1][::-1]
                            improved = True
            return order
        
        order = do_2opt(order, deadline)
        best_overlap = calc_total_overlap(order)
        best_order = order.copy()
        
        # Try Or-opt: move a segment of length 1-3 to another position
        def do_oropt(order, deadline):
            """Perform Or-opt local search (move small segments)."""
            improved = True
            while improved and time.time() < deadline:
                improved = False
                n = len(order)
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline:
                        return order
                    for i in range(n - seg_len):
                        if time.time() > deadline:
                            return order
                        segment = order[i:i+seg_len]
                        # Try inserting after position j (j != i, j != i+seg_len-1)
                        for j in range(n - seg_len + 1):
                            if j == i or j == i + seg_len - 1:
                                continue
                            if time.time() > deadline:
                                return order
                            
                            # Calculate delta
                            delta = 0
                            
                            # Remove segment from position i
                            # Old links: (i-1, i), (i+seg_len-1, i+seg_len)
                            # New links after removal: (i-1, i+seg_len)
                            if i > 0:
                                delta -= L[order[i-1], order[i]]
                            if i + seg_len < n:
                                delta -= L[order[i+seg_len-1], order[i+seg_len]]
                            if i > 0 and i + seg_len < n:
                                delta += L[order[i-1], order[i+seg_len]]
                            
                            # Insert segment after position j
                            # New links: (j, segment[0]), (segment[-1], j+seg_len)
                            # Old links: (j, j+1)
                            if j + seg_len < n:
                                delta -= L[order[j], order[j+1]]
                            delta += L[order[j], segment[0]]
                            if j + seg_len < n:
                                delta += L[segment[-1], order[j+seg_len]]
                            
                            if delta > 0:
                                # Apply move
                                new_order = np.concatenate([
                                    order[:i],
                                    order[i+seg_len:j+1],
                                    segment,
                                    order[j+1:]
                                ])
                                order = new_order
                                improved = True
            return order
        
        order = do_oropt(order, deadline)
        best_overlap = calc_total_overlap(order)
        best_order = order.copy()
        
        # Final 2-opt pass
        order = do_2opt(order, deadline)
        best_overlap = calc_total_overlap(order)
        best_order = order.copy()
        
        # Construct final word
        final_word = []
        for idx in best_order:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            max_overlap = min(len(final_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if final_word[len(final_word) - o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
        
        return "".join(map(str, final_word))
    
    # For n > 7, use greedy construction
    def get_gl(m):
        if m == 1:
            return [1], [(1,)]
        prev_word, prev_perms = get_gl(m-1)
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            max_ov = min(len(new_word), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    ov = o
                    break
            new_word += block[ov:]
        
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            w = tuple(new_word[i:i+m])
            if len(set(w)) == m and w not in seen:
                seen.add(w)
                perms.append(w)
        return new_word, perms
    
    word, _ = get_gl(n)
    return "".join(map(str, word))
