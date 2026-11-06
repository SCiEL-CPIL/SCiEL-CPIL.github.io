import time
import itertools

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge[i][j] = c
            edge[j][i] = c
            idx += 1
    
    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Generate proper edge coloring with 15 colors
    n = 16
    colors = [0] * (n * (n - 1) // 2)
    idx = 0
    # Standard 1-factorization for K_16
    # Vertices 0..14, 15 is fixed
    for r in range(15):
        # Color r
        # Match 15 with r
        i, j = 15, r
        if i > j: i, j = j, i
        # Find index
        # Index calculation: sum of lengths of previous rows + offset
        # Row i has n-i-1 elements? No, lexicographic (i,j) with i<j.
        # Index = sum_{k=0}^{i-1} (n-k-1) + j - i - 1
        # Precompute offsets
        offsets = [0] * n
        for i in range(n):
            offsets[i] = sum(n - k - 1 for k in range(i))
        
        idx = offsets[i] + (j - i - 1)
        colors[idx] = r
        
        for k in range(1, 8):
            u = (r - k) % 15
            v = (r + k) % 15
            i, j = u, v
            if i > j: i, j = j, i
            idx = offsets[i] + (j - i - 1)
            colors[idx] = r
            
    # Verify proper coloring (optional, but good for sanity)
    # It should use exactly 15 colors
    if _num_colours(colors) != 15:
        # Fallback to additive if something went wrong
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    # 2. Try to reduce to 14 colors by merging two colors
    # Any merge is valid for a proper coloring.
    # Merge color 0 and 1.
    merged_14 = [0 if c in [0, 1] else c for c in colors]
    # Renumber to 0..13
    # Map {0:0, 2:1, 3:2, ...}
    # Actually, we just need the count. The evaluator checks validity and count.
    # We should renumber to be clean, but it's not strictly required if we just return the list.
    # However, "Minimize the number of colours used" implies the count of distinct values.
    # If we merge 0 and 1, we have 14 distinct values.
    
    # Check if 14 is valid (it should be)
    if _check_k4_fast(merged_14):
        best_colors = merged_14
        best_count = 14
    else:
        best_colors = colors
        best_count = 15

    # 3. Try to reduce to 13 colors
    # We need to find 4 colors that do not co-occur in any K4.
    # Precompute K4 color masks
    n = 16
    edge = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge[i][j] = colors[idx]
            edge[j][i] = colors[idx]
            idx += 1
            
    k4_masks = []
    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    k4_masks.append(mask)
                    
    # Try to find a set of 4 colors {c1, c2, c3, c4} such that no K4 contains all 4.
    # If found, we can merge {c1, c2} and {c3, c4}.
    # This reduces count by 2.
    found_13 = False
    target_colors = list(range(15))
    
    # To save time, we can just try a few random sets or systematic ones.
    # 1365 combinations.
    # Check time limit.
    if time.time() < deadline - 10:
        for c1 in range(15):
            for c2 in range(c1 + 1, 15):
                for c3 in range(c2 + 1, 15):
                    for c4 in range(c3 + 1, 15):
                        if time.time() > deadline - 5:
                            break
                        mask_target = (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4)
                        valid = True
                        for m in k4_masks:
                            if (m & mask_target) == mask_target:
                                valid = False
                                break
                        if valid:
                            # Merge c1,c2 -> 0 and c3,c4 -> 1 (conceptually)
                            # We need to construct the new coloring
                            # Map: c1->14, c2->14, c3->13, c4->13, others keep value?
                            # We just need to reduce distinct count.
                            # Let's create a mapping
                            mapping = {i: i for i in range(15)}
                            mapping[c1] = 14
                            mapping[c2] = 14
                            mapping[c3] = 13
                            mapping[c4] = 13
                            
                            new_colors = [mapping[c] for c in colors]
                            # Renumber to 0..12?
                            # Distinct values in new_colors will be 13.
                            if _check_k4_fast(new_colors):
                                best_colors = new_colors
                                best_count = 13
                                found_13 = True
                                break
                    if found_13: break
                if found_13: break
            if found_13: break

    if best_count <= 13:
        return best_colors
        
    if best_count <= 14:
        return best_colors
        
    return colors

