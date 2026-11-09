import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask adjacency matrix
    M = [0] * n
    for u, v in edges:
        M[u] |= (1 << v)
        M[v] |= (1 << u)
        
    all_mask = (1 << n) - 1
    
    def get_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        return bits

    result = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s out of 60s
    
    while True:
        if time.time() > deadline:
            break
            
        # Find best biclique
        best_L_mask = 0
        best_R_mask = 0
        best_size = 0
        
        # Iterate over all potential seed edges
        for u in range(n):
            if M[u] == 0:
                continue
            # Iterate over neighbors v of u
            v_mask = M[u]
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                
                # Try to expand biclique starting with (u, v)
                # u in L, v in R
                L_mask = 1 << u
                R_mask = 1 << v
                
                # Expand
                while True:
                    # New L: vertices connected to all in R
                    new_L = all_mask
                    r_iter = R_mask
                    while r_iter:
                        lsb_r = r_iter & -r_iter
                        r = lsb_r.bit_length() - 1
                        new_L &= M[r]
                        r_iter ^= lsb_r
                        
                    # New R: vertices connected to all in L (current L_mask)
                    new_R = all_mask
                    l_iter = L_mask
                    while l_iter:
                        lsb_l = l_iter & -l_iter
                        l = lsb_l.bit_length() - 1
                        new_R &= M[l]
                        l_iter ^= lsb_l
                        
                    # Check for stability
                    if new_L == L_mask and new_R == R_mask:
                        break
                    
                    L_mask = new_L
                    R_mask = new_R
                    
                # Ensure we actually have a biclique (non-empty)
                if L_mask == 0 or R_mask == 0:
                    continue
                    
                size = L_mask.bit_count() * R_mask.bit_count()
                
                # Tie-breaking heuristic: prefer bicliques involving higher degree vertices?
                # For now, just max size.
                if size > best_size:
                    best_size = size
                    best_L_mask = L_mask
                    best_R_mask = R_mask

        if best_size == 0:
            # No edges left?
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if M[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            else:
                # Fallback: take a single edge
                for u in range(n):
                    if M[u]:
                        v_mask = M[u]
                        lsb = v_mask & -v_mask
                        v = lsb.bit_length() - 1
                        best_L_mask = 1 << u
                        best_R_mask = 1 << v
                        best_size = 1
                        break
                        
        # Convert masks to lists
        L_list = get_bits(best_L_mask)
        R_list = get_bits(best_R_mask)
        
        # Add to result
        result.append([L_list, R_list])
        
        # Remove edges from M
        # For each l in L, remove edges to all r in R
        for l in L_list:
            M[l] &= ~best_R_mask
        for r in R_list:
            M[r] &= ~best_L_mask
            
    # Validate and return
    # The problem requires returning a list of bicliques.
    # Types must be list of lists of ints.
    # We already constructed them as such.
    
    return result

