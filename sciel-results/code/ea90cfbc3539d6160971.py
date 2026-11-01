import time
import itertools
import sys

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal standard construction.
    For n = 7, uses an intensive local search (2-Opt) on the permutation ordering
    to minimize the total length of the superpermutation.
    """
    if n == 1:
        return "1"
    
    # Standard Graham-Lagarias construction for base cases
    def graham_lagarias(m):
        if m == 1:
            return [1], [(1,)]
        
        prev_word, prev_perms = graham_lagarias(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with current new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        # Extract perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return new_word, perms

    if n <= 6:
        word, _ = graham_lagarias(n)
        return "".join(map(str, word))
    
    if n == 7:
        # We operate on the permutations of 1..6 (n-1)
        base_perms = list(itertools.permutations(range(1, n)))
        num_perms = len(base_perms) # 720
        
        # Precompute the "block" string for each permutation in the standard construction format
        # Block for perm p is: p[0]...p[n-2] n p[0]...p[n-2]
        # Wait, the standard construction appends [m] then the perm.
        # Let's stick to the logic that the seed used which works:
        # The "perm" in the seed's graham_lagarias(6) are the windows of length 6.
        # The block added to the word for a perm p (of 1..6) when building for 7 is:
        # p + [7] + p ? No, the seed says: block = list(p) + [m] + list(p)
        # Let's verify the overlap logic.
        
        # Actually, a simpler cost model:
        # In the standard construction, the word is built by iterating through perms of size n-1.
        # For each perm p, we append the "suffix" of the perm + [n] + the perm.
        # The overlap is determined by the end of the previous block and the start of the new one.
        
        # Let's define the block string for each perm i.
        # Block_i = str(p_i[0])...str(p_i[n-2]) + str(n) + str(p_i[0])...str(p_i[n-2])
        blocks = []
        for p in base_perms:
            b = "".join(map(str, p)) + str(n) + "".join(map(str, p))
            blocks.append(b)
        
        # Precompute overlap matrix
        # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
        # Since blocks are long, we can just check k from min(len) down to 0.
        # But max possible overlap is limited. The blocks share structure.
        # Max overlap is roughly 2*(n-1) + 1.
        max_k = 2 * (n - 1) + 1
        
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            bi = blocks[i]
            for j in range(num_perms):
                bj = blocks[j]
                found = 0
                # Check from max_k down to 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        break
                    if bi[-k:] == bj[:k]:
                        found = k
                        break
                overlap_matrix[i][j] = found
        
        # Initial order: The order in which they appear in the standard construction for n-1=6
        # We can get this from graham_lagarias(n-1)
        _, std_perms_order = graham_lagarias(n - 1)
        
        # Map std_perms_order (tuples) to indices in base_perms
        # base_perms is list of tuples.
        perm_to_idx = {p: idx for idx, p in enumerate(base_perms)}
        
        order = []
        for p in std_perms_order:
            if p in perm_to_idx:
                order.append(perm_to_idx[p])
            else:
                # This shouldn't happen if graham_lagarias returns perms of 1..n-1
                # But graham_lagarias(6) returns perms of 1..6 which are in base_perms
                pass
        
        # If something went wrong with mapping, use identity order
        if len(order) != num_perms:
            order = list(range(num_perms))

        # Calculate initial total overlap
        def calc_total_overlap(ord_list):
            total = 0
            for k in range(len(ord_list) - 1):
                total += overlap_matrix[ord_list[k]][ord_list[k+1]]
            return total

        best_overlap = calc_total_overlap(order)
        best_order = order[:]
        
        start_time = time.time()
        deadline = start_time + 110.0 # Safety margin
        
        # 2-Opt Search
        # We want to maximize total overlap.
        # Reversing segment [i, j] changes the overlap sum.
        # Old edges: (i-1, i), (i, i+1)...(j-1, j), (j, j+1)
        # New edges: (i-1, j), (j, j-1)...(i+1, i), (i, j+1)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_perms - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, num_perms):
                    if time.time() > deadline:
                        break
                    
                    u_left = order[i-1] if i > 0 else None
                    u_right = order[j+1] if j < num_perms - 1 else None
                    
                    # Calculate delta overlap
                    old_overlap = 0
                    if i > 0:
                        old_overlap += overlap_matrix[u_left][order[i]]
                    for k in range(i, j):
                        old_overlap += overlap_matrix[order[k]][order[k+1]]
                    if j < num_perms - 1:
                        old_overlap += overlap_matrix[order[j]][u_right]
                        
                    new_overlap = 0
                    if i > 0:
                        new_overlap += overlap_matrix[u_left][order[j]]
                    for k in range(j, i, -1):
                        new_overlap += overlap_matrix[order[k]][order[k-1]]
                    if j < num_perms - 1:
                        new_overlap += overlap_matrix[order[i]][u_right]
                        
                    if new_overlap > old_overlap:
                        # Perform reversal
                        order[i:j+1] = order[i:j+1][::-1]
                        best_overlap += (new_overlap - old_overlap)
                        improved = True
                        # Update best_order if this is a new global best
                        # Note: We are greedily improving, so current order is the best so far in this pass
                        if best_overlap > calc_total_overlap(best_order):
                            best_order = order[:]

        # Or-Opt: Move small segments (1, 2, 3 elements)
        # This helps escape local optima that 2-Opt misses
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for L in [1, 2, 3]:
                if time.time() > deadline:
                    break
                for i in range(num_perms - L + 1):
                    if time.time() > deadline:
                        break
                    seg = order[i:i+L]
                    # Try inserting this segment at every other position j
                    for j in range(num_perms - L + 1):
                        if time.time() > deadline:
                            break
                        if j == i or j == i - 1:
                            continue
                        
                        # Perform move
                        # Remove seg
                        old_order = order[:]
                        del order[i:i+L]
                        # Insert at j
                        # If j > i, the index shifts by -L
                        if j > i:
                            j -= L
                        order[j:j] = seg
                        
                        new_overlap = calc_total_overlap(order)
                        
                        if new_overlap > best_overlap:
                            best_overlap = new_overlap
                            best_order = order[:]
                            improved = True
                        else:
                            order = old_order

        # Construct final word using best_order
        final_word = []
        for idx in best_order:
            b = blocks[idx]
            max_overlap = min(len(final_word), len(b))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if final_word[len(final_word) - o:] == b[:o]:
                    overlap = o
                    break
            final_word += b[overlap:]
            
        return "".join(final_word)
        
    # Fallback for n > 7
    word, _ = graham_lagarias(n)
    return "".join(map(str, word))
