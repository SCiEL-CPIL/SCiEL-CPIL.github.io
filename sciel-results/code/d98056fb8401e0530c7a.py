import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    total_space = q ** n

    # Precompute all words
    words = np.zeros((total_space, n), dtype=np.int8)
    for i in range(total_space):
        v = i
        for pos in range(n - 1, -1, -1):
            words[i, pos] = v % q
            v //= q

    # Precompute neighbor masks
    pos_mult = [q ** (n - 1 - pos) for pos in range(n)]
    neighbors = [0] * total_space
    for i in range(total_space):
        mask = 1 << i
        w = words[i]
        for pos in range(n):
            orig = w[pos]
            mult = pos_mult[pos]
            for delta in range(1, q):
                v1 = orig - delta
                if v1 >= 0:
                    mask |= (1 << (i - delta * mult))
                v2 = orig + delta
                if v2 < q:
                    mask |= (1 << (i + delta * mult))
        neighbors[i] = mask

    # Greedy Set Cover
    covered = 0
    code = []
    remaining = set(range(total_space))
    
    while remaining:
        if time.time() > deadline - 30:
            break
        best_idx = -1
        max_gain = 0
        max_possible = 1 + n * (q - 1)
        for i in remaining:
            gain = (neighbors[i] & ~covered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                if max_gain == max_possible:
                    break
        if best_idx == -1:
            break
        code.append(best_idx)
        covered |= neighbors[best_idx]
        remaining.discard(best_idx)
        
    code_set = set(code)
    
    # Local Search using coverage counts
    # covered_by_count[j] = number of codewords that cover word j
    covered_by_count = np.zeros(total_space, dtype=np.int32)
    for c in code:
        mask = neighbors[c]
        # Extract set bits and increment
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            covered_by_count[j] += 1
            mask ^= lsb

    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        
        # Pass 1: Deletion
        for i in range(len(code)):
            w_i = code[i]
            mask = neighbors[w_i]
            # Check if any word in neighbors[w_i] has count == 1
            # If all counts > 1, w_i is redundant
            redundant = True
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if covered_by_count[j] == 1:
                    redundant = False
                    break
                mask ^= lsb
            if redundant:
                # Remove w_i
                for j in range(total_space):
                    pass  # Need to decrement
                # Decrement counts for neighbors[w_i]
                mask2 = neighbors[w_i]
                while mask2:
                    lsb = mask2 & -mask2
                    j = lsb.bit_length() - 1
                    covered_by_count[j] -= 1
                    mask2 ^= lsb
                code.pop(i)
                code_set.remove(w_i)
                improved = True
                break
        
        # Pass 2: Swap
        if not improved:
            for i in range(len(code)):
                w_i = code[i]
                unique_mask = 0
                mask = neighbors[w_i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if covered_by_count[j] == 1:
                        unique_mask |= lsb
                    mask ^= lsb
                if unique_mask == 0:
                    continue
                
                # Find best swap
                best_j = -1
                max_gain = -1
                for j in range(total_space):
                    if j in code_set:
                        continue
                    # Check if j covers all unique words
                    if (unique_mask & ~neighbors[j]) != 0:
                        continue
                    # Compute gain: how many new words j covers
                    # gain = popcount(neighbors[j] & ~covered_by_others)
                    # But we don't have covered_by_others as a bitset.
                    # Instead, gain = popcount(neighbors[j]) - (words already covered by others)
                    # This is hard. Let's just use a heuristic: prefer j with high degree?
                    # Or just take the first valid one.
                    # For now, take first valid to keep it fast.
                    best_j = j
                    break
                
                if best_j != -1:
                    # Perform swap: remove w_i, add best_j
                    # Decrement counts for w_i
                    mask = neighbors[w_i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        covered_by_count[j] -= 1
                        mask ^= lsb
                    # Increment counts for best_j
                    mask = neighbors[best_j]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        covered_by_count[j] += 1
                        mask ^= lsb
                    code[i] = best_j
                    code_set.remove(w_i)
                    code_set.add(best_j)
                    improved = True
                    break

    return [words[i].tolist() for i in code]
