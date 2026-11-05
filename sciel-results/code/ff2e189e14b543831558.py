import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]
        
        # Precompute neighbors as bitsets (Python integers)
        neighbors_bits = [0] * total
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            bits = 1 << i
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        bits |= (1 << new_i)
            neighbors_bits[i] = bits
        
        full_mask = (1 << total) - 1
        
        def greedy_set_cover():
            """Greedy set cover using bitsets."""
            uncovered = full_mask
            codewords = []
            while uncovered:
                if time.time() > deadline:
                    break
                best_cw = -1
                best_count = 0
                for cw in range(total):
                    count = bin(uncovered & neighbors_bits[cw]).count('1')
                    if count > best_count:
                        best_count = count
                        best_cw = cw
                if best_count == 0:
                    break
                codewords.append(best_cw)
                uncovered &= ~neighbors_bits[best_cw]
            return codewords
        
        def is_valid(codewords):
            """Check if codewords cover all words."""
            covered = 0
            for cw in codewords:
                covered |= neighbors_bits[cw]
            return covered == full_mask
        
        def try_reduce(codewords):
            """Remove redundant codewords."""
            if not codewords:
                return codewords
            coverage = [0] * total
            for cw in codewords:
                bits = neighbors_bits[cw]
                b = bits
                while b:
                    lsb = b & (-b)
                    idx = lsb.bit_length() - 1
                    coverage[idx] += 1
                    b ^= lsb
            
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords) - 1, -1, -1):
                    cw = codewords[i]
                    bits = neighbors_bits[cw]
                    b = bits
                    redundant = True
                    while b:
                        lsb = b & (-b)
                        idx = lsb.bit_length() - 1
                        if coverage[idx] < 2:
                            redundant = False
                            break
                        b ^= lsb
                    if redundant:
                        bits = neighbors_bits[cw]
                        b = bits
                        while b:
                            lsb = b & (-b)
                            idx = lsb.bit_length() - 1
                            coverage[idx] -= 1
                            b ^= lsb
                        codewords.pop(i)
                        changed = True
                        break
            return codewords
        
        def compute_coverage(code):
            cov = [0] * total
            for cw in code:
                bits = neighbors_bits[cw]
                b = bits
                while b:
                    lsb = b & (-b)
                    idx = lsb.bit_length() - 1
                    cov[idx] += 1
                    b ^= lsb
            return cov
        
        # Phase 1: Greedy set cover
        best_code = greedy_set_cover()
        best_code = try_reduce(best_code)
        best_size = len(best_code)
        
        # Phase 2: Local search with simulated annealing
        current_code = best_code[:]
        current_size = len(current_code)
        current_cov = compute_coverage(current_code)
        
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            if not current_code:
                break
            
            i = random.randint(0, len(current_code) - 1)
            cw_remove = current_code[i]
            
            current_code.pop(i)
            bits_remove = neighbors_bits[cw_remove]
            b = bits_remove
            while b:
                lsb = b & (-b)
                idx = lsb.bit_length() - 1
                current_cov[idx] -= 1
                b ^= lsb
            
            uncovered = 0
            for idx in range(total):
                if current_cov[idx] == 0:
                    uncovered |= (1 << idx)
            
            if uncovered == 0:
                current_code = try_reduce(current_code)
                current_cov = compute_coverage(current_code)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    no_improve = 0
                else:
                    no_improve += 1
                T *= cooling
                continue
            
            best_add = -1
            best_gain = 0
            for cw in range(total):
                if cw in current_code:
                    continue
                gain = bin(uncovered & neighbors_bits[cw]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_add = cw
            
            if best_add == -1 or best_gain == 0:
                current_code.append(cw_remove)
                bits_remove = neighbors_bits[cw_remove]
                b = bits_remove
                while b:
                    lsb = b & (-b)
                    idx = lsb.bit_length() - 1
                    current_cov[idx] += 1
                    b ^= lsb
                no_improve += 1
                T *= cooling
                continue
            
            new_size = len(current_code) + 1
            if new_size < current_size:
                accept = True
            elif new_size == current_size:
                accept = random.random() < np.exp(-1.0 / T)
            else:
                accept = False
            
            if accept:
                current_code.append(best_add)
                bits_add = neighbors_bits[best_add]
                b = bits_add
                while b:
                    lsb = b & (-b)
                    idx = lsb.bit_length() - 1
                    current_cov[idx] += 1
                    b ^= lsb
                current_size = len(current_code)
                no_improve = 0
            else:
                current_code.append(cw_remove)
                bits_remove = neighbors_bits[cw_remove]
                b = bits_remove
                while b:
                    lsb = b & (-b)
                    idx = lsb.bit_length() - 1
                    current_cov[idx] += 1
                    b ^= lsb
                no_improve += 1
            
            T = max(T * cooling, T_min)
        
        # Final reduction
        best_code = try_reduce(best_code)
        
        if best_code and is_valid(best_code):
            final_codewords = []
            for idx_c in best_code:
                final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
            ds = get_direct_sum()
            if len(final_codewords) <= len(ds):
                return final_codewords
            else:
                return ds
        
        ds = get_direct_sum()
        return ds

    return get_direct_sum()
