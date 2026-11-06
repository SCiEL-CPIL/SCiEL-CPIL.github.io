import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    qres = set()
    for x in range(1, 19):
        qres.add((x*x) % 19)
    
    def build_linear_code():
        G = np.zeros((9, 18), dtype=np.uint8)
        for i in range(9):
            for j in range(17):
                pos = (j - i) % 19
                if pos in qres:
                    G[i, j] = 1
            G[i, 17] = G[i, :17].sum() % 2
        codewords = set()
        for mask in range(256):
            cw = np.zeros(18, dtype=np.uint8)
            for i in range(9):
                if (mask >> i) & 1:
                    cw = (cw + G[i]) % 2
            val = 0
            for j in range(18):
                val = (val << 1) | int(cw[j])
            codewords.add(val)
        return codewords, G
    
    all_cw_18, G = build_linear_code()
    
    cw_17 = set()
    for val in all_cw_18:
        if (val & 1) == 0:
            cw_17.add(val >> 1)
    
    cw_list = list(cw_17)
    min_d = 999
    for i in range(len(cw_list)):
        for j in range(i+1, len(cw_list)):
            d = (cw_list[i] ^ cw_list[j]).bit_count()
            if d < min_d:
                min_d = d
    
    if min_d < 6:
        best_shortened = None
        best_min_d = 0
        for coord in range(18):
            cw_short = set()
            for val in all_cw_18:
                if ((val >> (17 - coord)) & 1) == 0:
                    upper = val >> (18 - coord)
                    lower = val & ((1 << (17 - coord)) - 1)
                    cw_short.add((upper << (17 - coord)) | lower)
            if len(cw_short) == 256:
                cl = list(cw_short)
                md = 999
                for i in range(len(cl)):
                    for j in range(i+1, len(cl)):
                        d = (cl[i] ^ cl[j]).bit_count()
                        if d < md:
                            md = d
                if md > best_min_d:
                    best_min_d = md
                    best_shortened = cw_short
        if best_shortened is not None:
            cw_list = list(best_shortened)
            min_d = best_min_d
        else:
            cw_list = list(cw_17)
    
    if min_d < 6:
        cw_set = set()
        for w in range(1 << 17):
            if all((w ^ cw).bit_count() >= 6 for cw in cw_set):
                cw_set.add(w)
        cw_list = list(cw_set)
    
    code_list = list(cw_list)
    code_set = set(cw_list)
    
    best_code = list(code_list)
    best_size = len(code_list)
    
    all_words = np.arange(1 << 17, dtype=np.int64)
    rng = random.Random(42)
    
    # Precompute popcount table for 16-bit values
    _pc = np.empty(65536, dtype=np.int8)
    for _i in range(65536):
        _pc[_i] = _i.bit_count()
    popcount_table = _pc
    
    def fast_conflicts(new_word, code_arr):
        """Return indices of codewords with Hamming distance < MIN_DIST from new_word."""
        n = len(code_arr)
        if n == 0:
            return np.array([], dtype=np.int64)
        xors = code_arr ^ new_word
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        return np.where(dists < MIN_DIST)[0]
    
    def check_valid(candidate, code_arr, skip_idx=-1):
        """Check if candidate has distance >= MIN_DIST from all codewords except skip_idx."""
        if len(code_arr) == 0:
            return True
        xors = code_arr ^ candidate
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        if skip_idx >= 0:
            dists[skip_idx] = 999
        return np.all(dists >= MIN_DIST)
    
    def try_add_word(new_word, code_list, code_set):
        if new_word in code_set:
            return False, code_list, code_set
        
        code_arr = np.array(code_list, dtype=np.int64)
        conflict_idx = fast_conflicts(new_word, code_arr)
        
        if len(conflict_idx) == 0:
            return True, code_list + [new_word], code_set | {new_word}
        
        new_code = list(code_list)
        new_set = set(code_set)
        
        for ci in conflict_idx:
            idx = int(ci)
            cw = new_code[idx]
            fixed = False
            
            # Try single-bit flips
            for bit in range(N):
                new_cw = cw ^ (1 << bit)
                if new_cw == new_word:
                    continue
                if (new_cw ^ new_word).bit_count() < MIN_DIST:
                    continue
                new_code_arr = np.array(new_code, dtype=np.int64)
                if check_valid(new_cw, new_code_arr, skip_idx=idx):
                    new_set.discard(cw)
                    new_code[idx] = new_cw
                    new_set.add(new_cw)
                    fixed = True
                    break
            
            if not fixed:
                # Try two-bit flips
                for bit1 in range(N):
                    if fixed:
                        break
                    for bit2 in range(bit1 + 1, N):
                        new_cw = cw ^ (1 << bit1) ^ (1 << bit2)
                        if new_cw == new_word:
                            continue
                        if (new_cw ^ new_word).bit_count() < MIN_DIST:
                            continue
                        new_code_arr = np.array(new_code, dtype=np.int64)
                        if check_valid(new_cw, new_code_arr, skip_idx=idx):
                            new_set.discard(cw)
                            new_code[idx] = new_cw
                            new_set.add(new_cw)
                            fixed = True
                            break
            
            if not fixed:
                new_code.pop(idx)
                new_set.discard(cw)
                code_arr2 = np.array(new_code, dtype=np.int64)
                remaining = fast_conflicts(new_word, code_arr2)
                if len(remaining) > 0:
                    return False, code_list, code_set
        
        code_arr_final = np.array(new_code, dtype=np.int64)
        final_conf = fast_conflicts(new_word, code_arr_final)
        if len(final_conf) > 0:
            return False, code_list, code_set
        
        new_code.append(new_word)
        new_set.add(new_word)
        return True, new_code, new_set
    
    T0 = 10.0
    T_final = 0.01
    cooling = 0.9995
    
    current_code = list(code_list)
    current_set = set(code_list)
    T = T0
    
    no_improve_count = 0
    
    while time.time() < deadline:
        if time.time() - start_time > 52:
            break
        
        new_word = int(rng.choice(all_words))
        
        if new_word in current_set:
            no_improve_count += 1
            if no_improve_count > 20000:
                current_code = list(best_code)
                current_set = set(best_code)
                T = T0
                no_improve_count = 0
            continue
        
        success, new_code, new_set = try_add_word(new_word, current_code, current_set)
        
        if success:
            current_code = new_code
            current_set = new_set
            no_improve_count = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)
        else:
            no_improve_count += 1
        
        T = max(T * cooling, T_final)
        
        if no_improve_count > 5000:
            current_code = list(best_code)
            current_set = set(best_code)
            T = T0
            no_improve_count = 0
    
    result = best_code
    
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST}"
    
    return result
