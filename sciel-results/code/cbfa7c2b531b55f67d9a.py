import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    N = 32
    MAX_VAL = 512

    def list_to_mask(lst):
        m = 0
        for x in lst:
            if 1 <= x <= MAX_VAL:
                m |= (1 << x)
        return m

    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in set using bitmask."""
        valid_diffs = 0
        # d can be at most (MAX_VAL-1)//2
        max_d = (MAX_VAL - 1) // 2
        for d in range(1, max_d + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs

    def score_list(lst):
        if len(lst) != N or len(set(lst)) != N:
            return 0
        if any(x < 1 or x > MAX_VAL for x in lst):
            return 0
        return score_mask(list_to_mask(lst))

    def sumset(A, B):
        """Compute the sumset A + B = {a + b for a in A, b in B}."""
        result = set()
        for a in A:
            for b in B:
                s = a + b
                if 1 <= s <= MAX_VAL:
                    result.add(s)
        return result

    def geometric_set(base, size, start=1):
        """Generate a geometric-like set: start * base^i for i in range(size)."""
        s = set()
        val = start
        for _ in range(size):
            if val <= MAX_VAL:
                s.add(val)
            val *= base
        return s

    def cantor_like_set(digits, max_val):
        """
        Cantor-like set: numbers whose base-(2*digits) representation 
        only uses the first 'digits' values in each position.
        This creates a fractal-like structure with rich additive properties.
        """
        s = set()
        base = 2 * digits
        # Generate all numbers up to max_val that use only digits 0..digits-1 in base 'base'
        def gen(pos, max_pos, current):
            if current > max_val:
                return
            if pos == max_pos:
                if current >= 1:
                    s.add(current)
                return
            for d in range(digits):
                gen(pos + 1, max_pos, current + d * (base ** pos))
        gen(0, 6, 0)  # 6 positions should cover up to ~512
        return s

    def structured_sumset_candidates():
        """
        Generate diverse structured sets using sumsets of smaller structured components.
        The key insight: if A and B are structured, A+B can have many 3-APs.
        """
        candidates = []
        
        # Strategy 1: Sum of two geometric-like sets
        for base_a in [2, 3, 4, 5, 6]:
            for base_b in [2, 3, 4, 5]:
                A = geometric_set(base_a, 8, start=1)
                B = geometric_set(base_b, 8, start=1)
                S = sumset(A, B)
                if len(S) >= N:
                    # Take the smallest N elements
                    sorted_s = sorted(S)[:N]
                    candidates.append(sorted_s)
        
        # Strategy 2: Cantor-like sets
        for digits in [2, 3, 4]:
            C = cantor_like_set(digits, MAX_VAL)
            if len(C) >= N:
                sorted_c = sorted(C)[:N]
                candidates.append(sorted_c)
        
        # Strategy 3: Sum of arithmetic progression and geometric set
        for ap_start in [1, 2, 3]:
            for ap_step in [1, 2, 3]:
                for ap_len in [6, 7, 8]:
                    A = [ap_start + i * ap_step for i in range(ap_len)]
                    for geo_base in [2, 3, 4]:
                        B = geometric_set(geo_base, 6, start=1)
                        S = sumset(set(A), B)
                        if len(S) >= N:
                            sorted_s = sorted(S)[:N]
                            candidates.append(sorted_s)
        
        # Strategy 4: Product-like structure (multiplicative)
        for base in [2, 3, 4]:
            for offset in [0, 1, 2]:
                S = set()
                for i in range(8):
                    for j in range(8):
                        val = base ** i + base ** j + offset
                        if 1 <= val <= MAX_VAL:
                            S.add(val)
                if len(S) >= N:
                    sorted_s = sorted(S)[:N]
                    candidates.append(sorted_s)
        
        # Strategy 5: Digit-restricted sets in mixed bases
        for b1 in [2, 3, 4]:
            for b2 in [2, 3, 4]:
                S = set()
                for i in range(6):
                    for j in range(6):
                        val = i * b1 + j * b1 * b2
                        if 1 <= val <= MAX_VAL:
                            S.add(val)
                if len(S) >= N:
                    sorted_s = sorted(S)[:N]
                    candidates.append(sorted_s)
        
        # Strategy 6: Union of two structured sets (careful about size)
        for base1 in [2, 3]:
            for base2 in [3, 4, 5]:
                A = geometric_set(base1, 6, start=1)
                B = geometric_set(base2, 6, start=1)
                S = A | B
                if len(S) >= N:
                    sorted_s = sorted(S)[:N]
                    candidates.append(sorted_s)
        
        # Strategy 7: "Lattice" sets - points on a 2D grid mapped to 1D
        for gx in range(3, 8):
            for gy in range(3, 8):
                for sx in [1, 2, 3]:
                    for sy in [2, 3, 4, 5]:
                        S = set()
                        for i in range(gx):
                            for j in range(gy):
                                val = 1 + i * sx + j * sy
                                if 1 <= val <= MAX_VAL:
                                    S.add(val)
                        if len(S) >= N:
                            sorted_s = sorted(S)[:N]
                            candidates.append(sorted_s)
        
        # Strategy 8: Fibonacci-like sums
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        for i in range(len(fibs)):
            for j in range(i, len(fibs)):
                S = set()
                for a_idx in range(max(0, i - 2), min(len(fibs), i + 5)):
                    for b_idx in range(max(0, j - 2), min(len(fibs), j + 5)):
                        val = fibs[a_idx] + fibs[b_idx]
                        if 1 <= val <= MAX_VAL:
                            S.add(val)
                if len(S) >= N:
                    sorted_s = sorted(S)[:N]
                    candidates.append(sorted_s)
        
        # Strategy 9: Powers of 2 with offsets
        for offset in range(0, 10):
            S = set()
            for i in range(9):
                for j in range(9):
                    val = (1 << i) + (1 << j) + offset
                    if 1 <= val <= MAX_VAL:
                        S.add(val)
            if len(S) >= N:
                sorted_s = sorted(S)[:N]
                candidates.append(sorted_s)
        
        # Strategy 10: Modular structure
        for mod in [7, 8, 9, 10, 11, 12]:
            for residues in range(mod):
                S = set()
                for k in range(1, MAX_VAL + 1):
                    if k % mod in [residues, (residues + 1) % mod, (residues + 2) % mod]:
                        S.add(k)
                if len(S) >= N:
                    sorted_s = sorted(S)[:N]
                    candidates.append(sorted_s)
        
        return candidates

    def bitmask_local_search(mask, time_limit):
        """
        Local search using bitmask representation.
        Moves: flip a bit (add/remove element), swap two elements.
        """
        current_mask = mask
        current_score = score_mask(mask)
        current_list = mask_to_list(mask)
        
        if len(current_list) != N:
            return mask, current_score
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_new_score = current_score
            best_new_mask = None
            
            # Try flipping bits (add or remove elements)
            # Only consider flips that maintain exactly N elements
            # Try adding a new element and removing an existing one (swap)
            current_set = set(current_list)
            
            # Swap: remove one, add another
            for remove_val in current_list:
                for add_val in range(1, MAX_VAL + 1):
                    if add_val in current_set:
                        continue
                    new_mask = current_mask & ~(1 << remove_val)
                    new_mask |= (1 << add_val)
                    new_list = mask_to_list(new_mask)
                    if len(new_list) != N:
                        continue
                    new_score = score_mask(new_mask)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_mask = new_mask
                        improved = True
            
            # Also try: change one element to a nearby value
            if not improved:
                for i, val in enumerate(current_list):
                    for delta in range(-5, 6):
                        if delta == 0:
                            continue
                        new_val = val + delta
                        if new_val < 1 or new_val > MAX_VAL:
                            continue
                        if new_val in current_set:
                            continue
                        new_mask = current_mask & ~(1 << val)
                        new_mask |= (1 << new_val)
                        new_score = score_mask(new_mask)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_mask = new_mask
                            improved = True
            
            if improved:
                current_mask = best_new_mask
                current_score = best_new_score
                current_list = mask_to_list(current_mask)
                current_set = set(current_list)
        
        return current_mask, current_score

    # Main strategy:
    # 1. Generate diverse structured candidates
    # 2. Score them all
    # 3. Take top candidates and refine with bitmask local search
    
    candidates = structured_sumset_candidates()
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    
    # Score all candidates
    scored = []
    for cand in unique_candidates:
        if len(cand) == N and len(set(cand)) == N and all(1 <= x <= MAX_VAL for x in cand):
            sc = score_list(cand)
            scored.append((sc, cand))
    
    scored.sort(key=lambda x: -x[0])
    
    best_score = 0
    best_list = None
    
    # Refine top candidates
    top_k = min(10, len(scored))
    for idx in range(top_k):
        if time.time() > deadline - 10.0:
            break
        
        sc, cand = scored[idx]
        if sc > best_score:
            best_score = sc
            best_list = cand[:]
        
        # Refine with local search
        mask = list_to_mask(cand)
        local_deadline = min(time.time() + 3.0, deadline - 5.0)
        refined_mask, refined_score = bitmask_local_search(mask, local_deadline)
        
        if refined_score > best_score:
            best_score = refined_score
            best_list = mask_to_list(refined_mask)
    
    # If we have time, do a few more diverse starts
    if time.time() < deadline - 15.0:
        random.seed(123)
        for attempt in range(20):
            if time.time() > deadline - 10.0:
                break
            
            # Random structured start: mix of small and large gaps
            gaps = []
            for i in range(31):
                if random.random() < 0.3:
                    gaps.append(random.randint(5, 15))
                else:
                    gaps.append(random.randint(1, 5))
            
            total = sum(gaps)
            if total + 1 > MAX_VAL:
                scale = (MAX_VAL - 1) / total
                gaps = [max(1, int(g * scale)) for g in gaps]
            
            s = [1]
            cur = 1
            for g in gaps:
                cur += g
                if cur > MAX_VAL:
                    break
                s.append(cur)
            
            if len(s) == N:
                sc = score_list(s)
                if sc > best_score:
                    best_score = sc
                    best_list = s[:]
                
                if sc >= best_score - 2:
                    mask = list_to_mask(s)
                    local_deadline = min(time.time() + 2.0, deadline - 5.0)
                    refined_mask, refined_score = bitmask_local_search(mask, local_deadline)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_list = mask_to_list(refined_mask)
    
    # Final validation and return
    if best_list is not None and len(best_list) == N and len(set(best_list)) == N and all(1 <= x <= MAX_VAL for x in best_list):
        return sorted(best_list)
    
    # Fallback
    return list(range(1, 33))

if __name__ == "__main__":
    res = construct()
    assert len(res) == 32
    assert len(set(res)) == 32
    assert all(1 <= x <= 512 for x in res)
    print("Score:", score_list(res))
