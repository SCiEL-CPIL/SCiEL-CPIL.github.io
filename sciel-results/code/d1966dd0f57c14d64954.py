import random
import time
import numpy as np

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 28.0  # Safety margin for 60s limit

    def is_sidon(A):
        """Check if A is a Sidon set."""
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A):
        """Check if A is inclusion-maximal."""
        A_set = set(A)
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A_set:
                continue
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for y in A:
                    if x + y in sums:
                        compatible = False
                        break
            if compatible:
                return False
        return True

    def greedy_saturation(init_A=None, randomize_ties=False, seed=None, bias=0.0):
        """
        Greedy saturation: repeatedly add the element that blocks the most
        remaining candidates, until no more can be added (maximal).
        """
        if seed is not None:
            random.seed(seed)
            
        A = list(init_A) if init_A else []
        A_set = set(A)
        
        # Bitset of occupied sums
        sum_bits = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                sum_bits |= (1 << (A[i] + A[j]))
        
        # Bitset of available elements (not in A)
        avail = 0
        for x in range(1, N + 1):
            if x not in A_set:
                avail |= (1 << x)
        
        while avail:
            best_x = None
            best_score = -1
            best_sum_bits = None
            
            # Iterate over set bits in avail
            a = avail
            while a:
                # Get lowest set bit
                lb = a & (-a)
                x = lb.bit_length() - 1
                a ^= lb
                
                # Check compatibility: 2x not in sum_bits, and no x+y in sum_bits
                if (sum_bits >> (2 * x)) & 1:
                    continue
                compatible = True
                for y in A:
                    if (sum_bits >> (x + y)) & 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Compute new sums if we add x
                new_sum_bits = sum_bits | (1 << (2 * x))
                for y in A:
                    new_sum_bits |= (1 << (x + y))
                
                # Score: count how many available elements get blocked
                # An element z is blocked if:
                #   - 2z is in new_sum_bits, OR
                #   - there exists y in A U {x} such that z+y is in new_sum_bits
                score = 0
                b = avail & ~(1 << x)  # other available elements
                while b:
                    lb2 = b & (-b)
                    z = lb2.bit_length() - 1
                    b ^= lb2
                    
                    # Diagonal: 2z in new_sum_bits?
                    if (new_sum_bits >> (2 * z)) & 1:
                        score += 1
                        continue
                    
                    # Off-diagonal: z+y in new_sum_bits for some y in A U {x}?
                    blocked = False
                    if (new_sum_bits >> (z + x)) & 1:
                        blocked = True
                    else:
                        for y in A:
                            if (new_sum_bits >> (z + y)) & 1:
                                blocked = True
                                break
                    if blocked:
                        score += 1
                
                # Apply bias
                if bias != 0.0:
                    score -= x * bias
                
                # Apply randomization for tie-breaking
                if randomize_ties:
                    score += random.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_sum_bits = new_sum_bits
            
            if best_x is None:
                break
            
            A.append(best_x)
            A_set.add(best_x)
            sum_bits = best_sum_bits
            avail &= ~(1 << best_x)
        
        return A

    def try_remove_and_resaturate(A, i):
        """Remove A[i], check Sidon, then resaturate. Return result or None."""
        candidate = A[:i] + A[i+1:]
        if not candidate:
            return None
        if not is_sidon(candidate):
            return None
        return greedy_saturation(init_A=candidate)

    def local_search(A, time_budget):
        """
        Local search: try removals and swaps, resaturate, keep best.
        """
        A = list(A)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            current_size = len(A)
            
            # 1. Try all removals
            for i in range(len(A)):
                if time.time() > time_budget:
                    break
                result = try_remove_and_resaturate(A, i)
                if result is not None and len(result) < current_size:
                    A = result
                    current_size = len(A)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swaps: remove A[i], add random b, resaturate
            for i in range(len(A)):
                if time.time() > time_budget:
                    break
                a = A[i]
                A_minus = A[:i] + A[i+1:]
                
                # Try several random b's
                for _ in range(8):
                    b = random.randint(1, N)
                    if b in A_minus:
                        continue
                    candidate_swap = A_minus + [b]
                    if not is_sidon(candidate_swap):
                        continue
                    result = greedy_saturation(init_A=candidate_swap)
                    if result is not None and len(result) < current_size:
                        A = result
                        current_size = len(A)
                        improved = True
                        break
                if improved:
                    break
            
            # 3. Try 2-for-1: remove two elements, add one, resaturate
            if not improved and len(A) >= 3:
                for i in range(len(A)):
                    if time.time() > time_budget:
                        break
                    for j in range(i+1, len(A)):
                        A_minus2 = A[:i] + A[i+1:j] + A[j+1:]
                        for _ in range(3):
                            b = random.randint(1, N)
                            if b in A_minus2:
                                continue
                            candidate = A_minus2 + [b]
                            if not is_sidon(candidate):
                                continue
                            result = greedy_saturation(init_A=candidate)
                            if result is not None and len(result) < current_size:
                                A = result
                                current_size = len(A)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return A

    # Initial deterministic run
    best_set = greedy_saturation()
    best_size = len(best_set)
    
    # Apply local search
    improved = local_search(best_set, deadline - 5.0)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved

    # Randomized restarts
    attempt = 0
    while time.time() < deadline - 1.0:
        attempt += 1
        r = random.random()
        
        if r < 0.35:
            # Pure randomized greedy
            current_set = greedy_saturation(randomize_ties=True, seed=attempt)
        elif r < 0.55:
            # Start with random element
            init_x = random.randint(1, N)
            current_set = greedy_saturation(init_A=[init_x], randomize_ties=True, seed=attempt)
        elif r < 0.75:
            # Start with two random compatible elements
            x1 = random.randint(1, N)
            x2 = random.randint(1, N)
            tries = 0
            while (x2 == x1 or x1 + x2 == 2*x1 or x1 + x2 == 2*x2) and tries < 50:
                x2 = random.randint(1, N)
                tries += 1
            init = [x1, x2]
            if is_sidon(init):
                current_set = greedy_saturation(init_A=init, randomize_ties=True, seed=attempt)
            else:
                current_set = greedy_saturation(randomize_ties=True, seed=attempt)
        else:
            # Use bias to favor smaller or larger elements
            bias_val = random.uniform(0.005, 0.03) * (1 if random.random() < 0.5 else -1)
            current_set = greedy_saturation(randomize_ties=True, seed=attempt, bias=bias_val)
        
        # Apply local search if promising
        if len(current_set) <= best_size + 2 and time.time() < deadline - 3.0:
            improved = local_search(current_set, deadline - 1.0)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Size: {len(s)}")
    # Verify
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            assert sm not in sums, f"Sum {sm} repeated: {s[i]}+{s[j]}"
            sums.add(sm)
    print("Sidon: OK")
    # Check maximality
    s_set = set(s)
    for x in range(1, 257):
        if x in s_set:
            continue
        compatible = True
        if 2*x in sums:
            compatible = False
        else:
            for y in s:
                if x + y in sums:
                    compatible = False
                    break
        assert not compatible, f"Not maximal: {x} can be added"
    print("Maximal: OK")
