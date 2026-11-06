import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 58.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_a, best_b = [], []
    best_size = 0
    
    def can_add(x, S, sum_bits, diff_bits, other_diff_bits):
        # Check sum collisions: x+s for s in S, and x+x
        # If (x+s) collides with an existing sum, invalid.
        # Existing sums are in sum_bits.
        # New sums are x+s for s in S, and 2x.
        # Collision if (sum_bits >> (x+s)) & 1 or (sum_bits >> (2x)) & 1
        
        # Optimization: Check 2x first as it's a single op
        if (sum_bits >> (2 * x)) & 1:
            return False
            
        for s in S:
            if (sum_bits >> (x + s)) & 1:
                return False
                
        # Check diff collisions with other set
        # New diffs: x-s for s in S, s-x for s in S.
        # Collision if (other_diff_bits >> (x-s+255)) & 1 or (other_diff_bits >> (s-x+255)) & 1
        # Note: diff_bits of current set is not checked here because we assume S is valid.
        # But wait, do new diffs collide with existing diffs in S?
        # The problem statement says "Pair sums inside each set... must be unique".
        # It does NOT explicitly say differences inside a set must be unique (that's a B_2 sequence, not Sidon).
        # Sidon set definition: all sums a_i + a_j (i <= j) are distinct.
        # It implies that if a_i + a_j = a_k + a_l, then {a_i, a_j} = {a_k, a_l}.
        # This does NOT require differences to be unique within the set.
        # Example: {1, 2, 4}. Sums: 2, 3, 5, 4, 6, 8. All unique.
        # Diffs: 1, 3, 2. All unique here.
        # Example: {1, 3, 4}. Sums: 2, 4, 5, 6, 7, 8. All unique.
        # Diffs: 2, 3, 1. All unique.
        # Is it possible to have a Sidon set with repeated differences?
        # If a-b = c-d, then a+d = b+c.
        # If {a,d} != {b,c}, then this is a collision of sums.
        # If {a,d} = {b,c}, then a=b and d=c (so b=d, a=c) or a=c and d=b.
        # If a=c and d=b, then a-b = a-b, trivial.
        # So, in a Sidon set, all non-zero differences are distinct!
        # Proof: If a-b = c-d and a!=b, c!=d.
        # Then a+d = b+c.
        # Since it's a Sidon set, the pair {a,d} must equal {b,c}.
        # Case 1: a=b and d=c. Then a-b=0, contradiction (non-zero diff).
        # Case 2: a=c and d=b. Then a-b = a-b. This is the same difference.
        # So, yes, in a Sidon set, all non-zero absolute differences are distinct.
        # Therefore, we MUST also check that new diffs do not collide with existing diffs in the SAME set.
        
        for s in S:
            d1 = x - s + 255
            d2 = s - x + 255
            # Check against other set
            if (other_diff_bits >> d1) & 1 or (other_diff_bits >> d2) & 1:
                return False
            # Check against current set
            if (diff_bits >> d1) & 1 or (diff_bits >> d2) & 1:
                return False
                
        return True

    def greedy_build(order_a, order_b):
        S_a = []
        S_b = []
        sum_bits_a = 0
        sum_bits_b = 0
        diff_bits_a = 0
        diff_bits_b = 0
        
        # We will iterate through a combined list? Or separate?
        # The current seed uses separate lists.
        # Let's try a unified candidate list where we decide which set to add to.
        # This allows better balancing.
        
        # Actually, the seed's approach of building A then B is suboptimal because B is built
        # with A's diffs already fixed, limiting B's options.
        # Building them simultaneously allows more flexibility.
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if len(S_a) < MAX_SIZE and can_add(x, S_a, sum_bits_a, diff_bits_a, diff_bits_b):
                S_a.append(x)
                for s in S_a[:-1]:
                    sum_bits_a |= (1 << (x + s))
                    diff_bits_a |= (1 << (x - s + 255))
                    diff_bits_a |= (1 << (s - x + 255))
                sum_bits_a |= (1 << (2 * x))
                continue
            
            # Try adding to B
            if len(S_b) < MAX_SIZE and can_add(x, S_b, sum_bits_b, diff_bits_b, diff_bits_a):
                S_b.append(x)
                for s in S_b[:-1]:
                    sum_bits_b |= (1 << (x + s))
                    diff_bits_b |= (1 << (x - s + 255))
                    diff_bits_b |= (1 << (s - x + 255))
                sum_bits_b |= (1 << (2 * x))
                continue
            
            # If neither, skip x
        
        return S_a, S_b

    def local_improve(S_a, S_b):
        # Rebuild bits
        sum_bits_a = 0
        sum_bits_b = 0
        diff_bits_a = 0
        diff_bits_b = 0
        
        for a in S_a:
            for b in S_a:
                if a <= b:
                    sum_bits_a |= (1 << (a + b))
            for b in S_a:
                if a != b:
                    diff_bits_a |= (1 << (a - b + 255))
                    
        for a in S_b:
            for b in S_b:
                if a <= b:
                    sum_bits_b |= (1 << (a + b))
            for b in S_b:
                if a != b:
                    diff_bits_b |= (1 << (a - b + 255))
        
        set_a = set(S_a)
        set_b = set(S_b)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try to add to A
            for x in range(1, N + 1):
                if x in set_a:
                    continue
                if len(S_a) >= MAX_SIZE:
                    break
                if can_add(x, S_a, sum_bits_a, diff_bits_a, diff_bits_b):
                    S_a.append(x)
                    set_a.add(x)
                    for s in S_a[:-1]:
                        sum_bits_a |= (1 << (x + s))
                        diff_bits_a |= (1 << (x - s + 255))
                        diff_bits_a |= (1 << (s - x + 255))
                    sum_bits_a |= (1 << (2 * x))
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try to add to B
            for x in range(1, N + 1):
                if x in set_b:
                    continue
                if len(S_b) >= MAX_SIZE:
                    break
                if can_add(x, S_b, sum_bits_b, diff_bits_b, diff_bits_a):
                    S_b.append(x)
                    set_b.add(x)
                    for s in S_b[:-1]:
                        sum_bits_b |= (1 << (x + s))
                        diff_bits_b |= (1 << (x - s + 255))
                        diff_bits_b |= (1 << (s - x + 255))
                    sum_bits_b |= (1 << (2 * x))
                    improved = True
                    break
        
        return S_a, S_b

    random.seed(42)
    
    while time.time() < deadline - 2.0:
        a, b = greedy_build(None, None)
        current_size = len(a) + len(b)
        
        if current_size > best_size:
            best_size = current_size
            best_a, best_b = a, b
            
        if time.time() < deadline - 3.0:
            a_imp, b_imp = local_improve(a, b)
            imp_size = len(a_imp) + len(b_imp)
            if imp_size > best_size:
                best_size = imp_size
                best_a, best_b = a_imp, b_imp
                
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
        
    return [best_a, best_b]
