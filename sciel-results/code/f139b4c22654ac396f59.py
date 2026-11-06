import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0

def construct():
    start_time = time.time()
    best_size = 0
    best_A = []
    best_B = []
    
    def get_sum_bits(S):
        bits = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                bits |= (1 << (S[i] + S[j]))
        return bits

    def get_diff_bits(S):
        bits = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                bits |= (1 << abs(S[i] - S[j]))
        return bits

    def check_add(c, S, sum_bits, diff_bits, other_diff_bits):
        if (sum_bits >> (2*c)) & 1:
            return False
        for a in S:
            s = c + a
            if (sum_bits >> s) & 1:
                return False
            d = abs(c - a)
            if (diff_bits >> d) & 1:
                return False
            if (other_diff_bits >> d) & 1:
                return False
        return True

    def improve(A, B):
        # Randomized local search
        # 1. Try adds
        # 2. Try random swaps
        sum_bits_A = get_sum_bits(A)
        sum_bits_B = get_sum_bits(B)
        diff_bits_A = get_diff_bits(A)
        diff_bits_B = get_diff_bits(B)
        
        for _ in range(1000): # Limit iterations per restart
            # Try adds
            added = False
            for c in range(1, N+1):
                if c in A or c in B:
                    continue
                if check_add(c, A, sum_bits_A, diff_bits_A, diff_bits_B):
                    A.append(c)
                    sum_bits_A = get_sum_bits(A)
                    diff_bits_A = get_diff_bits(A)
                    added = True
                    break
                if check_add(c, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    B.append(c)
                    sum_bits_B = get_sum_bits(B)
                    diff_bits_B = get_diff_bits(B)
                    added = True
                    break
            if added:
                continue
                
            # Try random swap
            if A:
                idx = random.randint(0, len(A)-1)
                x = A[idx]
                c = random.randint(1, N)
                if c not in A and c not in B:
                    A_new = A[:idx] + A[idx+1:] + [c]
                    # Check validity of A_new
                    # We need to check if A_new is Sidon and diffs don't conflict with B
                    # This requires full check
                    sb = get_sum_bits(A_new)
                    db = get_diff_bits(A_new)
                    # Check Sidon: len(sums) == len(set(sums)) is not enough, we need unique pair sums
                    # Actually, get_sum_bits just sets bits. If two pairs have same sum, bit is set once.
                    # So we must check if number of pairs equals number of bits set.
                    num_pairs = len(A_new) * (len(A_new) + 1) // 2
                    if sb.bit_count() == num_pairs:
                        # Check diff conflict
                        if not (db & diff_bits_B):
                            A = A_new
                            sum_bits_A = sb
                            diff_bits_A = db
                            continue
                            
            if B:
                idx = random.randint(0, len(B)-1)
                x = B[idx]
                c = random.randint(1, N)
                if c not in A and c not in B:
                    B_new = B[:idx] + B[idx+1:] + [c]
                    sb = get_sum_bits(B_new)
                    db = get_diff_bits(B_new)
                    num_pairs = len(B_new) * (len(B_new) + 1) // 2
                    if sb.bit_count() == num_pairs:
                        if not (db & diff_bits_A):
                            B = B_new
                            sum_bits_B = sb
                            diff_bits_B = db
                            continue
                            
        return A, B

    while time.time() - start_time < DEADLINE:
        A = []
        B = []
        
        # Greedy build
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        sum_bits_A = 0
        sum_bits_B = 0
        diff_bits_A = 0
        diff_bits_B = 0
        
        for c in candidates:
            if check_add(c, A, sum_bits_A, diff_bits_A, diff_bits_B):
                A.append(c)
                sum_bits_A = get_sum_bits(A)
                diff_bits_A = get_diff_bits(A)
            elif check_add(c, B, sum_bits_B, diff_bits_B, diff_bits_A):
                B.append(c)
                sum_bits_B = get_sum_bits(B)
                diff_bits_B = get_diff_bits(B)
            
            if len(A) == MAX_EACH and len(B) == MAX_EACH:
                break
        
        A, B = improve(A, B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B

    return [best_A, best_B]
