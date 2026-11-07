import random
import time

def construct():
    N = 128
    deadline = time.time() + 85.0
    
    def can_add_bits(S_bits, x_idx):
        # x_idx is 0..127
        # x = x_idx + 1
        max_A = N - 1 - x_idx
        if max_A < 0:
            return True
        
        L_A = []
        for A in range(max_A + 1):
            if (S_bits >> A) & 1:
                if (S_bits >> (x_idx + A)) & 1:
                    L_A.append(A)
                    
        len_L = len(L_A)
        for i in range(len_L):
            A1 = L_A[i]
            for j in range(i + 1, len_L):
                A2 = L_A[j]
                if (S_bits >> (A1 + A2)) & 1:
                    return False
        return True

    def get_valid_adds(S_bits):
        adds = []
        for idx in range(N):
            if not (S_bits >> idx) & 1:
                if can_add_bits(S_bits, idx):
                    adds.append(idx)
        return adds

    best_size = 0
    best_bits = 0
    
    def bit_count(x):
        return bin(x).count('1')
        
    def bits_to_list(b):
        return [i+1 for i in range(N) if (b >> i) & 1]
        
    num_restarts = 5000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        S_bits = 0
        curr_size = 0
        
        # Greedy
        # Try a few orderings
        for order_type in range(3):
            if time.time() > deadline: break
            S_bits = 0
            curr_size = 0
            if order_type == 0:
                order = list(range(N))
            elif order_type == 1:
                order = list(range(N-1, -1, -1))
            else:
                order = list(range(N))
                random.shuffle(order)
            
            for idx in order:
                if can_add_bits(S_bits, idx):
                    S_bits |= (1 << idx)
                    curr_size += 1
            
            if curr_size > best_size:
                best_size = curr_size
                best_bits = S_bits
                
        # Local Search on the best of the greedy starts?
        # Or just continue from the last one?
        # Let's use the last one.
        
        # Local Search
        for step in range(200):
            if time.time() > deadline:
                break
                
            adds = get_valid_adds(S_bits)
            if adds:
                idx = random.choice(adds)
                S_bits |= (1 << idx)
                curr_size += 1
                if curr_size > best_size:
                    best_size = curr_size
                    best_bits = S_bits
                continue
            
            # No adds, try swaps
            swapped = False
            # Sample swaps
            for _ in range(50):
                idx_y = random.randint(0, N-1)
                if not (S_bits >> idx_y) & 1:
                    continue
                idx_x = random.randint(0, N-1)
                if (S_bits >> idx_x) & 1:
                    continue
                if idx_y == idx_x:
                    continue
                
                S_new = S_bits & ~(1 << idx_y)
                if can_add_bits(S_new, idx_x):
                    S_bits = S_new | (1 << idx_x)
                    swapped = True
                    break
            
            if not swapped:
                # Remove one
                if curr_size > 0:
                    # Remove random
                    idx_y = random.randint(0, N-1)
                    if (S_bits >> idx_y) & 1:
                        S_bits &= ~(1 << idx_y)
                        curr_size -= 1
                else:
                    break

    return bits_to_list(best_bits)
