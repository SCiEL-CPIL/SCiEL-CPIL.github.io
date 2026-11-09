import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    # Precompute bit masks
    bit_masks = [0] * (N + 1)
    for i in range(1, N + 1):
        bit_masks[i] = 1 << (i - 1)

    # Helper to check if adding x to bitset creates a 3-AP
    # Optimized to iterate only over set bits
    def can_add(bitset, x):
        # x is not in bitset
        b = bitset
        while b:
            lsb = b & -b
            v = lsb.bit_length()
            b ^= lsb
            
            # Case 1: x is middle. u = 2x - v
            u = 2 * x - v
            if 1 <= u <= N:
                if bitset & bit_masks[u]:
                    return False
                
            # Case 2: v is middle. u = 2v - x
            u = 2 * v - x
            if 1 <= u <= N:
                if bitset & bit_masks[u]:
                    return False
                
            # Case 3: u is middle. v + x = 2u => u = (v+x)/2
            if (v + x) % 2 == 0:
                u = (v + x) // 2
                if 1 <= u <= N and u != v:
                    if bitset & bit_masks[u]:
                        return False
        return True

    def get_elements(bitset):
        return [i for i in range(1, N + 1) if bitset & bit_masks[i]]

    def get_non_elements(bitset):
        return [i for i in range(1, N + 1) if not (bitset & bit_masks[i])]

    # 1. Greedy Initialization
    current_bitset = 0
    current_score = 0.0
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda x: recip[x], reverse=True)
    
    for x in candidates:
        if can_add(current_bitset, x):
            current_bitset |= bit_masks[x]
            current_score += recip[x]

    best_bitset = current_bitset
    best_score = current_score

    # 2. Simulated Annealing
    temp = 0.05
    cooling_rate = 0.9999
    min_temp = 1e-6
    stuck_count = 0

    while time.time() - start_time < TIME_LIMIT:
        if temp < min_temp:
            temp = 0.05
            current_bitset = best_bitset
            current_score = best_score
            stuck_count = 0
            continue

        move_type = random.random()
        
        if move_type < 0.3:
            # Add move
            non_elements = get_non_elements(current_bitset)
            if non_elements:
                x_add = random.choice(non_elements)
                if can_add(current_bitset, x_add):
                    delta = recip[x_add]
                    current_bitset |= bit_masks[x_add]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset
                        stuck_count = 0
                    else:
                        stuck_count += 1

        elif move_type < 0.5:
            # Remove move
            elements = get_elements(current_bitset)
            if elements:
                x_rem = random.choice(elements)
                delta = -recip[x_rem]
                if random.random() < math.exp(delta / temp):
                    current_bitset &= ~bit_masks[x_rem]
                    current_score += delta
                    # Removals are usually bad for score, so don't reset stuck_count?
                    # But it might be necessary to escape.
                    stuck_count += 1

        else:
            # Swap move
            elements = get_elements(current_bitset)
            non_elements = get_non_elements(current_bitset)
            if elements and non_elements:
                x_rem = random.choice(elements)
                x_add = random.choice(non_elements)
                
                trial_bitset = current_bitset & ~bit_masks[x_rem]
                if can_add(trial_bitset, x_add):
                    trial_bitset |= bit_masks[x_add]
                    delta = recip[x_add] - recip[x_rem]
                    
                    if delta > 0:
                        current_bitset = trial_bitset
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                            stuck_count = 0
                        else:
                            stuck_count += 1
                    else:
                        if random.random() < math.exp(delta / temp):
                            current_bitset = trial_bitset
                            current_score += delta
                            stuck_count += 1
                        else:
                            stuck_count += 1
                else:
                    stuck_count += 1

        if stuck_count > 2000:
            current_bitset = best_bitset
            current_score = best_score
            stuck_count = 0

        temp *= cooling_rate

    # Final polish: try to add any number that can be added
    improved = True
    while improved:
        improved = False
        non_elements = get_non_elements(best_bitset)
        for x in non_elements:
            if can_add(best_bitset, x):
                best_bitset |= bit_masks[x]
                improved = True

    result = get_elements(best_bitset)
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
