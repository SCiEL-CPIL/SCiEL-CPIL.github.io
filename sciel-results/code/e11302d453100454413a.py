import time
import random
import math

def construct():
    N = 256
    four_ap_masks = [[] for _ in range(N+1)]
    three_ap_masks = [[] for _ in range(N+1)]
    
    # Precomputation
    for a in range(1, N+1):
        for d in range(1, N+1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > N: break
            
            # 4-AP masks
            four_ap_masks[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (e-1)))
            four_ap_masks[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (e-1)))
            four_ap_masks[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (e-1)))
            four_ap_masks[e].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))
            
            # 3-AP masks
            three_ap_masks[a].append((1 << (b-1)) | (1 << (c-1)))
            three_ap_masks[b].append((1 << (a-1)) | (1 << (c-1)))
            three_ap_masks[c].append((1 << (a-1)) | (1 << (b-1)))

    def is_valid_add(mask, x):
        for m in four_ap_masks[x]:
            if mask & m == m:
                return False
        return True

    def get_gain_add(mask, x):
        gain = 0
        for m in three_ap_masks[x]:
            if mask & m == m:
                gain += 1
        return gain

    def get_loss_remove(mask, x):
        loss = 0
        for m in three_ap_masks[x]:
            if mask & m == m:
                loss += 1
        return loss

    best_mask = 0
    best_score = 0
    
    # Greedy Init
    for _ in range(20):
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        temp_mask = 0
        temp_score = 0
        for x in candidates:
            if is_valid_add(temp_mask, x):
                gain = get_gain_add(temp_mask, x)
                temp_mask |= (1 << (x-1))
                temp_score += gain
        if temp_score > best_score:
            best_score = temp_score
            best_mask = temp_mask
            
    current_mask = best_mask
    current_score = best_score
    
    start_time = time.time()
    time_limit = 80.0
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        if move_type < 0.3:
            x = random.randint(1, N)
            if not (current_mask & (1 << (x-1))):
                if is_valid_add(current_mask, x):
                    gain = get_gain_add(current_mask, x)
                    delta = gain
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask |= (1 << (x-1))
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        elif move_type < 0.6:
            x = random.randint(1, N)
            if current_mask & (1 << (x-1)):
                loss = get_loss_remove(current_mask, x)
                delta = -loss
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask &= ~(1 << (x-1))
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
        else:
            u = random.randint(1, N)
            v = random.randint(1, N)
            if u != v and (current_mask & (1 << (u-1))) and not (current_mask & (1 << (v-1))):
                mask_without_u = current_mask & ~(1 << (u-1))
                if is_valid_add(mask_without_u, v):
                    loss = get_loss_remove(current_mask, u)
                    gain = get_gain_add(mask_without_u, v)
                    delta = gain - loss
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = mask_without_u | (1 << (v-1))
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    result = []
    for i in range(1, N+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    return sorted(result)
