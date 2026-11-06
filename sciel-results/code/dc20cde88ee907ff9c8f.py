import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map elements to APs containing them
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count

    def _exp(x):
        """Safe exponential for simulated annealing."""
        if x < -100:
            return 0.0
        if x > 100:
            return 1e100
        return math.exp(x)

    def is_valid_add(s, x):
        """Check if adding x to s creates a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if b in s and c in s and d in s:
                    return False
            elif x == b:
                if a in s and c in s and d in s:
                    return False
            elif x == c:
                if a in s and b in s and d in s:
                    return False
            else:
                if a in s and b in s and c in s:
                    return False
        return True

    def delta_add(s, x):
        """Change in 3-AP count if we add x to s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s:
                    delta += 1
            elif x == b:
                if a in s and c in s:
                    delta += 1
            else:
                if a in s and b in s:
                    delta += 1
        return delta

    def delta_remove(s, x):
        """Change in 3-AP count if we remove x from s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s:
                    delta += 1
            elif x == b:
                if a in s and c in s:
                    delta += 1
            else:
                if a in s and b in s:
                    delta += 1
        return -delta

    def is_valid_swap(s, x_out, x_in):
        """Check if swapping x_out for x_in is valid (no 4-APs)."""
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            others = []
            if x_in == a:
                others = [b, c, d]
            elif x_in == b:
                others = [a, c, d]
            elif x_in == c:
                others = [a, b, d]
            else:
                others = [a, b, c]
            
            all_in_s = True
            x_out_in_others = False
            for y in others:
                if y not in s:
                    all_in_s = False
                    break
                if y == x_out:
                    x_out_in_others = True
            
            if all_in_s and not x_out_in_others:
                return False
        return True

    def greedy_construct(random_seed):
        random.seed(random_seed)
        s = set()
        available = list(range(1, N + 1))
        
        while available:
            candidates = []
            for x in available:
                if is_valid_add(s, x):
                    d = delta_add(s, x)
                    candidates.append((d, x))
            
            if not candidates:
                break
                
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            top_k = min(5, len(candidates))
            if random.random() < 0.8:
                best_x = candidates[0][1]
            else:
                best_x = random.choice([c[1] for c in candidates[:top_k]])
            
            s.add(best_x)
            available.remove(best_x)
            
        return s

    def local_search_1move(s):
        """Try add and remove moves to improve 3-AP count."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                if not is_valid_add(s, x):
                    continue
                gain = delta_add(s, x)
                if gain > 0:
                    s.add(x)
                    improved = True
                    break
            if not improved:
                for x in list(s):
                    loss = delta_remove(s, x)
                    if loss >= 0:
                        s.discard(x)
                        improved = True
                        break
        return s

    def local_search_swap(s):
        """Try swap moves to improve 3-AP count."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            for x_out in list(s):
                loss = -delta_remove(s, x_out)
                if loss == 0:
                    continue
                for x_in in range(1, N + 1):
                    if x_in in s:
                        continue
                    if not is_valid_swap(s, x_out, x_in):
                        continue
                    gain = 0
                    for ap_idx in elem_to_3aps[x_in]:
                        a, b, c = three_aps[ap_idx]
                        if x_in == a:
                            if b in s and c in s and b != x_out and c != x_out:
                                gain += 1
                        elif x_in == b:
                            if a in s and c in s and a != x_out and c != x_out:
                                gain += 1
                        else:
                            if a in s and b in s and a != x_out and b != x_out:
                                gain += 1
                    
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_out, x_in)
            if best_swap is not None:
                x_out, x_in = best_swap
                s.remove(x_out)
                s.add(x_in)
                improved = True
        return s

    def count_4aps(s):
        count = 0
        for ap in four_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s and ap[3] in s:
                count += 1
        return count

    def repair(s, max_iterations=50):
        """Remove elements to break all 4-APs."""
        s = set(s)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            found = False
            for ap in four_aps:
                a, b, c, d = ap
                if a in s and b in s and c in s and d in s:
                    # Remove the element with the lowest 3-AP contribution
                    min_loss = None
                    min_elem = None
                    for x in [a, b, c, d]:
                        loss = -delta_remove(s, x)
                        if min_loss is None or loss < min_loss:
                            min_loss = loss
                            min_elem = x
                    s.discard(min_elem)
                    found = True
                    break
            if not found:
                break
        return s

    def simulated_annealing_penalized(s, t0=1.0, t1=0.01, steps=2000, penalty=50):
        """Simulated annealing that can temporarily accept 4-AP violations."""
        s = set(s)
        current_score = count_3aps(s)
        current_4aps = count_4aps(s)
        current_obj = current_score - penalty * current_4aps
        best_s = set(s)
        best_obj = current_obj
        
        for step in range(steps):
            if time.time() > deadline:
                break
            T = t0 * (t1 / t0) ** (step / steps)
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [x for x in range(1, N + 1) if x not in s]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_s = s | {x}
                delta_3 = delta_add(s, x)
                # Check if this creates a 4-AP
                delta_4 = 0
                for ap_idx in elem_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if x == a:
                        if b in s and c in s and d in s:
                            delta_4 += 1
                    elif x == b:
                        if a in s and c in s and d in s:
                            delta_4 += 1
                    elif x == c:
                        if a in s and b in s and d in s:
                            delta_4 += 1
                    else:
                        if a in s and b in s and c in s:
                            delta_4 += 1
                new_score = current_score + delta_3
                new_4aps = current_4aps + delta_4
                new_obj = new_score - penalty * new_4aps
            elif move_type < 0.7:
                # Remove a random element
                if not s:
                    continue
                x = random.choice(list(s))
                new_s = s - {x}
                delta_3 = delta_remove(s, x)
                # Check if this breaks a 4-AP
                delta_4 = 0
                for ap_idx in elem_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if x == a:
                        if b in s and c in s and d in s:
                            delta_4 -= 1
                    elif x == b:
                        if a in s and c in s and d in s:
                            delta_4 -= 1
                    elif x == c:
                        if a in s and b in s and d in s:
                            delta_4 -= 1
                    else:
                        if a in s and b in s and c in s:
                            delta_4 -= 1
                new_score = current_score + delta_3
                new_4aps = current_4aps + delta_4
                new_obj = new_score - penalty * new_4aps
            else:
                # Swap
                if not s:
                    continue
                x_out = random.choice(list(s))
                candidates_in = [x for x in range(1, N + 1) if x not in s]
                if not candidates_in:
                    continue
                x_in = random.choice(candidates_in)
                
                # Calculate changes
                loss_3 = -delta_remove(s, x_out)
                gain_3 = 0
                for ap_idx in elem_to_3aps[x_in]:
                    a, b, c = three_aps[ap_idx]
                    if x_in == a:
                        if b in s and c in s and b != x_out and c != x_out:
                            gain_3 += 1
                    elif x_in == b:
                        if a in s and c in s and a != x_out and c != x_out:
                            gain_3 += 1
                    else:
                        if a in s and b in s and a != x_out and b != x_out:
                            gain_3 += 1
                
                # Check 4-AP changes
                delta_4 = 0
                # 4-APs broken by removing x_out
                for ap_idx in elem_to_4aps[x_out]:
                    a, b, c, d = four_aps[ap_idx]
                    if x_out == a:
                        if b in s and c in s and d in s:
                            delta_4 -= 1
                    elif x_out == b:
                        if a in s and c in s and d in s:
                            delta_4 -= 1
                    elif x_out == c:
                        if a in s and b in s and d in s:
                            delta_4 -= 1
                    else:
                        if a in s and b in s and c in s:
                            delta_4 -= 1
                # 4-APs created by adding x_in
                for ap_idx in elem_to_4aps[x_in]:
                    a, b, c, d = four_aps[ap_idx]
                    if x_in == a:
                        if b in s and c in s and d in s and b != x_out and c != x_out and d != x_out:
                            delta_4 += 1
                    elif x_in == b:
                        if a in s and c in s and d in s and a != x_out and c != x_out and d != x_out:
                            delta_4 += 1
                    elif x_in == c:
                        if a in s and b in s and d in s and a != x_out and b != x_out and d != x_out:
                            delta_4 += 1
                    else:
                        if a in s and b in s and c in s and a != x_out and b != x_out and c != x_out:
                            delta_4 += 1
                
                new_score = current_score + gain_3 - loss_3
                new_4aps = current_4aps + delta_4
                new_obj = new_score - penalty * new_4aps
                new_s = (s - {x_out}) | {x_in}
            
            # Accept or reject
            delta_obj = new_obj - current_obj
            if delta_obj > 0 or random.random() < _exp(delta_obj / T):
                s = new_s
                current_score = new_score
                current_4aps = new_4aps
                current_obj = new_obj
                
                # Periodic repair
                if step % 100 == 0 and current_4aps > 0:
                    s = repair(s)
                    current_score = count_3aps(s)
                    current_4aps = count_4aps(s)
                    current_obj = current_score - penalty * current_4aps
                
                if current_4aps == 0 and current_score > best_obj:
                    best_obj = current_score
                    best_s = set(s)
        
        return best_s

    best_s = set()
    best_score = -1
    
    num_restarts = 25
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        s = greedy_construct(i)
        score = count_3aps(s)
        
        if score > best_score:
            best_score = score
            best_s = set(s)
        
        s = local_search_1move(s)
        score = count_3aps(s)
        
        if score > best_score:
            best_score = score
            best_s = set(s)
        
        s = local_search_swap(s)
        score = count_3aps(s)
        
        if score > best_score:
            best_score = score
            best_s = set(s)
        
        if time.time() < deadline - 15:
            s = simulated_annealing_penalized(s, t0=0.5, t1=0.01, steps=800, penalty=50)
            score = count_3aps(s)
            if count_4aps(s) == 0:
                if score > best_score:
                    best_score = score
                    best_s = set(s)
    
    if best_s and time.time() < deadline - 5:
        s = simulated_annealing_penalized(best_s, t0=0.3, t1=0.005, steps=2000, penalty=50)
        score = count_3aps(s)
        if count_4aps(s) == 0:
            if score > best_score:
                best_score = score
                best_s = set(s)
    
    return sorted(best_s)
