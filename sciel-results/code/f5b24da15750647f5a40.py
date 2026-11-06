import time
import random

N = 256
MAX_SUM = 2 * N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 28.0
    
    best_set = None
    best_size = N + 1
    
    for attempt in range(50):
        if time.time() > deadline - 1.0:
            break
            
        S = []
        sum_present = [False] * (MAX_SUM + 1)
        A = list(range(1, N + 1))
        
        while A:
            if time.time() > deadline - 0.3:
                break
            
            if len(A) == 1:
                x = A[0]
                S.append(x)
                sum_present[2 * x] = True
                for y in S[:-1]:
                    sum_present[x + y] = True
                break
            
            best_x = None
            best_block_count = -1
            candidates = A[:]
            
            if attempt > 0:
                random.shuffle(candidates)
            
            for x in candidates:
                if sum_present[2 * x]:
                    continue
                can_add = True
                for y in S:
                    if sum_present[x + y]:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                block_count = 0
                for z in A:
                    if z == x:
                        continue
                    if sum_present[2 * z]:
                        block_count += 1
                        continue
                    blocked = False
                    for y in S:
                        if sum_present[z + y]:
                            blocked = True
                            break
                    if blocked:
                        block_count += 1
                        continue
                    if sum_present[z + x]:
                        block_count += 1
                        continue
                
                if block_count > best_block_count or (block_count == best_block_count and (best_x is None or x > best_x)):
                    best_block_count = block_count
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            sum_present[2 * best_x] = True
            for y in S[:-1]:
                sum_present[best_x + y] = True
            
            A.remove(best_x)
            to_remove = set()
            for z in A:
                if sum_present[2 * z]:
                    to_remove.add(z)
                    continue
                for y in S:
                    if sum_present[z + y]:
                        to_remove.add(z)
                        break
            if to_remove:
                A = [z for z in A if z not in to_remove]
        
        if S and len(S) < best_size:
            all_sums = set()
            valid = True
            for i, a in enumerate(S):
                for j in range(i, len(S)):
                    s_val = a + S[j]
                    if s_val in all_sums:
                        valid = False
                        break
                    all_sums.add(s_val)
                if not valid:
                    break
            if valid:
                maximal = True
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    can_add = True
                    if sum_present[2 * x]:
                        can_add = False
                    else:
                        for y in S:
                            if sum_present[x + y]:
                                can_add = False
                                break
                    if can_add:
                        maximal = False
                        break
                if maximal:
                    best_size = len(S)
                    best_set = list(S)
        
        if best_size <= 16:
            break

    if best_set:
        return best_set
    else:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
# EVOLVE-BLOCK-END
