import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin for 90s kill
    
    # Seed solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    def is_pairwise_coprime(nums):
        unique_nums = list(set(nums))
        n = len(unique_nums)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(unique_nums[i], unique_nums[j]) != 1:
                    return False
        return True

    def get_score(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if not is_pairwise_coprime(distinct_sums):
            return 0
        return len(distinct_sums)

    # Verify seed
    seed_score = get_score(best_A, best_B)
    if seed_score > best_score:
        best_score = seed_score

    # Search
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Generate random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        
        if not is_pairwise_coprime(distinct_sums):
            continue
            
        score = len(distinct_sums)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score == 9:
                break
            
        # Local Search
        # We try to improve the current A, B
        # We keep a copy of the current best in this lineage
        cur_A = A[:]
        cur_B = B[:]
        cur_score = score
        
        for step in range(200):
            if time.time() > deadline:
                break
            if best_score == 9:
                break
                
            # Mutate
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old_val = cur_A[idx]
                new_val = random.randint(1, 128)
                if new_val in cur_A and new_val != old_val:
                    continue
                cur_A[idx] = new_val
                if len(set(cur_A)) != 3:
                    cur_A[idx] = old_val
                    continue
                target = cur_A
                other = cur_B
                is_A = True
            else:
                idx = random.randint(0, 2)
                old_val = cur_B[idx]
                new_val = random.randint(1, 128)
                if new_val in cur_B and new_val != old_val:
                    continue
                cur_B[idx] = new_val
                if len(set(cur_B)) != 3:
                    cur_B[idx] = old_val
                    continue
                target = cur_B
                other = cur_A
                is_A = False

            sums = [a + b for a in cur_A for b in cur_B]
            distinct_sums = list(set(sums))
            
            if not is_pairwise_coprime(distinct_sums):
                # Revert
                target[idx] = old_val
                continue
            
            new_score = len(distinct_sums)
            if new_score > cur_score:
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_A = cur_A[:]
                    best_B = cur_B[:]
                    if best_score == 9:
                        break
            else:
                # Revert if not improved (greedy)
                # Or keep with some probability for simulated annealing?
                # Let's stick to greedy for now to ensure validity
                target[idx] = old_val

    return [best_A, best_B]
