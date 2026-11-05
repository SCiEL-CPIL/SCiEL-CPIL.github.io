import time
import random
import sys

N = 256
DEADLINE_MARGIN = 10  # seconds before the 90s limit

def count_3aps_fast(A):
    count = 0
    for d in range(1, 128):
        m = A & (A >> d) & (A >> (2*d))
        count += m.bit_count()
    return count

def has_4ap_fast(A):
    for d in range(1, 86):
        m = A & (A >> d) & (A >> (2*d)) & (A >> (3*d))
        if m:
            return True
    return False

def bitmask_to_list(A):
    return [i+1 for i in range(N) if (A >> i) & 1]

def list_to_bitmask(lst):
    A = 0
    for x in lst:
        A |= (1 << (x-1))
    return A

def greedy_construct_random(seed):
    random.seed(seed)
    order = list(range(1, N+1))
    random.shuffle(order)
    A = 0
    for x in order:
        bit = 1 << (x-1)
        new_A = A | bit
        if not has_4ap_fast(new_A):
            A = new_A
    return A

def local_search(A, time_limit):
    start_time = time.time()
    current_score = count_3aps_fast(A)
    best_A = A
    best_score = current_score
    
    T = 5.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit and T > 0.01:
        # Pick a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Find unselected elements
            # To speed up, we can just pick a random index
            x = random.randint(1, N)
            bit = 1 << (x-1)
            if A & bit:
                continue # Already selected
            new_A = A | bit
            if has_4ap_fast(new_A):
                T *= cooling
                continue
            new_score = count_3aps_fast(new_A)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                A = new_A
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
        elif move_type == 'remove':
            if A == 0:
                T *= cooling
                continue
            x = random.randint(1, N)
            bit = 1 << (x-1)
            if not (A & bit):
                continue # Not selected
            new_A = A ^ bit
            new_score = count_3aps_fast(new_A)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                A = new_A
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
        else: # swap
            x = random.randint(1, N)
            y = random.randint(1, N)
            bit_x = 1 << (x-1)
            bit_y = 1 << (y-1)
            # x must be out, y must be in
            if (A & bit_x) or not (A & bit_y):
                T *= cooling
                continue
            new_A = (A ^ bit_y) | bit_x
            if has_4ap_fast(new_A):
                T *= cooling
                continue
            new_score = count_3aps_fast(new_A)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                A = new_A
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
        
        T *= cooling
        
    return best_A, best_score

def construct():
    start_time = time.time()
    
    best_A = 0
    best_score = -1
    
    # Multiple random greedy starts
    for i in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        A = greedy_construct_random(i)
        score = count_3aps_fast(A)
        if score > best_score:
            best_score = score
            best_A = A
            
    # Local search on the best one
    if best_A and time.time() - start_time < DEADLINE_MARGIN:
        remaining = DEADLINE_MARGIN - (time.time() - start_time)
        if remaining > 5:
            best_A, best_score = local_search(best_A, remaining)
            
    # Convert back to list
    return bitmask_to_list(best_A)

if __name__ == "__main__":
    result = construct()
    # Validation
    A = list_to_bitmask(result)
    print(f"Size: {len(result)}")
    print(f"3-APs: {count_3aps_fast(A)}")
    print(f"4-APs: {has_4ap_fast(A)}")
