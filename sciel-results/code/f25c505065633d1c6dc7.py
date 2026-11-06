import random
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_set = set()
    best_score = -1
    
    def count_3aps(A):
        count = 0
        for a in A:
            for d in range(1, (N - a) // 2 + 1):
                b = a + d
                c = a + 2 * d
                if b in A and c in A:
                    count += 1
        return count

    def is_4ap_free_after_add(A, x):
        for d in range(1, (N - 1) // 3 + 1):
            if (x + d in A) and (x + 2 * d in A) and (x + 3 * d in A):
                return False
            if x - d >= 1 and (x - d in A) and (x + d in A) and (x + 2 * d in A):
                return False
            if x - 2 * d >= 1 and (x - 2 * d in A) and (x - d in A) and (x + d in A):
                return False
            if x - 3 * d >= 1 and (x - 3 * d in A) and (x - 2 * d in A) and (x - d in A):
                return False
        return True

    def gain_3aps(A, x):
        gain = 0
        for d in range(1, (N - 1) // 2 + 1):
            if x - d >= 1 and x + d <= N:
                if (x - d in A) and (x + d in A):
                    gain += 1
            if x + 2 * d <= N:
                if (x + d in A) and (x + 2 * d in A):
                    gain += 1
            if x - 2 * d >= 1:
                if (x - 2 * d in A) and (x - d in A):
                    gain += 1
        return gain

    def local_search(A):
        A = set(A)
        score = count_3aps(A)
        
        while time.time() < deadline:
            improved = False
            
            # Try to add elements
            candidates = []
            for x in range(1, N + 1):
                if x not in A:
                    if is_4ap_free_after_add(A, x):
                        g = gain_3aps(A, x)
                        candidates.append((g, x))
            
            if candidates:
                candidates.sort(reverse=True)
                best_g, best_x = candidates[0]
                A.add(best_x)
                score += best_g
                improved = True
                continue
            
            # If no adds, try swaps
            swap_improved = False
            # Try a few random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(A) < 2:
                    break
                x = random.choice(list(A))
                y = random.randint(1, N)
                if y in A:
                    continue
                
                A_temp = set(A)
                A_temp.remove(x)
                if is_4ap_free_after_add(A_temp, y):
                    A_new = set(A_temp)
                    A_new.add(y)
                    new_score = count_3aps(A_new)
                    
                    if new_score > score:
                        A = A_new
                        score = new_score
                        improved = True
                        swap_improved = True
                        break
            
            if not improved:
                break
        
        return A, score

    # Multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            # Base 4 init
            A = set()
            for i in range(256):
                s = ''
                n = i
                while n > 0:
                    s = str(n % 4) + s
                    n //= 4
                if s == '':
                    s = '0'
                if '2' not in s:
                    A.add(i + 1)
        else:
            # Randomized Greedy Init
            A = set()
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            for x in elements:
                if is_4ap_free_after_add(A, x):
                    A.add(x)
        
        A_final, score = local_search(A)
        
        if score > best_score:
            best_score = score
            best_set = A_final

    return sorted(best_set)
