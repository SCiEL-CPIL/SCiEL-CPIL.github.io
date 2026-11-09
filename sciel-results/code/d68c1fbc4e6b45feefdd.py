"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    n = 256
    
    def check_fast(selected_list, candidate):
        """Check if adding candidate to selected_list maintains the property."""
        # Case 1: candidate is 'a' (smallest in some triple)
        # For all b, c in selected_list with b > candidate and c >= b: (b+c) % candidate != 0
        for i in range(len(selected_list)):
            b = selected_list[i]
            if b <= candidate:
                continue
            for j in range(i, len(selected_list)):
                c = selected_list[j]
                if c < b:
                    continue
                if (b + c) % candidate == 0:
                    return False
        
        # Case 2: candidate is 'b' or 'c'
        # For each a in selected_list with a < candidate:
        for a in selected_list:
            if a >= candidate:
                continue
            # candidate as b: need c >= candidate, check (candidate + c) % a != 0
            for c in selected_list:
                if c >= candidate:
                    if (candidate + c) % a == 0:
                        return False
            # candidate as c: need b with a < b <= candidate, check (b + candidate) % a != 0
            for b in selected_list:
                if a < b <= candidate:
                    if (b + candidate) % a == 0:
                        return False
        
        return True
    
    def full_check(sel):
        """Full feasibility check."""
        sel_sorted = sorted(sel)
        k = len(sel_sorted)
        for i in range(k):
            a = sel_sorted[i]
            for j in range(i+1, k):
                b = sel_sorted[j]
                for l in range(j, k):
                    c = sel_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def divisor_count(x):
        if x == 1:
            return 1
        count = 0
        i = 1
        while i * i <= x:
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
            i += 1
        return count
    
    div_counts = [divisor_count(i) for i in range(n+1)]
    
    best_result = list(range(171, 257))  # 86 elements
    
    # Strategy 1: Greedy by divisor count (fewer divisors first)
    candidates = list(range(1, n+1))
    candidates.sort(key=lambda x: (div_counts[x], x))
    selected = []
    for num in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if check_fast(selected, num):
            selected.append(num)
    if len(selected) > len(best_result):
        best_result = selected[:]
    
    # Strategy 2: Greedy from large to small
    selected2 = []
    for num in range(n, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        if check_fast(selected2, num):
            selected2.append(num)
    if len(selected2) > len(best_result):
        best_result = selected2[:]
    
    # Strategy 3: Start with upper tail, add smaller
    selected3 = list(range(171, 257))
    for num in range(170, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        if check_fast(selected3, num):
            selected3.append(num)
    if len(selected3) > len(best_result):
        best_result = selected3[:]
    
    # Strategy 4: Tiered by divisor count
    for max_divs in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        if time.time() - start_time > DEADLINE:
            break
        tier1 = [x for x in range(1, n+1) if div_counts[x] <= max_divs]
        tier1.sort(key=lambda x: (div_counts[x], x))
        selected4 = []
        for num in tier1:
            if check_fast(selected4, num):
                selected4.append(num)
        remaining = [x for x in range(1, n+1) if x not in selected4]
        remaining.sort(key=lambda x: (div_counts[x], x))
        for num in remaining:
            if check_fast(selected4, num):
                selected4.append(num)
        if len(selected4) > len(best_result):
            best_result = selected4[:]
    
    # Strategy 5: Local search - swap improvements
    if time.time() - start_time < DEADLINE - 5:
        best_set = set(best_result)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for rem in list(best_set):
                if time.time() - start_time > DEADLINE - 2:
                    break
                temp_set = list(best_set - {rem})
                additions = []
                for add in range(1, n+1):
                    if add in best_set or add == rem:
                        continue
                    if check_fast(temp_set + additions, add):
                        additions.append(add)
                    if len(additions) >= 2:
                        break
                if len(additions) >= 2:
                    best_set = set(temp_set + additions)
                    improved = True
    
    if len(best_set) > len(best_result):
        best_result = list(best_set)
    
    # Final verification
    if not full_check(best_result):
        best_result = list(range(171, 257))
    
    return best_result
# EVOLVE-BLOCK-END
