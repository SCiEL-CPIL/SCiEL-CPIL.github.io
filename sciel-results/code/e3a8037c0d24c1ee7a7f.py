import random
import sys
from itertools import combinations, combinations_with_replacement
import heapq
import time

def construct():
    n = 256
    max_size = 64
    random.seed(42)
    
    def check_sidon_add(S, x, sums_set):
        if (x + x) in sums_set:
            return False
        for s in S:
            if (x + s) in sums_set:
                return False
        return True

    def check_diff_add(S, x, other_diffs):
        for s in S:
            d1 = x - s
            d2 = s - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i <= j:
                    sums.add(S[i] + S[j])
        return sums

    def greedy_fill(candidates, A, B, sums_A, sums_B, diffs_A, diffs_B):
        for x in candidates:
            added = False
            if len(A) < max_size:
                if check_sidon_add(A, x, sums_A) and check_diff_add(A, x, diffs_B):
                    A.append(x)
                    sums_A.add(x + x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                    for s in A[:-1]:
                        diffs_A.add(x - s)
                        diffs_A.add(s - x)
                    added = True
            
            if not added and len(B) < max_size:
                if check_sidon_add(B, x, sums_B) and check_diff_add(B, x, diffs_A):
                    B.append(x)
                    sums_B.add(x + x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                    for s in B[:-1]:
                        diffs_B.add(x - s)
                        diffs_B.add(s - x)
                    added = True
        return len(A) + len(B)

    best_total = 0
    best_A = []
    best_B = []
    
    start_time = time.time()
    timeout = 55 # Leave 5s margin
    
    # Phase 1: Randomized Greedy
    for trial in range(300):
        if time.time() - start_time > timeout:
            break
            
        random.seed(trial)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        total = greedy_fill(candidates, A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
    # Phase 2: Local Search with Kicks
    A = best_A[:]
    B = best_B[:]
    sums_A = get_sums(A)
    sums_B = get_sums(B)
    diffs_A = get_diffs(A)
    diffs_B = get_diffs(B)
    
    current_total = len(A) + len(B)
    
    improved = True
    while improved and (time.time() - start_time < timeout):
        improved = False
        
        # Try removing from A
        for i in range(len(A)):
            if time.time() - start_time > timeout:
                break
            elem = A[i]
            A_temp = A[:i] + A[i+1:]
            B_temp = B[:]
            
            sums_A_temp = get_sums(A_temp)
            diffs_A_temp = get_diffs(A_temp)
            sums_B_temp = get_sums(B_temp)
            diffs_B_temp = get_diffs(B_temp)
            
            candidates = [x for x in range(1, n+1) if x not in A_temp and x not in B_temp]
            random.shuffle(candidates)
            
            total_temp = greedy_fill(candidates, A_temp, B_temp, sums_A_temp, sums_B_temp, diffs_A_temp, diffs_B_temp)
            
            if total_temp > current_total:
                A = A_temp
                B = B_temp
                sums_A = sums_A_temp
                sums_B = sums_B_temp
                diffs_A = diffs_A_temp
                diffs_B = diffs_B_temp
                current_total = total_temp
                best_total = current_total
                best_A = A[:]
                best_B = B[:]
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing from B
        for i in range(len(B)):
            if time.time() - start_time > timeout:
                break
            elem = B[i]
            A_temp = A[:]
            B_temp = B[:i] + B[i+1:]
            
            sums_A_temp = get_sums(A_temp)
            diffs_A_temp = get_diffs(A_temp)
            sums_B_temp = get_sums(B_temp)
            diffs_B_temp = get_diffs(B_temp)
            
            candidates = [x for x in range(1, n+1) if x not in A_temp and x not in B_temp]
            random.shuffle(candidates)
            
            total_temp = greedy_fill(candidates, A_temp, B_temp, sums_A_temp, sums_B_temp, diffs_A_temp, diffs_B_temp)
            
            if total_temp > current_total:
                A = A_temp
                B = B_temp
                sums_A = sums_A_temp
                sums_B = sums_B_temp
                diffs_A = diffs_A_temp
                diffs_B = diffs_B_temp
                current_total = total_temp
                best_total = current_total
                best_A = A[:]
                best_B = B[:]
                improved = True
                break

    return [best_A, best_B]
