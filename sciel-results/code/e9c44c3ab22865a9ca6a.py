import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    def is_pairwise_coprime(nums):
        if len(nums) < 2:
            return True
        nums = sorted(nums)
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)
    
    # Current best
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)  # should be 5
    
    # Strategy: Search for A and B such that all 9 sums are distinct and pairwise coprime
    # Key: if all 9 sums are primes, they're automatically pairwise coprime
    # So we want a_i + b_j to all be distinct primes
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    
    # Phase 1: Quick targeted search with small values
    # Try to find 9 distinct prime sums
    found = False
    
    # Try A from small odd numbers, B from small numbers
    # If A has all odd and B has all even (or vice versa), sums are all odd
    # Odd numbers > 2 can be prime
    
    for a1 in range(1, 30, 2):  # odd
        if time.time() > deadline:
            break
        for a2 in range(a1 + 2, 30, 2):  # odd, at least 2 apart
            if time.time() > deadline:
                break
            for a3 in range(a2 + 2, 30, 2):  # odd
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                for b1 in range(2, 30, 2):  # even
                    if time.time() > deadline:
                        break
                    for b2 in range(b1 + 2, 30, 2):  # even
                        if time.time() > deadline:
                            break
                        for b3 in range(b2 + 2, 30, 2):  # even
                            if time.time() > deadline:
                                break
                            B = [b1, b2, b3]
                            
                            # Check if all 9 sums are distinct primes
                            sums = []
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if s in sums or not is_prime[s]:
                                        valid = False
                                        break
                                    sums.append(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                best_A = A
                                best_B = B
                                best_score = 9
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    
    if found:
        return [best_A, best_B]
    
    # Phase 2: Broader search - allow non-prime but pairwise coprime sums
    # Use a smarter evaluation: check pairwise coprime incrementally
    
    def check_coprime_incremental(existing_sums, new_sum):
        """Check if new_sum is coprime to all existing sums"""
        for s in existing_sums:
            if math.gcd(s, new_sum) != 1:
                return False
        return True
    
    # Try various A patterns
    for a1 in range(1, 50):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                # For this A, try to find best B
                best_B_for_A = None
                best_score_for_A = best_score
                
                for b1 in range(1, 50):
                    if time.time() > deadline:
                        break
                    # Precompute sums with b1
                    s1 = [a + b1 for a in A]
                    
                    for b2 in range(b1 + 1, 50):
                        if time.time() > deadline:
                            break
                        s2 = [a + b2 for a in A]
                        
                        # Check if s1 and s2 are pairwise coprime
                        valid = True
                        all_sums = []
                        for s in s1:
                            if s in all_sums:
                                valid = False
                                break
                            all_sums.append(s)
                        if valid:
                            for s in s2:
                                if s in all_sums:
                                    valid = False
                                    break
                                if not check_coprime_incremental(all_sums, s):
                                    valid = False
                                    break
                                all_sums.append(s)
                        
                        if not valid:
                            continue
                        
                        for b3 in range(b2 + 1, 50):
                            if time.time() > deadline:
                                break
                            s3 = [a + b3 for a in A]
                            
                            # Check if s3 is coprime to all existing
                            final_sums = list(all_sums)
                            valid3 = True
                            for s in s3:
                                if s in final_sums:
                                    valid3 = False
                                    break
                                if not check_coprime_incremental(final_sums, s):
                                    valid3 = False
                                    break
                                final_sums.append(s)
                            
                            if valid3:
                                score = len(final_sums)
                                if score > best_score_for_A:
                                    best_score_for_A = score
                                    best_B_for_A = [b1, b2, b3]
                                if best_score_for_A >= 9:
                                    break
                        if best_score_for_A >= 9:
                            break
                    if best_score_for_A >= 9:
                        break
                
                if best_score_for_A > best_score:
                    best_score = best_score_for_A
                    best_A = A
                    best_B = best_B_for_A
                if best_score >= 9:
                    break
            if best_score >= 9:
                break
        if best_score >= 9:
            break
    
    if best_score >= 9:
        return [best_A, best_B]
    
    # Phase 3: Try even larger range if time permits
    if time.time() < deadline - 10:
        for a1 in range(1, 80):
            if time.time() > deadline:
                break
            for a2 in range(a1 + 1, 80):
                if time.time() > deadline:
                    break
                for a3 in range(a2 + 1, 80):
                    if time.time() > deadline:
                        break
                    A = [a1, a2, a3]
                    
                    best_B_for_A = None
                    best_score_for_A = best_score
                    
                    for b1 in range(1, 80):
                        if time.time() > deadline:
                            break
                        s1 = [a + b1 for a in A]
                        
                        for b2 in range(b1 + 1, 80):
                            if time.time() > deadline:
                                break
                            s2 = [a + b2 for a in A]
                            
                            valid = True
                            all_sums = []
                            for s in s1:
                                if s in all_sums:
                                    valid = False
                                    break
                                all_sums.append(s)
                            if valid:
                                for s in s2:
                                    if s in all_sums:
                                        valid = False
                                        break
                                    if not check_coprime_incremental(all_sums, s):
                                        valid = False
                                        break
                                    all_sums.append(s)
                            
                            if not valid:
                                continue
                            
                            for b3 in range(b2 + 1, 80):
                                if time.time() > deadline:
                                    break
                                s3 = [a + b3 for a in A]
                                
                                final_sums = list(all_sums)
                                valid3 = True
                                for s in s3:
                                    if s in final_sums:
                                        valid3 = False
                                        break
                                    if not check_coprime_incremental(final_sums, s):
                                        valid3 = False
                                        break
                                    final_sums.append(s)
                                
                                if valid3:
                                    score = len(final_sums)
                                    if score > best_score_for_A:
                                        best_score_for_A = score
                                        best_B_for_A = [b1, b2, b3]
                                    if best_score_for_A >= 9:
                                        break
                            if best_score_for_A >= 9:
                                break
                        if best_score_for_A >= 9:
                            break
                    
                    if best_score_for_A > best_score:
                        best_score = best_score_for_A
                        best_A = A
                        best_B = best_B_for_A
                    if best_score >= 9:
                        break
                if best_score >= 9:
                    break
            if best_score >= 9:
                break
    
    return [list(best_A), list(best_B)]
