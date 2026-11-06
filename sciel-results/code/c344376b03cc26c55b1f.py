import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form u^2 * v^3 where v is square-free.
    We generate all u^2 * v^3 and use a set to handle duplicates.
    """
    powerful = set()
    max_v = int(limit ** (1.0 / 3.0)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v * v * v
        if v3 > limit:
            break
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    # Try different M values
    # With sparse pair-sum counting, we can handle larger M than FFT
    # K powerful numbers -> O(K^2) operations but vectorized
    # For M=500M, K~48k, K^2 ~ 2.3e9 which is too much for pure Python
    # But with numpy chunking, we can process in chunks
    
    limits_to_try = [500_000_000, 300_000_000, 200_000_000, 100_000_000]
    
    for M in limits_to_try:
        try:
            if time.time() > deadline - 60:
                break
                
            gen_start = time.time()
            powerful_list = np.array(_generate_powerful_numbers(M), dtype=np.int64)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 30:
                continue
                
            K = len(powerful_list)
            if K == 0:
                continue
            
            # Sparse pair-sum counting using chunked numpy
            # For each chunk of a-values, compute all a+b sums and count them
            # Use a dictionary-like approach with numpy bincount on sums
            
            # We want to find N that maximizes the number of pairs (a,b) with a<=b, a+b=N
            # Strategy: compute all pair sums, then find the most common sum
            
            # To avoid O(K^2) memory, process in chunks
            # For chunk i: a_values = powerful_list[start:end]
            # sums = a_values[:, None] + powerful_list[None, :]
            # But this is still O(chunk_size * K) memory
            
            # Better: for each a in powerful_list, count how many b >= a satisfy a+b = N
            # But we don't know N in advance
            
            # Alternative: use the fact that we just need to find N with max count
            # Process in chunks: for each chunk of a-values, compute sums with all b-values
            # where b >= a, then use np.bincount to count occurrences of each sum
            
            # However, np.bincount requires non-negative integers and the sum can be up to 2*M
            # For M=500M, 2*M = 1e9, so bincount array would be 1e9 * 8 bytes = 8GB - too much
            
            # Instead, let's use a different approach:
            # For smaller M where 2*M is manageable, use bincount
            # For larger M, use a sampling/heuristic approach
            
            if M <= 50_000_000:
                # Use bincount for smaller M
                max_sum = 2 * M + 1
                if max_sum * 8 < 2_000_000_000:  # 2GB limit
                    # Process in chunks to avoid O(K^2) memory
                    chunk_size = max(1, min(K, 1000))
                    sum_counts = np.zeros(max_sum, dtype=np.int32)
                    
                    for start in range(0, K, chunk_size):
                        end = min(start + chunk_size, K)
                        a_chunk = powerful_list[start:end]
                        
                        # For each a in a_chunk, find b values where b >= a
                        # sums = a_chunk[:, None] + powerful_list[None, :]
                        # This creates a (chunk_size, K) array
                        sums = a_chunk[:, None] + powerful_list[None, :]
                        
                        # Only count pairs where a <= b (i.e., sums >= 2*a)
                        # Actually, we want a <= b, so b >= a
                        # For each row i (a = a_chunk[i]), count columns j where powerful_list[j] >= a_chunk[i]
                        # This is equivalent to: for each a, count b >= a such that a+b = N
                        # But we're counting all sums, then we'll filter by a<=b later
                        
                        # Actually, let's just count all ordered pairs first, then divide by 2 (approximately)
                        # Or better: only count b >= a
                        for i, a_val in enumerate(a_chunk):
                            # Find indices where powerful_list >= a_val
                            idx = np.searchsorted(powerful_list, a_val)
                            if idx < K:
                                b_vals = powerful_list[idx:]
                                row_sums = a_val + b_vals
                                # Count these sums
                                valid = row_sums < max_sum
                                if np.any(valid):
                                    sum_counts[row_sums[valid]] += 1
                    
                    # Find the N with max count
                    # Note: this counts ordered pairs where a <= b, so it's already unordered
                    top_k = 50
                    if max_sum > top_k:
                        top_indices = np.argpartition(sum_counts, -top_k)[-top_k:]
                    else:
                        top_indices = np.arange(max_sum)
                    top_indices = top_indices[np.argsort(sum_counts[top_indices])[::-1]]
                    
                    powerful_set = set(powerful_list.tolist())
                    
                    best_count = 0
                    best_N = 2
                    best_pairs = [[1, 1]]
                    
                    for idx in top_indices:
                        if time.time() > deadline - 5:
                            break
                        
                        N_cand = int(idx)
                        if N_cand < 2:
                            continue
                        
                        count = 0
                        pairs = []
                        half_N = N_cand // 2
                        for a in powerful_list:
                            if a > half_N:
                                break
                            b = N_cand - a
                            if b in powerful_set:
                                count += 1
                                pairs.append([int(a), int(b)])
                        
                        if count > best_count:
                            best_count = count
                            best_N = N_cand
                            best_pairs = pairs
                        
                        if sum_counts[idx] < best_count:
                            break
                    
                    if best_count > len(best_result['pairs']):
                        best_result = {'N': best_N, 'pairs': best_pairs}
                    
                    return best_result
                else:
                    continue
            else:
                # For larger M, use a heuristic sampling approach
                # Sample random pairs and count sums, then verify top candidates
                if time.time() > deadline - 10:
                    continue
                
                # Generate a large number of random pairs
                n_samples = 500_000
                idx_a = np.random.randint(0, K, size=n_samples)
                idx_b = np.random.randint(0, K, size=n_samples)
                
                # Ensure a <= b
                mask = idx_a > idx_b
                idx_a[mask], idx_b[mask] = idx_b[mask], idx_a[mask]
                
                a_vals = powerful_list[idx_a]
                b_vals = powerful_list[idx_b]
                sums = a_vals + b_vals
                
                # Count occurrences of each sum
                # Use bincount if max sum is reasonable
                max_sum_sampled = int(sums.max()) + 1
                if max_sum_sampled < 1_000_000_000:
                    sum_counts_sampled = np.bincount(sums, minlength=max_sum_sampled)
                    
                    top_k = 100
                    if max_sum_sampled > top_k:
                        top_indices = np.argpartition(sum_counts_sampled, -top_k)[-top_k:]
                    else:
                        top_indices = np.arange(max_sum_sampled)
                    top_indices = top_indices[np.argsort(sum_counts_sampled[top_indices])[::-1]]
                    
                    powerful_set = set(powerful_list.tolist())
                    
                    best_count = 0
                    best_N = 2
                    best_pairs = [[1, 1]]
                    
                    for idx in top_indices:
                        if time.time() > deadline - 5:
                            break
                        
                        N_cand = int(idx)
                        if N_cand < 2:
                            continue
                        
                        count = 0
                        pairs = []
                        half_N = N_cand // 2
                        for a in powerful_list:
                            if a > half_N:
                                break
                            b = N_cand - a
                            if b in powerful_set:
                                count += 1
                                pairs.append([int(a), int(b)])
                        
                        if count > best_count:
                            best_count = count
                            best_N = N_cand
                            best_pairs = pairs
                        
                        # Early termination heuristic
                        if sum_counts_sampled[idx] * (K / n_samples) < best_count:
                            break
                    
                    if best_count > len(best_result['pairs']):
                        best_result = {'N': best_N, 'pairs': best_pairs}
                    
                    return best_result
                else:
                    continue
                    
        except MemoryError:
            continue
        except Exception as e:
            continue

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
