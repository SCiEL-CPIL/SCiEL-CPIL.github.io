import numpy as np
from scipy.signal import fftconvolve
import time
from math import isqrt
import sys

def construct():
    start_time = time.time()
    # Safety margin: return before 110s
    deadline = start_time + 110.0
    
    # We want to maximize the number of representations.
    # Search space: Powerful numbers up to L.
    # Larger L -> more powerful numbers -> potentially more collisions.
    
    # Try a larger L. 
    # L=10^8 took ~10s.
    # Convolution complexity is O(L log L).
    # 4x L -> ~4x time + overhead.
    # Let's try L=3*10^8.
    
    L_candidates = [300_000_000, 200_000_000, 100_000_000]
    
    best_result = None
    best_score = 0
    
    for L in L_candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            # Generate powerful numbers up to L
            powerful = set()
            # v^3 <= L
            max_v = int(L ** (1.0/3.0)) + 2
            for v in range(1, max_v + 1):
                v3 = v ** 3
                if v3 > L:
                    break
                # u^2 * v^3 <= L => u <= sqrt(L / v^3)
                max_u = isqrt(L // v3)
                for u in range(1, max_u + 1):
                    p = u * u * v3
                    powerful.add(p)
            
            P = sorted(powerful)
            P_set = set(P)
            
            # Create indicator vector
            # Size L+1
            ind = np.zeros(L + 1, dtype=np.float64)
            for p in P:
                ind[p] = 1.0
            
            # Check time before heavy convolution
            if time.time() > deadline - 40:
                # Skip convolution if we are running out of time, 
                # but we need a result. 
                # If we haven't found a best_result yet, we must do something.
                # But if we are in the loop, we might have skipped previous Ls?
                # No, we only break if time is low.
                # If we are here, we haven't computed convolution for this L.
                # Let's just proceed and hope it's fast, or break and use previous.
                # Actually, if this is the first L, we must compute.
                # Let's just proceed.
                pass

            # Compute convolution
            # fftconvolve is faster for large arrays
            # mode='full' gives size 2*L + 1
            try:
                counts = fftconvolve(ind, ind, mode='full')
            except Exception:
                # Fallback to numpy convolve if scipy fails or is slow?
                # Or just use smaller L
                counts = np.convolve(ind, ind, mode='full')
            
            # Find N with max count
            # N must be >= 2
            # counts[i] is the number of pairs (a,b) with a+b=i
            # We want to maximize counts[i]
            
            # Ensure we don't pick index 0 or 1 (counts are 0 anyway)
            # Find argmax
            max_idx = np.argmax(counts)
            N = int(max_idx)
            
            # Extract pairs
            pairs = []
            for a in P:
                if a > N:
                    break
                b = N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            score = len(pairs)
            
            if score > best_score:
                best_score = score
                best_result = {'N': N, 'pairs': pairs}
                
            # If we have a good result and time is running out, break?
            # The larger L gives better results, so we want to try the largest feasible L.
            # But if the largest L fails or times out, we fall back.
            # So we continue to smaller Ls only if the larger one failed or we want to be safe?
            # Actually, if L=3e8 succeeds, it's likely better than L=2e8.
            # So if it succeeds, we can break and return.
            # But we should ensure it's valid.
            
            # Verify N <= 10^18
            if N > 10**18:
                continue
                
            # If we successfully processed a large L, we can return immediately 
            # if we are confident it's the best. 
            # Since L is decreasing, the first successful large L is the best candidate.
            # However, we should check if we have time to verify or if we should just return.
            # The task requires maximizing. Larger L -> more powerful numbers -> higher max sum count.
            # So the first successful L in the list (largest) is the best.
            return best_result

        except MemoryError:
            # Try smaller L
            continue
        except Exception:
            # Try smaller L
            continue

    # If all failed or we broke out of loop, return the best found
    if best_result:
        return best_result
    
    # Fallback: very small N
    return {'N': 2, 'pairs': [[1, 1]]}
