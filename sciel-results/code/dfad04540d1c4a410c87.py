import time
from math import isqrt
import numpy as np
from scipy.signal import fftconvolve
from collections import defaultdict

def construct():
    start_time = time.time()
    # Safety margin: return before 140s (evaluator kills at 150s)
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

    # We want to maximize the number of representations.
    # Search space: Powerful numbers up to L.
    # Larger L -> more powerful numbers -> potentially more collisions.

    # Strategy: Try larger L values first since we have more time budget
    # L=10^8 took ~10s, L=3*10^8 should be ~30-40s
    # With 140s budget, we can try up to 5*10^8
    L_candidates = [500_000_000, 300_000_000, 200_000_000, 100_000_000]

    best_result = None
    best_score = 0

    for L in L_candidates:
        if remaining() < 20:
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
            if remaining() < 30:
                # Not enough time for convolution, skip this L
                continue

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

            # If we successfully processed a large L, return immediately
            # Since L is decreasing, first successful large L gives best result
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
