import time

def construct():
        start_time = time.time()
        deadline = start_time + 50  # Safety margin
        
        N = 1000001
        # Sieve for primes
        is_prime = [True] * N
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(N**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, N, i):
                    is_prime[j] = False
        
        # Compute Largest Prime Factor
        lpf = [0] * N
        for p in range(2, N):
            if is_prime[p]:
                for j in range(p, N, p):
                    lpf[j] = p
        
        # Search for min max(lpf[n], lpf[n+1])
        best_n = 10000
        best_val = max(lpf[10000], lpf[100001]) # Wait, lpf[100001] is out of bounds if N=1000001? No, 100001 < 1000001.
        # Actually, n goes up to 1000000. n+1 goes up to 1000001.
        # So we need lpf up to 1000001.
        
        # Let's re-verify N.
        # Max n = 1000000. Max n+1 = 1000001.
        # So array size must be at least 1000002.
        
        # Re-do with correct N
        N = 1000002
        is_prime = [True] * N
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(N**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, N, i):
                    is_prime[j] = False
        
        lpf = [0] * N
        for p in range(2, N):
            if is_prime[p]:
                for j in range(p, N, p):
                    lpf[j] = p
        
        best_n = 10000
        best_val = max(lpf[10000], lpf[10001])
        
        for n in range(10000, 1000000 + 1):
            if time.time() > deadline:
                break
            val = max(lpf[n], lpf[n+1])
            if val < best_val:
                best_val = val
                best_n = n
        
        return best_n
