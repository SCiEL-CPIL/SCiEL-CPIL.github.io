import time
import sys

def construct():
    """
    Find m in 0..1000000 maximizing f(12, m).
    Uses periodicity LCM(1..12)=27720.
    Precomputes full masks for speed.
    """
    n = 12
    max_L = 144
    period = 27720
    
    # Precompute full masks for each m and i
    # full_masks[m][i] is a bitmask of length max_L
    # Bit j is 1 if (m + 1 + j) % i == 0
    full_masks = [[0] * (n + 1) for _ in range(period)]
    
    for m in range(period):
        for i in range(1, n + 1):
            target = (-(m + 1)) % i
            mask = 0
            # j ranges 0 to max_L - 1
            # j % i == target
            # first j is target
            if target < max_L:
                j = target
                while j < max_L:
                    mask |= (1 << j)
                    j += i
            full_masks[m][i] = mask

    # Helper for matching
    def _can_match(m, L):
        # Get masks for current L
        masks = []
        limit_mask = (1 << L) - 1
        for i in range(1, n + 1):
            msk = full_masks[m][i] & limit_mask
            if msk == 0:
                return False
            # Count bits for sorting
            cnt = bin(msk).count('1')
            masks.append((cnt, msk, i))
        
        # Sort by most constrained first
        masks.sort(key=lambda x: x[0])
        
        # DFS
        def dfs(k, used_mask):
            if k == n:
                return True
            cnt, msk, i = masks[k]
            avail = msk & ~used_mask
            while avail:
                lsb = avail & -avail
                avail ^= lsb
                if dfs(k + 1, used_mask | lsb):
                    return True
            return False
        
        return dfs(0, 0)

    def _f_val(m):
        lo, hi = 1, max_L
        ans = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if _can_match(m, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

    best_m = 0
    best_f = 0
    start_time = time.time()
    
    for m in range(period):
        if time.time() - start_time > 60:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
            
    return int(best_m)

if __name__ == "__main__":
    construct()
