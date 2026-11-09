import time
import random

def construct():
    deadline = time.time() + 85.0
    N = 512
    MAXC = 48

    def check_3ap_free(S, x):
        """Check that x doesn't form a 3-term AP with any pair in S, and x isn't mean of 2 elements in S."""
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S[i], S[j]
                # Check if x is mean of a,b: 2*x == a+b
                if 2 * x == a + b:
                    return False
                # Check if a is mean of x,b: 2*a == x+b
                if 2 * a == x + b:
                    return False
                # Check if b is mean of x,a: 2*b == x+a
                if 2 * b == x + a:
                    return False
        return True

    def build_dp(S):
        """Build subset-sum DP: dp[k] is a bitmask where bit s is set iff some subset of size k sums to s."""
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def verify_full(S):
        """Verify that no element is the mean of any subset of >=2 other elements."""
        S = list(S)
        n = len(S)
        if n < 3:
            return True
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    def test_add_fast(S, x):
        """Fast check: can x be added to S? Uses 3-AP prefilter + partial DP check."""
        if not check_3ap_free(S, x):
            return False
        # Full check for x not being mean of any subset of S
        n = len(S)
        if n < 2:
            return True
        dp = build_dp(S)
        maxfull = sum(S)
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp[k] >> t) & 1:
                return False
        # Check no a in S becomes mean of subset containing x
        # For each a in S, need: no subset T of (S \ {a}) with |T|>=1 such that sum(T)+x = (|T|+1)*a
        # i.e., sum(T) = (|T|+1)*a - x
        for a in S:
            sub = [s for s in S if s != a]
            m = len(sub)
            if m < 1:
                continue
            dp_sub = build_dp(sub)
            maxo = sum(sub)
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                if (dp_sub[j] >> t) & 1:
                    return False
        return True

    # Phase 1: Greedy with 3-AP filter to get a good starting set
    random.seed(42)
    base = list(range(1, N + 1))
    
    best = []
    best_sz = 0

    # Try multiple orders
    orders = []
    orders.append(base[:])
    orders.append(base[::-1])
    for _ in range(20):
        o = base[:]
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline - 20:
            break
        cur = []
        for x in order:
            if len(cur) >= MAXC:
                break
            if test_add_fast(cur, x):
                cur.append(x)
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
        if best_sz >= MAXC:
            break

    # Verify the greedy result
    if best and not verify_full(best):
        # Trim to valid
        while best and not verify_full(best):
            best.pop()
        best_sz = len(best)

    # Phase 2: Simulated Annealing
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        T = 10.0
        T_min = 0.1
        cool_rate = 0.995
        
        sa_best = S[:]
        sa_best_sz = len(S)
        
        while time.time() < deadline - 3 and T > T_min:
            T *= cool_rate
            
            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.4 and len(S) < MAXC:
                # Add move
                if not unused:
                    continue
                u = random.choice(unused)
                if test_add_fast(S, u):
                    newS = S + [u]
                    new_cost = -len(newS)  # We want to maximize size
                    old_cost = -len(S)
                    delta = new_cost - old_cost  # = -1 (improvement)
                    # Always accept improvements
                    S = newS
                    Sset.add(u)
                    unused.remove(u)
                    if len(S) > sa_best_sz:
                        sa_best_sz = len(S)
                        sa_best = S[:]
                        if sa_best_sz >= MAXC:
                            break
                # If rejected, just skip (don't add invalid)
                
            elif move_type < 0.7 and len(S) > 2:
                # Remove move
                idx = random.randint(0, len(S) - 1)
                u = S[idx]
                newS = S[:idx] + S[idx + 1:]
                # Removing always keeps validity (subset of valid set is valid)
                S = newS
                Sset.remove(u)
                unused.append(u)
                # No size improvement possible from removal
                
            else:
                # Swap move
                if not unused or not S:
                    continue
                idx = random.randint(0, len(S) - 1)
                old_el = S[idx]
                u = random.choice(unused)
                # Check if swap is valid: remove old_el, add u
                newS_without = S[:idx] + S[idx + 1:]
                if test_add_fast(newS_without, u):
                    newS = newS_without + [u]
                    S = newS
                    Sset.remove(old_el)
                    Sset.add(u)
                    unused.remove(u)
                    unused.append(old_el)
                    # Size unchanged, but may enable future additions
                    if len(S) > sa_best_sz:
                        sa_best_sz = len(S)
                        sa_best = S[:]
        
        if sa_best_sz > best_sz:
            best = sa_best[:]
            best_sz = sa_best_sz

    # Final verification
    if best and not verify_full(best):
        # Trim to valid
        while best and not verify_full(best):
            best.pop()

    return best
