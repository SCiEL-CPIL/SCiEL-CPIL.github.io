"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def score_set(U):
        """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
        if not U or max(U) == 0:
            return 0.0
        U_sorted = sorted(set(U))
        max_U = U_sorted[-1]
        n = len(U_sorted)
        
        # Compute sumset
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(U_sorted[i] + U_sorted[j])
        
        # Compute difference set
        diffset = set()
        for i in range(n):
            for j in range(n):
                diffset.add(U_sorted[i] - U_sorted[j])
        
        s = len(sumset)
        d = len(diffset)
        if s == 0 or d == 0:
            return 0.0
        
        ratio = d / s
        if ratio <= 0:
            return 0.0
        
        log_denom = math.log(2 * max_U + 1)
        if log_denom <= 0:
            return 0.0
        
        exponent = 1.0 + math.log(ratio) / log_denom
        return exponent

    def build_geometric(b, m):
        """Build U = {floor(b^k) for k=0..m}, ensuring 0 is present and values are distinct."""
        U = set()
        current = 1.0
        for k in range(m + 1):
            val = int(round(current))
            if val < 0:
                val = 0
            U.add(val)
            current *= b
        # Ensure 0 is present
        U.add(0)
        return sorted(U)

    def ternary_search_b(b_lo, b_hi, m, time_limit):
        """Ternary search over base b for fixed m."""
        for _ in range(100):
            if time.time() > time_limit:
                break
            m1 = m  # just to avoid confusion
            b1 = b_lo + (b_hi - b_lo) / 3.0
            b2 = b_hi - (b_hi - b_lo) / 3.0
            
            U1 = build_geometric(b1, m)
            U2 = build_geometric(b2, m)
            
            s1 = score_set(U1)
            s2 = score_set(U2)
            
            if s1 < s2:
                b_lo = b1
            else:
                b_hi = b2
        
        b_opt = (b_lo + b_hi) / 2.0
        U_opt = build_geometric(b_opt, m)
        return U_opt, score_set(U_opt)

    # Strategy: sweep over m (number of terms), and for each m, ternary search over b
    # The geometric progression U = {floor(b^0), floor(b^1), ..., floor(b^m)}
    # For the score to be high, we want |U-U| large relative to |U+U|,
    # and max(U) small relative to the ratio.
    
    # For a geometric sequence with base b, the difference set has roughly 2*m*(m+1)/2 + 1 elements
    # and the sumset has roughly m*(m+1)/2 + 1 elements (for distinct powers).
    # Actually for powers of b, U+U has at most (m+1)(m+2)/2 elements and U-U has at most (m+1)^2 elements.
    # The ratio is approximately 2*(m+1)/(m+2) for large m... wait, that's not right.
    # Let me think more carefully.
    # U = {b^0, b^1, ..., b^m}, |U| = m+1
    # U+U: all sums b^i + b^j for 0<=i,j<=m. For b>=2, these are mostly distinct.
    # Number of distinct sums: for i<=j, b^i+b^j is unique. Count = (m+1)(m+2)/2.
    # U-U: all differences b^i - b^j. For i!=j, b^i - b^j. Count of distinct: (m+1)^2 - 1 (excluding 0 counted once) + 1 = (m+1)^2.
    # Actually U-U includes 0 and all b^i - b^j. For b>=2, all these are distinct except when i=j (gives 0).
    # So |U-U| = (m+1)^2 and |U+U| = (m+1)(m+2)/2.
    # Ratio = 2(m+1)/(m+2).
    # max(U) = b^m (approximately, since we floor).
    # Score = 1 + log(2(m+1)/(m+2)) / log(2*b^m + 1)
    # = 1 + log(2(m+1)/(m+2)) / (m * log(2b)) approximately.
    
    # For this to be > 1, we need log(2(m+1)/(m+2)) > 0, i.e., 2(m+1)/(m+2) > 1, i.e., m > 0.
    # The score approaches 1 from above as m increases (since log(2(m+1)/(m+2)) -> 0).
    # But we also need the denominator to not grow too fast.
    
    # Actually, the key insight: for a pure geometric sequence, the ratio |U-U|/|U+U| is bounded by ~2,
    # so log(ratio) is at most log(2). The denominator grows as m*log(b).
    # So the score is 1 + log(2(m+1)/(m+2)) / (m * log(2*b+1)).
    # This is maximized when the denominator is small, i.e., small m and small b.
    
    # But wait - we can also include 0 in the set, which changes things.
    # With 0 in U, U+U = U (since 0+x=x) plus the sums. Actually U+U still includes all a+b.
    # U-U includes all a-b, and with 0, it includes all elements and their negatives.
    
    # Let me just do the search empirically.
    
    best_score = 0.0
    best_set = [0, 1, 2]
    best_set_score = score_set(best_set)
    best_score = best_set_score
    
    # Try various m values and search over b
    # m ranges from 2 to maybe 50
    # b ranges from 1.1 to 10
    
    for m in range(2, 60):
        if time.time() > deadline:
            break
        
        # For small m, b can be larger; for large m, b should be closer to 1
        # Let's try a range of b values
        if m <= 5:
            b_lo, b_hi = 1.01, 5.0
        elif m <= 15:
            b_lo, b_hi = 1.005, 3.0
        elif m <= 30:
            b_lo, b_hi = 1.001, 2.0
        else:
            b_lo, b_hi = 1.0001, 1.5
        
        U_opt, s_opt = ternary_search_b(b_lo, b_hi, m, deadline)
        
        if s_opt > best_score:
            best_score = s_opt
            best_set = U_opt
    
    # Also try some non-geometric structured sets as fallback
    # e.g., {0, 1, 3, 9, 27, ...} (powers of 3)
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        U = [0]
        val = 1
        for k in range(1, 20):
            U.append(val)
            val *= base
            if val > 10**8:
                break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Try mixed-radix style: {0, 1, b1, b1*b2, b1*b2*b3, ...}
    # This is essentially the same as geometric if all bases are equal.
    
    # Try adding small perturbations to the best geometric set
    if time.time() < deadline - 10:
        # Local search around best_set
        current_set = best_set.copy()
        current_score = best_score
        temp = 0.05
        for iteration in range(200):
            if time.time() > deadline:
                break
            if not current_set or max(current_set) == 0:
                break
            
            # Try adding a small perturbation
            max_val = max(current_set)
            if max_val == 0:
                break
            
            # Add a random element near existing ones
            if len(current_set) < 100:
                candidate = random.randint(1, max_val + max_val // 10)
                if candidate not in current_set:
                    new_set = current_set + [candidate]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current_set.copy()
                    temp *= 0.99
            else:
                break
    
    return best_set

# EVOLVE-BLOCK-END
