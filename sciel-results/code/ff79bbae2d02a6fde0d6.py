"""
Rewritten program combining incremental evaluation with smooth-number biased search.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A = sorted(A)
        n_local = len(A)
        sumset = set()
        for i in range(n_local):
            for j in range(i, n_local):
                sumset.add(A[i] + A[j])
        prodset = set()
        for i in range(n_local):
            for j in range(i, n_local):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))

    def is_smooth(x, max_prime=7):
        """Check if x is a smooth number (prime factors <= max_prime)."""
        for p in [2, 3, 5, 7]:
            while x % p == 0:
                x //= p
        return x == 1

    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs

    # Generate a pool of "good" candidate numbers for moves
    # Focus on smooth numbers and divisors of highly composite numbers
    good_numbers = set()
    # Add small smooth numbers
    for i in range(1, 500):
        if is_smooth(i, 7):
            good_numbers.add(i)
    
    # Add divisors of highly composite numbers
    highly_composite = [720, 2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400, 55440, 83160, 110880]
    for hc in highly_composite:
        divs = get_divisors(hc)
        for d in divs:
            if d <= max_val:
                good_numbers.add(d)
    
    # Add some random smooth numbers up to max_val
    # To keep it fast, we don't scan all 10000, but we already have a good base.
    # Let's add a few more by checking multiples of small smooth numbers
    for base in [2, 3, 5, 7, 11, 13]:
        for k in range(1, max_val // base + 1):
            val = base * k
            if val <= max_val and is_smooth(val, 13):
                good_numbers.add(val)
    
    good_numbers = sorted(list(good_numbers))
    # Ensure 1..20 are included just in case
    for i in range(1, 21):
        good_numbers.append(i)
    good_numbers = sorted(list(set(good_numbers)))

    # --- Strategy 1: Structured Initializations ---
    initial_sets = []
    
    # 1. Sequential
    initial_sets.append(list(range(1, 21)))
    
    # 2. Geometric (Powers of 2)
    initial_sets.append([2**i for i in range(20)])
    
    # 3. Small Smooth Numbers
    smooth_sorted = sorted([x for x in good_numbers if x <= 100])
    if len(smooth_sorted) >= 20:
        initial_sets.append(smooth_sorted[:20])
    
    # 4. Divisors of 2520 (first 20)
    divs_2520 = sorted([d for d in get_divisors(2520) if d <= max_val])
    if len(divs_2520) >= 20:
        initial_sets.append(divs_2520[:20])
        
    # 5. Divisors of 5040 (first 20)
    divs_5040 = sorted([d for d in get_divisors(5040) if d <= max_val])
    if len(divs_5040) >= 20:
        initial_sets.append(divs_5040[:20])

    # 6. Lattice 2^a * 3^b
    lattice = set()
    for a in range(8):
        for b in range(8):
            val = (2**a) * (3**b)
            if val <= max_val:
                lattice.add(val)
    if len(lattice) >= 20:
        initial_sets.append(sorted(lattice)[:20])
        
    # 7. Lattice 2^a * 5^b
    lattice2 = set()
    for a in range(8):
        for b in range(8):
            val = (2**a) * (5**b)
            if val <= max_val:
                lattice2.add(val)
    if len(lattice2) >= 20:
        initial_sets.append(sorted(lattice2)[:20])

    # Evaluate initial sets
    for A in initial_sets:
        if len(A) != 20 or len(set(A)) != 20:
            continue
        if any(x < 1 or x > max_val for x in A):
            continue
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_set = sorted(A)[:]

    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)

    # --- Incremental Evaluation Class ---
    class IncrementalEval:
        def __init__(self, A):
            self.A = sorted(A)
            self.n = len(self.A)
            self.sumset = set()
            self.prodset = set()
            for i in range(self.n):
                for j in range(i, self.n):
                    self.sumset.add(self.A[i] + self.A[j])
                    self.prodset.add(self.A[i] * self.A[j])
            self.sum_counts = {s: 0 for s in self.sumset}
            self.prod_counts = {p: 0 for p in self.prodset}
            for i in range(self.n):
                for j in range(i, self.n):
                    s = self.A[i] + self.A[j]
                    p = self.A[i] * self.A[j]
                    self.sum_counts[s] += 1
                    self.prod_counts[p] += 1
        
        def get_score(self):
            return max(len(self.sumset), len(self.prodset))
        
        def replace_element(self, idx, new_val):
            old_val = self.A[idx]
            if old_val == new_val:
                return
            
            # Remove old
            for i in range(self.n):
                if i == idx: continue
                s = old_val + self.A[i]
                p = old_val * self.A[i]
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    self.sumset.discard(s)
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    self.prodset.discard(p)
                    del self.prod_counts[p]
            
            s_diag = old_val + old_val
            p_diag = old_val * old_val
            self.sum_counts[s_diag] -= 1
            if self.sum_counts[s_diag] == 0:
                self.sumset.discard(s_diag)
                del self.sum_counts[s_diag]
            self.prod_counts[p_diag] -= 1
            if self.prod_counts[p_diag] == 0:
                self.prodset.discard(p_diag)
                del self.prod_counts[p_diag]
            
            # Add new
            self.A[idx] = new_val
            self.A.sort()
            idx = self.A.index(new_val)
            
            for i in range(self.n):
                if i == idx: continue
                s = new_val + self.A[i]
                p = new_val * self.A[i]
                if s not in self.sum_counts:
                    self.sumset.add(s)
                    self.sum_counts[s] = 0
                self.sum_counts[s] += 1
                if p not in self.prod_counts:
                    self.prodset.add(p)
                    self.prod_counts[p] = 0
                self.prod_counts[p] += 1
            
            s_diag = new_val + new_val
            p_diag = new_val * new_val
            if s_diag not in self.sum_counts:
                self.sumset.add(s_diag)
                self.sum_counts[s_diag] = 0
            self.sum_counts[s_diag] += 1
            if p_diag not in self.prod_counts:
                self.prodset.add(p_diag)
                self.prod_counts[p_diag] = 0
            self.prod_counts[p_diag] += 1

    # --- Search Phase ---
    random.seed(42)
    
    # We will run a local search from the best initial set, and potentially restart if stuck
    # But given the time limit, a single deep search from the best start with smart moves is better.
    
    current_A = best_set[:]
    current_score = best_score
    inc_eval = IncrementalEval(current_A)
    
    # Pre-compute indices of good numbers for fast lookup? 
    # Actually, we just need to pick random good numbers.
    
    # Helper to get a random good number near a value
    # Since good_numbers is sorted, we can use bisect to find neighbors
    import bisect
    
    def get_random_good_neighbor(val, spread=100):
        """Get a random good number within [val-spread, val+spread] or random global good number."""
        if random.random() < 0.5:
            # Local neighborhood in good_numbers
            left = bisect.bisect_left(good_numbers, val - spread)
            right = bisect.bisect_right(good_numbers, val + spread)
            if right > left:
                return good_numbers[random.randint(left, right - 1)]
            else:
                return random.choice(good_numbers)
        else:
            # Global random from good numbers
            return random.choice(good_numbers)

    temperature = 10.0
    cooling = 0.99995
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Move generation
        i = random.randint(0, 19)
        old_val = current_A[i]
        
        # Choose new value
        if random.random() < 0.7:
            # Biased towards smooth/structured numbers
            new_val = get_random_good_neighbor(old_val, spread=50)
        else:
            # Random small nudge to explore local space
            nudge = random.randint(-5, 5)
            new_val = old_val + nudge
        
        if new_val < 1 or new_val > max_val:
            continue
        if new_val in current_A:
            continue
            
        inc_eval.replace_element(i, new_val)
        new_score = inc_eval.get_score()
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or random.random() < (1.0 / (1.0 + 0.5 * abs(delta))) * min(1.0, temperature / 5.0):
            current_score = new_score
            current_A = inc_eval.A[:]
            if new_score < best_score:
                best_score = new_score
                best_set = current_A[:]
        else:
            inc_eval.replace_element(i, old_val)
            current_A = inc_eval.A[:]
            
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    # --- Final Polish: Exhaustive local search ---
    if time.time() < deadline - 5:
        polish_A = best_set[:]
        polish_score = best_score
        polish_eval = IncrementalEval(polish_A)
        
        improved = True
        while improved:
            improved = False
            for i in range(20):
                old_val = polish_A[i]
                # Try small nudges
                for nudge in range(-10, 11):
                    if nudge == 0: continue
                    new_val = old_val + nudge
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in polish_A:
                        continue
                    
                    polish_eval.replace_element(i, new_val)
                    new_score = polish_eval.get_score()
                    
                    if new_score < polish_score:
                        polish_score = new_score
                        polish_A = polish_eval.A[:]
                        improved = True
                        break # Restart inner loop as array changed
                    else:
                        polish_eval.replace_element(i, old_val)
                        polish_A = polish_eval.A[:]
                if improved:
                    break # Restart outer loop
            
            if polish_score < best_score:
                best_score = polish_score
                best_set = polish_A[:]

    return best_set
# EVOLVE-BLOCK-END
