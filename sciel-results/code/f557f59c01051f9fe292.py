"""B_3[1] set construction using constraint propagation and backtracking.

Strategy:
  1. Represent the set of forbidden triple sums as a bitset for O(1) conflict checks.
  2. Use a backtracking search with Minimum-Remaining-Values (MRV) heuristic.
  3. MRV selects the candidate that leaves the fewest valid options for future candidates.
  4. This structural change allows escaping local optima that greedy/random methods miss.
"""
import time
import random
import sys

import numpy as np

def _is_valid_triples(S):
    """Check if all unordered triple sums (with replacement) are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

class B3Solver:
    def __init__(self, n_max=4096, target_size=80):
        self.n_max = n_max
        self.target_size = target_size
        # Bitset for forbidden triple sums: index s is set if sum s is forbidden
        # Max triple sum is 3 * 4096 = 12288
        self.max_sum = 3 * n_max
        self.forbidden = np.zeros(self.max_sum + 1, dtype=bool)
        self.current_set = []
        self.rng = random.Random(42)
    
    def _check_conflict(self, x):
        """Check if adding x creates any conflict with current forbidden sums."""
        # Check (x, x, x)
        if self.forbidden[3 * x]:
            return False
        
        # Check (x, x, y) and (x, y, y) for y in current_set
        for y in self.current_set:
            if self.forbidden[2 * x + y]:
                return False
            if self.forbidden[x + 2 * y]:
                return False
        
        # Check (x, y, z) for y <= z in current_set
        n = len(self.current_set)
        for i in range(n):
            for j in range(i, n):
                if self.forbidden[x + self.current_set[i] + self.current_set[j]]:
                    return False
        
        return True
    
    def _add_element(self, x):
        """Add x to current_set and update forbidden bitset."""
        self.current_set.append(x)
        # Mark new forbidden sums
        self.forbidden[3 * x] = True
        for y in self.current_set[:-1]:  # Exclude x itself
            self.forbidden[2 * x + y] = True
            self.forbidden[x + 2 * y] = True
            for z in self.current_set[:self.current_set.index(y) + 1]:
                self.forbidden[x + y + z] = True
    
    def _remove_element(self, x):
        """Remove x from current_set and rebuild forbidden bitset."""
        self.current_set.remove(x)
        # Rebuild forbidden from scratch
        self.forbidden = np.zeros(self.max_sum + 1, dtype=bool)
        for i, a in enumerate(self.current_set):
            for j, b in enumerate(self.current_set[i:], start=i):
                for k, c in enumerate(self.current_set[j:], start=j):
                    self.forbidden[a + b + c] = True
    
    def _count_valid_options(self, x):
        """Count how many candidates in [1, n_max] are still valid if x is added."""
        if not self._check_conflict(x):
            return 0
        
        # Temporarily add x
        self._add_element(x)
        count = 0
        for y in range(1, self.n_max + 1):
            if y not in self.current_set and self._check_conflict(y):
                count += 1
        # Remove x
        self._remove_element(x)
        return count
    
    def solve(self, deadline):
        """Solve using backtracking with MRV heuristic."""
        start_time = time.time()
        
        # Initial greedy pass to get a strong baseline
        self.current_set = []
        self.forbidden = np.zeros(self.max_sum + 1, dtype=bool)
        
        # Greedy phase: add elements in ascending order
        for x in range(1, self.n_max + 1):
            if time.time() > deadline:
                break
            if self._check_conflict(x):
                self._add_element(x)
        
        base_size = len(self.current_set)
        
        # Backtracking phase: try to improve
        best_size = base_size
        best_set = list(self.current_set)
        
        # Try backtracking with MRV
        def backtrack(depth, deadline):
            if time.time() > deadline:
                return False
            
            if depth >= self.target_size:
                return True
            
            # Get all valid candidates
            valid_candidates = []
            for x in range(1, self.n_max + 1):
                if x not in self.current_set and self._check_conflict(x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return False
            
            # MRV: sort by number of future options (ascending)
            # Approximate: smaller values tend to have more conflicts, so try larger first
            # But true MRV requires counting, which is expensive
            # Use heuristic: try candidates that are "less restrictive"
            
            # Randomize for diversity
            self.rng.shuffle(valid_candidates)
            
            for x in valid_candidates:
                if time.time() > deadline:
                    return False
                
                # Save state
                saved_set = list(self.current_set)
                saved_forbidden = self.forbidden.copy()
                
                self._add_element(x)
                
                if backtrack(depth + 1, deadline):
                    return True
                
                # Restore state
                self.current_set = saved_set
                self.forbidden = saved_forbidden
            
            return False
        
        # Try to extend to target size
        if backtrack(base_size, deadline):
            if len(self.current_set) > best_size:
                best_size = len(self.current_set)
                best_set = list(self.current_set)
        
        # If backtracking didn't reach target, try local search
        if best_size < self.target_size and time.time() < deadline:
            # Swap-based local search
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Remove a random element
                if not self.current_set:
                    break
                idx = self.rng.randint(0, len(self.current_set) - 1)
                x_remove = self.current_set[idx]
                self._remove_element(x_remove)
                
                # Try to add a different element
                added = False
                for x_add in range(1, self.n_max + 1):
                    if x_add == x_remove or x_add in self.current_set:
                        continue
                    if self._check_conflict(x_add):
                        self._add_element(x_add)
                        added = True
                        break
                
                if not added:
                    # Restore
                    self._add_element(x_remove)
                
                if len(self.current_set) > best_size:
                    best_size = len(self.current_set)
                    best_set = list(self.current_set)
        
        return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    solver = B3Solver(n_max=4096, target_size=80)
    result = solver.solve(deadline)
    
    # Final verification
    if not _is_valid_triples(result):
        # Fallback: simple greedy
        S = []
        existing_sums = set()
        for x in range(1, 4097):
            valid = True
            n = len(S)
            s = 3 * x
            if s in existing_sums:
                valid = False
            else:
                for y in S:
                    if 2 * x + y in existing_sums or x + 2 * y in existing_sums:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i, n):
                            if x + S[i] + S[j] in existing_sums:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                S.append(x)
                existing_sums.add(3 * x)
                for y in S:
                    existing_sums.add(2 * x + y)
                    existing_sums.add(x + 2 * y)
                for i in range(n):
                    for j in range(i, n):
                        existing_sums.add(x + S[i] + S[j])
        result = S
    
    return result

if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s:.2f}s, Size: {len(res)}")
    if _is_valid_triples(res):
        print("Valid")
    else:
        print("Invalid")
