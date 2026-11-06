"""
Strategy: Constructive Additive Search with Incremental State and Stochastic Local Search.

Instead of starting with [0, N] and deleting elements (which is forced into specific deletion patterns by the greedy repair loop), we start with an empty set and add elements one by one.
We maintain an incremental state of residue class coverage for all relevant primes.
1. Greedy Construction: Add elements that cause the least "damage" (fewest new primes to become fully covered).
2. Stochastic Local Search: Use the remaining time to perform random add/swap operations to improve the set size, using the fast incremental checker.
"""
import time
import random
import sys

N = 5000
T_START = time.time()
DEADLINE_MARGIN = 10.0 # Safety margin

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _primes(limit):
    """Sieve of Eratosthenes: list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]

class AdmissibleState:
    """
    Maintains the state of an admissible tuple construction.
    Tracks which residue classes are covered for each prime p <= max_size.
    A prime p is 'full' if all p residue classes are covered.
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.primes = _primes(max_size)
        # For each prime, a bitmask of covered residues.
        # Also track count of covered residues to quickly know if it's full.
        self.masks = {}
        self.counts = {}
        self.is_full = {}
        
        for p in self.primes:
            self.masks[p] = 0
            self.counts[p] = 0
            self.is_full[p] = False
        
        self.current_size = 0
        self.num_full_primes = 0 # Number of primes p <= current_size that are full
        
    def _check_fullness(self, p):
        """Check if prime p is full based on current mask."""
        # We only care about primes p <= current_size
        if p > self.current_size:
            return False
        return self.counts[p] == p

    def add(self, x):
        """
        Add x to the set. Returns True if the set remains admissible (no prime p <= new_size is full).
        If it becomes inadmissible, it reverts the change and returns False.
        """
        self.current_size += 1
        new_size = self.current_size
        
        # We need to check primes p <= new_size.
        # Adding x affects residues for all primes.
        # But a prime p only becomes 'full' if it was p-1 and we add the missing residue.
        # Actually, we just update the mask and count.
        
        # Optimization: Only update primes p <= new_size? 
        # No, a prime p > new_size cannot be full because we only have new_size elements.
        # So we only need to track primes p <= new_size.
        
        # However, when size increases, a new prime might enter the "checkable" range.
        # If p == new_size, it becomes checkable.
        
        # Let's update all primes p <= new_size.
        # This is O(pi(new_size)) per add. pi(5000) ~ 669.
        # 5000 adds * 669 ~ 3.3M ops. This is fast in Python if optimized, but might be tight.
        # Let's try to be smart.
        
        # We track which primes are currently full.
        # If a prime becomes full, we revert.
        
        # Update masks
        newly_full = []
        for p in self.primes:
            if p > new_size:
                break
            
            r = x % p
            bit = 1 << r
            
            if not (self.masks[p] & bit):
                self.masks[p] |= bit
                self.counts[p] += 1
                
                # Check if it just became full
                if self.counts[p] == p:
                    newly_full.append(p)
            # If bit was already set, count doesn't change, mask doesn't change.
            
        # If any prime became full, it's invalid.
        if newly_full:
            # Revert
            self.current_size -= 1
            for p in newly_full:
                # We only need to revert the specific change if it caused fullness?
                # No, we added x, which might have affected multiple primes.
                # But if we revert, we must undo the mask/count changes for x.
                # It's easier to just not apply the changes in the first place if we can predict?
                # No, we have to apply to know.
                # So we must revert.
                
                # Revert logic:
                # For every prime p <= new_size (which is now old_size+1), if x contributed to the mask, remove it.
                # This is expensive.
                
                # Alternative: Copy state? No, too slow.
                
                # Better: Only update if we are sure? No.
                
                # Let's just revert the specific primes we touched.
                # We touched all primes p <= new_size.
                for p in self.primes:
                    if p > new_size:
                        break
                    r = x % p
                    bit = 1 << r
                    if self.masks[p] & bit:
                        # It was set. Was it set before?
                        # We don't know without storing previous state.
                        # This approach is getting messy.
                        
                        pass
            # Let's use a simpler "try and rollback" by storing the previous state of the affected primes?
            # Or just accept that `add` might be slow if it fails often.
            
            # Actually, a simpler rollback:
            # We know x was added.
            # For each p <= new_size, if (x % p) was the missing bit, it's now full.
            # To rollback, we just need to clear the bit for x in all primes p <= new_size.
            # But we only incremented count if the bit was NOT set.
            # So we only need to decrement count for primes where the bit was NOT set before.
            
            # Let's refactor `add` to be safer.
            pass # Placeholder for logic below
            
            # Refined Add:
            # 1. Calculate residues for all p <= new_size.
            # 2. Check which primes would become full.
            # 3. If any, return False (and don't modify state).
            # 4. If none, apply modifications.
            
            # This requires 2 passes or storing temp vars.
            # Given N=5000, 2 passes over ~670 primes is fine.
            
            # Let's restart the `add` logic in the implementation below.
            pass
            
        return False # Placeholder

    def add_safe(self, x):
        """
        Attempt to add x. Returns True if successful, False otherwise.
        Uses a two-pass approach to avoid rollback overhead.
        """
        self.current_size += 1
        new_size = self.current_size
        
        # Pass 1: Check if adding x makes any prime p <= new_size full.
        for p in self.primes:
            if p > new_size:
                break
            r = x % p
            bit = 1 << r
            if not (self.masks[p] & bit):
                # This bit is new.
                if self.counts[p] + 1 == p:
                    # This prime would become full.
                    self.current_size -= 1
                    return False
        
        # Pass 2: Apply changes.
        for p in self.primes:
            if p > new_size:
                break
            r = x % p
            bit = 1 << r
            if not (self.masks[p] & bit):
                self.masks[p] |= bit
                self.counts[p] += 1
                # It cannot be full because we checked in Pass 1.
        
        return True

    def remove(self, x):
        """
        Remove x from the set. Always succeeds if x is in the set.
        """
        self.current_size -= 1
        old_size = self.current_size + 1 # Size before removal
        
        for p in self.primes:
            if p > old_size:
                break
            r = x % p
            bit = 1 << r
            if self.masks[p] & bit:
                self.masks[p] &= ~bit
                self.counts[p] -= 1

    def can_add(self, x):
        """Check if x can be added without making the set inadmissible."""
        # Temporarily increment size to check primes up to new size
        new_size = self.current_size + 1
        for p in self.primes:
            if p > new_size:
                break
            r = x % p
            bit = 1 << r
            if not (self.masks[p] & bit):
                if self.counts[p] + 1 == p:
                    return False
        return True


def construct():
    # Initialize state for max possible size N+1
    state = AdmissibleState(N + 1)
    current_set = []
    current_set_bool = [False] * (N + 1)
    
    # 1. Greedy Construction
    # We want to add elements.
    # To maximize size, we should add elements that "cost" the least.
    # Cost = number of primes that become "closer to full" or "full" (but full is invalid).
    # Actually, since we can't become full, we just want to add elements that don't restrict future adds too much.
    # A good heuristic: Add elements that cover residues for small primes where we have few options?
    # Or simply: Randomized Greedy.
    # Let's try: Add elements in a random order, but prioritize those that don't "block" many other elements.
    
    # A simpler, robust heuristic:
    # Start with 0.
    # Then, repeatedly pick the element x not in set that minimizes the number of "conflicts" it creates for future steps?
    # This is expensive.
    
    # Alternative:
    # Just add elements in increasing order? [0, 1, 2, ...]
    # Let's test if simple sequential addition works well.
    # If we add 0, 1, 2...
    # p=2: 0%2=0, 1%2=1. Full. So we can't add 1 if 0 is there?
    # Wait, if p=2 and size becomes 2, we need both 0 and 1 mod 2.
    # If we have {0}, size=1. p=2 > 1, not checked.
    # Add 1. Size=2. p=2 <= 2. 0%2=0, 1%2=1. Full. Invalid.
    # So we cannot have {0, 1}.
    # We must skip some.
    
    # So we need a selection strategy.
    # Let's use a greedy "least interference" approach.
    # For each candidate x, calculate how many primes p <= len(set)+1 would have their "freedom" reduced?
    # Or simply: Try to add x. If it fails, skip.
    # But the order matters.
    
    # Let's use a randomized greedy approach with restarts if time permits, or just one good pass.
    # Strategy:
    # 1. Shuffle candidates.
    # 2. Iterate through candidates.
    # 3. If `can_add(x)` is True, add it.
    # This is O(N * pi(N)) which is ~ 5000 * 670 = 3.3M ops. Very fast.
    
    candidates = list(range(N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if state.can_add(x):
            state.add_safe(x)
            current_set.append(x)
            current_set_bool[x] = True
            
        if _time_left() < 5.0:
            break

    best_size = len(current_set)
    
    # 2. Local Improvement
    # Now we have a valid set. Can we improve it?
    # We can try to swap elements or add more if we removed some?
    # Actually, the greedy pass stopped when no more could be added in that order.
    # But a different order might yield a larger set.
    # Or, we can try to "repair" by swapping.
    
    # Let's do a local search:
    # Pick a random element y in set.
    # Pick a random element x not in set.
    # If swapping y with x allows us to add MORE elements later?
    # This is complex.
    
    # Simpler: Try to add any remaining element if we can free up space? No, size is fixed by the constraint.
    # Actually, the size is determined by the set.
    # If we have a set of size K, can we find a set of size K+1?
    # We can try to add an element x not in set. If `can_add(x)` is false, it means some prime p is full.
    # To add x, we must remove an element y from set that shares the residue class mod p that is "critical".
    
    # Let's implement a "Swap to Add" local search.
    # For each x not in set:
    #   If can_add(x) is False:
    #     Find the prime p that is full.
    #     Find an element y in set such that y % p == x % p.
    #     If we remove y, does p become not full? Yes, unless there are other elements with same residue.
    #     But removing y might break admissibility for other primes? No, removing never breaks admissibility (it only reduces coverage).
    #     So, if we remove y, we can add x. Size remains same.
    #     But maybe after swapping, we can add ANOTHER element z?
    
    # This is getting complicated. Let's stick to the greedy construction which is very fast.
    # The greedy construction with random shuffle is likely to find a large set.
    # To improve, we can run multiple random shuffles and keep the best.
    
    best_set = list(current_set)
    best_state = state
    
    # Time check
    while _time_left() > 2.0:
        # Restart greedy with new shuffle
        state = AdmissibleState(N + 1)
        current_set = []
        current_set_bool = [False] * (N + 1)
        
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if state.can_add(x):
                state.add_safe(x)
                current_set.append(x)
                current_set_bool[x] = True
                
        if len(current_set) > len(best_set):
            best_set = current_set
            best_state = state
            
        if _time_left() < 2.0:
            break

    return best_set

# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify admissibility
    def is_admissible(points):
        m = len(points)
        primes = _primes(m)
        for p in primes:
            residues = set(v % p for v in points)
            if len(residues) == p:
                return False
        return True
    print("Admissible:", is_admissible(res))
