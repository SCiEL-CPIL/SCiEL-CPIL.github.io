"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    # Helper to check Sidon property for a set
    def is_sidon(aset):
        if not aset:
            return True
        sums = set()
        for i in range(len(aset)):
            for j in range(i, len(aset)):
                s = aset[i] + aset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper to get nonzero differences
    def get_diffs(aset):
        diffs = set()
        for i in range(len(aset)):
            for j in range(len(aset)):
                if i != j:
                    diffs.add(aset[i] - aset[j])
        return diffs

    # Initial greedy construction to get a valid starting point
    def initial_greedy():
        A = []
        B = []
        # We try to build A and B simultaneously but independently initially, 
        # then fix difference conflicts? 
        # Actually, the seed program's logic is:
        # for x in range: try adding to A, then try adding to B.
        # This creates a strong bias towards small numbers.
        
        # Let's do a smarter greedy: 
        # We want to maximize |A|+|B|.
        # Differences of A and B must be disjoint.
        
        # Strategy: Pick A first using a greedy Sidon construction, 
        # then pick B avoiding A's differences.
        
        # But we can interleave.
        
        # Let's just use the seed logic as a baseline start, then improve.
        sets = [[], []]
        for x in range(1, n + 1):
            for side in (0, 1):
                if len(sets[side]) >= max_size:
                    continue
                candidate = sets[side] + [x]
                if not is_sidon(candidate):
                    continue
                diffs = get_diffs(candidate)
                other_diffs = get_diffs(sets[1 - side])
                if not (diffs & other_diffs):
                    sets[side] = candidate
        return sets[0], sets[1]

    A, B = initial_greedy()
    
    # Local Search / Randomized Improvement
    # We will try to add elements that were skipped, or swap elements to free up space/differences.
    # Since we are bounded by 60s, we will do a randomized restart or hill climbing.
    
    best_size = len(A) + len(B)
    best_A, best_B = A, B
    
    # Pre-compute potential candidates not in A or B
    # Actually, A and B can overlap. So candidates are 1..256.
    
    # To speed up, we maintain difference sets
    def current_state(A, B):
        diffA = get_diffs(A)
        diffB = get_diffs(B)
        return diffA, diffB

    diffA, diffB = current_state(A, B)
    
    # Randomized greedy improvement
    # Try to add elements from 1..256 in random order to A or B if possible.
    # If we hit a dead end, we might need to backtrack or restart.
    
    # Given the time limit, a simple randomized greedy with multiple restarts might be better.
    # Or, a single long randomized greedy pass.
    
    # Let's try a randomized greedy from scratch multiple times, keeping the best.
    # The seed program is deterministic. Randomization might find a larger set.
    
    def randomized_greedy(seed):
        random.seed(seed)
        A = []
        B = []
        diffA = set()
        diffB = set()
        
        # Shuffle the candidate pool
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # We can also shuffle the order of sides (0 or 1)
        
        for x in candidates:
            # Try adding to A
            if len(A) < max_size:
                # Check Sidon for A + [x]
                # Optimization: check sums against existing sums
                # For small sizes, brute force is ok.
                # But we are doing this many times.
                
                # Fast Sidon check
                valid_sidon = True
                # Check x+x and x+a against existing sums
                # We don't store sums, just check against elements.
                # Sum s is unique if no a,b in A s.t. a+b=s.
                # With x added, new sums are x+x and x+a for a in A.
                # We need to ensure these are not in existing sums of A, and not equal to each other.
                
                # Let's just use the is_sidon function for simplicity first, 
                # but optimize if it's too slow.
                # Given n=256 and max_size=64, is_sidon is O(k^2) = 4096 ops.
                # 256 candidates * 2 sides * 4096 = ~2M ops per run. 
                # 85s / (2M ops) is plenty for thousands of restarts.
                
                if is_sidon(A + [x]):
                    new_diffs = get_diffs(A + [x])
                    if not (new_diffs & diffB):
                        # Accept
                        A.append(x)
                        diffA = new_diffs
                        continue # Move to next x, don't try B for this x? 
                        # The seed program tries A then B for the same x.
                        # Let's try both.
            
            # Try adding to B
            if len(B) < max_size:
                if is_sidon(B + [x]):
                    new_diffs = get_diffs(B + [x])
                    if not (new_diffs & diffA):
                        # Accept
                        B.append(x)
                        diffB = new_diffs
                        continue
            
            # If neither worked, x is skipped.
        
        return A, B

    # Run multiple randomized greedy passes
    # We have ~85 seconds.
    # Let's estimate time for one pass.
    # If one pass takes ~0.01s, we can do 8000 passes.
    
    # Let's do a timed loop.
    t0 = time.time()
    num_restarts = 0
    
    while time.time() < deadline:
        A, B = randomized_greedy(num_restarts)
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A, best_B = A, B
            # If we hit the theoretical max or close, we might stop early?
            # Max Sidon set in [1,256] is approx sqrt(256)=16? No, Sidon sets can be larger.
            # Actually, max size of Sidon set in [1,N] is ~sqrt(N). 
            # Wait, Sidon set definition: all sums a+b are unique.
            # Max size is roughly sqrt(2N). For N=256, sqrt(512) ~ 22.
            # So 64 is definitely not the max size for a single Sidon set.
            # The constraint "at most 64" is loose.
            # So we are likely hitting the Sidon limit or the difference disjointness limit.
            
        num_restarts += 1
        # If best_size is high, maybe we should break?
        # Let's just run until deadline.

    return [best_A, best_B]

# EVOLVE-BLOCK-END
