import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    # Fixed seed for reproducibility if needed, but random is fine
    # Using a random seed based on time for variety if evaluated multiple times
    # However, for deterministic evaluation, fixed seed is safer to ensure consistent score
    random.seed(42)
    
    def greedy_with_randomness():
        """
        Performs a single greedy pass with randomized choices.
        At each step, it considers all valid candidates and picks one randomly,
        or with some probability skips the best candidate to explore alternatives.
        """
        seq = []
        prods = set()
        
        # We want to build a sequence of numbers in [2, 128]
        # The sequence must be strictly increasing.
        
        # To make it efficient, we iterate through potential next numbers.
        # But standard greedy picks the smallest valid one.
        # Randomized greedy: 
        # 1. Maintain a set of available numbers (initially 2..128).
        # 2. Find all valid next numbers.
        # 3. Pick one randomly from a subset (e.g., top 5 smallest valid, or all valid with some bias).
        
        # A simpler and often effective variant:
        # Iterate x from 2 to 128.
        # Decide whether to include x based on a probability that might depend on 
        # how many numbers are left or just a fixed probability.
        # However, "strictly increasing" means if we skip x, we can never go back to it.
        # So we must decide on the fly.
        
        # Let's try: 
        # For x in range(2, 129):
        #   If x is valid to add to current seq:
        #     With prob p, add it.
        #     Else, skip it (and thus lose it forever).
        #   Else:
        #     Skip it.
        
        # Tuning p is important. A low p keeps the sequence sparse but might allow larger numbers?
        # Actually, larger numbers create larger products, which are less likely to collide.
        # So skipping small numbers (which have many small factors and create small products)
        # might be beneficial.
        
        # Let's try a strategy where we bias towards larger numbers by skipping small ones
        # with higher probability.
        
        for x in range(2, 129):
            # Check if x can be added
            if not seq:
                # First element is always valid
                # But we might skip it to start with a larger number?
                # Starting with a larger number reduces the range of subsequent numbers.
                # Usually starting small is better for length, unless small numbers are "bad".
                # Let's just decide based on probability.
                pass
            
            # Calculate new suffix products if we were to add x
            # To be fast, we only check validity if we are considering adding it.
            
            # Heuristic: 
            # Probability of skipping x.
            # If x is small, maybe skip with higher prob?
            # If x is large, skip with lower prob?
            
            # Let's use a simple random skip probability.
            # p_skip = 0.1? 0.3?
            # Let's vary it in the outer loop.
            
            # For this specific function instance, let's use a fixed p_skip passed in?
            # No, let's make it a local variable randomized per run.
            
            # Actually, let's look at the "inspiration": 
            # "Randomized greedy exploration escapes the local optimum... avoiding early inclusion of 'conflict-heavy' composites"
            
            # Conflict-heavy composites: numbers with many small prime factors.
            # e.g., 12 = 2^2 * 3. Products involving 12 will be multiples of 12.
            # Primes are "cleaner".
            
            # Let's define a "score" for each number x.
            # Numbers with fewer prime factors are better?
            # Or just randomize the order of consideration?
            
            # Alternative Randomized Strategy:
            # 1. Generate a random permutation of [2, 128]? No, must be increasing.
            # 2. So we must process in increasing order.
            
            # Let's try: 
            # At each step x, if valid, add it with probability p.
            # Let's try p = 0.8, 0.9, etc.
            
            # To make this function self-contained for one run:
            p_add = 0.7 # Base probability
            # Maybe adjust p_add based on x?
            # If x < 10, p_add = 0.5?
            
            # Let's just use a simple random check.
            if random.random() < p_add:
                # Check validity
                # We need to check if adding x creates duplicate products.
                # Suffix products of (seq + [x]) are:
                # x, x*seq[-1], x*seq[-1]*seq[-2], ...
                
                # Optimization: 
                # If we maintain the list of suffix products of the current sequence,
                # we can update them quickly?
                # Current suffix products: s_1, s_2, ..., s_k where s_i is product of last i elements.
                # New suffix products: x, x*s_1, x*s_2, ...
                
                # We can maintain a list `suffix_prods` where suffix_prods[i] is product of last i+1 elements.
                # Initially empty.
                # When adding x:
                # new_suffix_prods = [x] + [x * p for p in suffix_prods]
                # Check if any element in new_suffix_prods is in `prods`.
                # If not, add all to `prods` and update `suffix_prods`.
                
                # This is O(N) per check, where N is current length.
                # Total O(N^2) per greedy pass. N is max ~100. 10000 ops. Very fast.
                
                # Let's implement this efficiently.
                
                # We need to store suffix_prods in the outer scope of this inner function?
                # No, we are in the loop. Let's keep `suffix_prods` in the function scope.
                
                # Wait, `suffix_prods` needs to persist across iterations of the for loop.
                # So define it before the loop.
                
                # Re-structuring:
                # I will move the logic to use `suffix_prods`.
                
                # Let's restart the inner logic with this optimization.
                
                # Actually, I can just write the check inline.
                
                # To avoid code duplication, let's assume `suffix_prods` is maintained.
                
                # Let's refactor the function to use `suffix_prods`.
                
                # I'll rewrite the whole `greedy_with_randomness` to be clean.
                
                pass # Placeholder, I will rewrite the function below.
            
            # This manual loop is getting messy. Let's define the helper cleanly.
            
        return seq

    # Let's define a cleaner version of the greedy pass.
    def run_greedy(p_add_base):
        """
        Runs a single greedy pass.
        p_add_base: Base probability of adding a valid number.
        """
        seq = []
        prods = set()
        suffix_prods = [] # suffix_prods[i] = product of last i+1 elements in seq
        
        for x in range(2, 129):
            # Decide if we want to try adding x
            # We can bias against small numbers if we want, but let's keep it simple first.
            # Maybe: if x is small, we are more likely to skip it?
            # Let's just use p_add_base.
            
            if random.random() < p_add_base:
                # Check validity
                # New suffix products would be:
                # new_sp = [x] + [x * p for p in suffix_prods]
                
                # Check for collision
                valid = True
                new_sp = [x]
                # Optimization: if x is in prods, invalid
                if x in prods:
                    valid = False
                else:
                    for p in suffix_prods:
                        np = x * p
                        if np in prods:
                            valid = False
                            break
                        new_sp.append(np)
                
                if valid:
                    # Add to state
                    seq.append(x)
                    prods.update(new_sp)
                    suffix_prods = new_sp # Update suffix products for next iteration
            
        return seq

    # Main search loop
    # Try different probabilities
    # Also, maybe try some runs with p_add = 1.0 (deterministic greedy) to establish baseline?
    # The deterministic greedy is the "seed" behavior (score 1.0).
    # We want to beat it.
    
    # Let's run many iterations.
    # Each iteration is O(128 * L) ~ O(128 * 100) = 12800 ops.
    # 10,000 iterations = 128,000,000 ops. In Python, this might be 10-20 seconds.
    # Let's aim for 5,000 iterations to be safe.
    
    max_iters = 5000
    iters_done = 0
    
    # List of probabilities to try
    # We can randomize p_add slightly in each run.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Randomize the probability slightly around a base
        # Base 0.8? 0.9?
        # Let's try a range.
        p_add = random.uniform(0.5, 0.95)
        
        seq = run_greedy(p_add)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            # Print for debugging? No, keep silent.
            
        iters_done += 1

    # If no sequence was found (unlikely), fallback
    if not best_seq:
        # Deterministic greedy
        seq = run_greedy(1.0)
        best_seq = seq
        best_len = len(seq)

    return best_seq
