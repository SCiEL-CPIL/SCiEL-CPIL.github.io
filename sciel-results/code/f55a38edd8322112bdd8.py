# ... imports ...

N = 18
DEADLINE = 84.0
_rng = random.Random(20240517)

def _verify_full(net):
    # ... same as before ...
    pass

def _verify_hard(net, hard):
    # ... same as before ...
    pass

def _build_hard():
    """A random subset of masks for fast verification."""
    # Sample 5000 random masks
    all_masks = list(range(1 << N))
    _rng.shuffle(all_masks)
    return all_masks[:5000]

def _seed_odd_even():
    # ... same ...
    pass

def _seed_bitonic():
    # Removed or fixed. I'll just return [] to be safe, and filter it out.
    return []

def _seed_batcher():
    # ... same ...
    pass

def construct():
    t0 = time.time()
    hard = _build_hard()

    seeds = [_seed_odd_even(), _seed_batcher()]
    seeds = [s for s in seeds if s and _verify_full(s)]
    if not seeds:
        return _seed_odd_even()

    best = min(seeds, key=len)
    best_len = len(best)
    best_net = list(best)
    
    # Simulated Annealing
    cur_net = list(best)
    cur_len = len(cur_net)
    T = 1.0
    cooling = 0.995
    
    while time.time() - t0 < DEADLINE:
        if T < 0.01:
            break
        T *= cooling
        
        # Generate neighbor
        op = _rng.choice(['remove', 'add', 'swap', 'move'])
        cand = list(cur_net)
        
        if op == 'remove' and len(cand) > 0:
            i = _rng.randrange(len(cand))
            cand.pop(i)
        elif op == 'add' and len(cand) < 4096:
            a = _rng.randrange(N)
            b = _rng.randrange(N)
            if a != b:
                if a > b: a, b = b, a
                pos = _rng.randrange(len(cand) + 1)
                cand.insert(pos, (a, b))
        elif op == 'swap' and len(cand) >= 2:
            a = _rng.randrange(len(cand))
            b = _rng.randrange(len(cand))
            if a != b:
                cand[a], cand[b] = cand[b], cand[a]
        elif op == 'move' and len(cand) >= 2:
            a = _rng.randrange(len(cand))
            b = _rng.randrange(len(cand))
            if a != b:
                item = cand.pop(a)
                cand.insert(b, item)
        
        # Evaluate
        # We want to minimize length, but must be valid.
        # If invalid, penalty.
        # Use hard subset for speed.
        is_valid_hard = _verify_hard(cand, hard)
        
        # Cost function:
        # If valid on hard, cost = len(cand)
        # If invalid, cost = len(cand) + 1000 (large penalty)
        # This encourages validity.
        
        cur_cost = cur_len if _verify_hard(cur_net, hard) else cur_len + 1000
        cand_cost = len(cand) if is_valid_hard else len(cand) + 1000
        
        delta = cand_cost - cur_cost
        
        if delta < 0 or _rng.random() < pow(2.71828, -delta / T):
            cur_net = cand
            cur_len = len(cur_net)
            
            # If this candidate is valid on hard, check if it's a new best valid solution
            if is_valid_hard:
                # Optional: quick full check on best candidates? 
                # Might be too slow. Only check if len < best_len.
                if cur_len < best_len:
                    if _verify_full(cur_net):
                        best_net = list(cur_net)
                        best_len = cur_len

    # Final check
    if not _verify_full(best_net):
        return _seed_odd_even()
        
    return best_net
