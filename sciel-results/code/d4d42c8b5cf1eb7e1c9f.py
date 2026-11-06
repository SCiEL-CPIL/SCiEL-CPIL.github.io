import numpy as np
import time
import random

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    
    # Initialize residues to 0
    residues = {d: 0 for d in divisors}
    
    # Initialize coverage count array
    count = np.zeros(N, dtype=np.int8)
    
    # Apply initial residues
    for d in divisors:
        r = residues[d]
        count[r::d] += 1
        
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = np.sum(count == 0)
    best_residues = residues.copy()
    
    # Precompute indices for each divisor? 
    # arange(r, N, d) is fast enough.
    
    # We will perform multiple sweeps.
    # To diversify, we can shuffle the order of divisors in each sweep.
    
    while time.time() - start_time < timeout:
        improved = False
        # Shuffle divisors to avoid bias towards specific order
        shuffled_divs = divisors[:]
        random.shuffle(shuffled_divs)
        
        for d in shuffled_divs:
            if time.time() - start_time > timeout:
                break
                
            r_old = residues[d]
            
            # Calculate loss: number of x in class (d, r_old) that are covered exactly once
            # If we remove this class, these become uncovered.
            x_old = np.arange(r_old, N, d)
            loss = np.sum(count[x_old] == 1)
            
            # Calculate gain for all possible new residues r_new
            # gain(r_new) = number of x in class (d, r_new) that are currently uncovered
            # We want to maximize gain.
            
            # Get indices of currently uncovered residues
            # If there are no uncovered residues, we are done (score 0)
            if best_score == 0:
                return [[d, int(residues[d])] for d in divisors]
                
            uncovered_idx = np.nonzero(count == 0)[0]
            
            if len(uncovered_idx) == 0:
                return [[d, int(residues[d])] for d in divisors]
                
            # For each uncovered x, x % d gives the residue class it falls into.
            # Count how many uncovered x fall into each residue class mod d.
            mods = uncovered_idx % d
            gains = np.bincount(mods, minlength=d)
            
            # Find the best residue to switch to
            best_r = np.argmax(gains)
            max_gain = gains[best_r]
            
            # Check if switching improves the score
            # delta = loss - max_gain
            # We want delta < 0, i.e., max_gain > loss
            
            if best_r != r_old and max_gain > loss:
                # Update count array
                count[r_old::d] -= 1
                count[best_r::d] += 1
                residues[d] = best_r
                improved = True
                
                # Update current score estimate
                # score_change = loss - max_gain
                # best_score is the global best, we track current score separately for early exit?
                # Actually, we can just recompute if we want to be safe, but it's O(N).
                # The loop is fast, so maybe just continue.
                # However, if we reach 0, we can exit early.
                # Let's check if score became 0.
                # This check is O(N). Doing it every improvement might be costly if many improvements.
                # But N=51975 is small.
                # Let's just track the score change.
                # current_score -= (max_gain - loss)
                # If current_score == 0, exit.
                
        if not improved:
            # If no improvement in a full sweep, we might be stuck in a local minimum.
            # Try a random perturbation: pick a few divisors and change their residues randomly.
            # Then restart the sweep.
            
            # If we've been stuck for a while, maybe restart with a different initial state?
            # For now, just try a few random moves.
            
            # Randomly pick 5 divisors and change their residues
            for _ in range(5):
                if time.time() - start_time > timeout:
                    break
                d = random.choice(divisors)
                r_old = residues[d]
                r_new = random.randint(0, d - 1)
                if r_new == r_old:
                    continue
                
                # Evaluate this specific move
                x_old = np.arange(r_old, N, d)
                loss = np.sum(count[x_old] == 1)
                
                x_new = np.arange(r_new, N, d)
                gain = np.sum(count[x_new] == 0)
                
                # Accept if it doesn't make things worse? Or always accept to escape?
                # Let's accept if it's neutral or better, or with some probability if worse.
                # Simple: accept if gain >= loss.
                if gain >= loss:
                    count[r_old::d] -= 1
                    count[r_new::d] += 1
                    residues[d] = r_new
                    improved = True
                    break # Restart sweep if we found a move? Or just continue?
                    
            if not improved:
                # Completely stuck. Try a random restart?
                # Or just break and return best found.
                # Let's try a few more sweeps with different shuffles before giving up.
                # But if 'improved' is False, the next sweep will likely also fail unless we perturb.
                # The perturbation above only breaks if it found a move. If it didn't, improved is still False.
                # So we break the while loop.
                break
                
        # Update global best
        current_score = np.sum(count == 0)
        if current_score < best_score:
            best_score = current_score
            best_residues = residues.copy()
            
    return [[d, int(best_residues[d])] for d in divisors]
