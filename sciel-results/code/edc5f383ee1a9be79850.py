"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _beam_search_blocks(N, num_blocks, beam_width=50, max_time=140.0):
    """
    Search for a sequence composed of disjoint rectangular pulses (blocks).
    Each block has a start index, width, and sign (+1 or -1).
    We use beam search to explore the space of block placements.
    """
    start_time = time.time()
    
    # Initial beam: single blocks of various widths and positions
    # To keep the search space manageable, we restrict widths to a set of candidates
    # and positions to a grid.
    
    # Candidate widths: 1, 2, 4, 8, 16, 32, 64, 128, 256
    candidate_widths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Candidate positions: every 1 or 2 indices? Let's use step 1 for fine detail initially, 
    # but that's N * len(widths) * 2 signs = 400 * 9 * 2 = 7200 initial states.
    # That's fine.
    
    # State: (score, h_array, blocks_list)
    # h_array is needed for quick evaluation? No, we can reconstruct from blocks.
    # But reconstruction is O(N). With beam_width 50 and depth 10, that's 500 reconstructions.
    # 500 * 400 = 200k ops. Very fast.
    
    # However, evaluating score requires FFT which is O(N log N).
    # 500 * (400 log 400) ~ 500 * 3600 ~ 1.8M ops. Still fast.
    
    # Let's represent a state as a list of (start, width, sign).
    # We must ensure blocks are disjoint and sorted.
    
    # Initialize beam with single blocks
    initial_states = []
    for w in candidate_widths:
        for s in range(0, N - w + 1, 2): # Step 2 to reduce initial count
            for sign in [1, -1]:
                blocks = [(s, w, sign)]
                # Construct h
                h = np.zeros(N)
                h[s:s+w] = sign
                val = _score(h)
                # We want to minimize val, so push with val as key
                initial_states.append((val, blocks))
                
    # Sort and keep top beam_width
    initial_states.sort(key=lambda x: x[0])
    beam = initial_states[:beam_width]
    
    best_val = beam[0][0] if beam else 1e10
    best_blocks = beam[0][1] if beam else []
    
    # Expand beam
    # At each step, add a new block disjoint from existing ones.
    # To keep complexity low, we only consider adding a block that fits in the largest gap.
    # Or simply iterate through all possible new blocks and check disjointness.
    
    for depth in range(1, num_blocks):
        next_beam = []
        
        # Time check
        if time.time() - start_time > max_time - 10:
            break
            
        for val, blocks in beam:
            # Determine available gaps
            # Sort blocks by start
            sorted_blocks = sorted(blocks, key=lambda x: x[0])
            
            # Gaps are: [0, start_0], [end_0, start_1], ..., [end_last, N]
            # We will try to place a new block in each gap.
            
            prev_end = 0
            for b_start, b_width, b_sign in sorted_blocks:
                gap_start = prev_end
                gap_end = b_start
                gap_len = gap_end - gap_start
                
                if gap_len > 0:
                    # Try adding a block in this gap
                    # To reduce candidates, only try widths that fit and are "significant"
                    for w in candidate_widths:
                        if w > gap_len:
                            continue
                        # Try a few positions in the gap
                        # If gap is small, try all. If large, try a few.
                        step = 1 if gap_len < 20 else 2
                        for s in range(gap_start, gap_end - w + 1, step):
                            for sign in [1, -1]:
                                new_blocks = sorted(blocks + [(s, w, sign)], key=lambda x: x[0])
                                # Check disjointness (should be guaranteed by gap logic, but safe to verify if we change gap logic)
                                # Construct h
                                h = np.zeros(N)
                                for ns, nw, nsig in new_blocks:
                                    h[ns:ns+nw] = nsig
                                
                                val_new = _score(h)
                                next_beam.append((val_new, new_blocks))
                
                prev_end = b_start + b_width
            
            # Last gap
            gap_start = prev_end
            gap_end = N
            gap_len = gap_end - gap_start
            if gap_len > 0:
                for w in candidate_widths:
                    if w > gap_len:
                        continue
                    step = 1 if gap_len < 20 else 2
                    for s in range(gap_start, gap_end - w + 1, step):
                        for sign in [1, -1]:
                            new_blocks = sorted(blocks + [(s, w, sign)], key=lambda x: x[0])
                            h = np.zeros(N)
                            for ns, nw, nsig in new_blocks:
                                h[ns:ns+nw] = nsig
                            
                            val_new = _score(h)
                            next_beam.append((val_new, new_blocks))
        
        if not next_beam:
            break
            
        # Sort and prune
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
        if beam[0][0] < best_val:
            best_val = beam[0][0]
            best_blocks = beam[0][1]
            
        # Early stopping if we hit a very low value?
        # The theoretical minimum for sum=1 is 1/N = 0.0025.
        # If we get close to that, we can stop.
        if best_val < 0.003:
            break
            
    # Reconstruct best h
    h_best = np.zeros(N)
    for s, w, sign in best_blocks:
        h_best[s:s+w] = sign
        
    return h_best, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0 # Safety margin
    
    # Run beam search for different numbers of blocks
    # More blocks allow for more complex shapes, but search space explodes.
    # Let's try 1, 2, 3, 4 blocks.
    
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Baseline: constant sequence
    # Note: _score normalizes by sum^2, so scaling doesn't matter, but sum=1 is standard.
    
    for num_blocks in [1, 2, 3, 4]:
        if time.time() > deadline - 20:
            break
            
        try:
            h_candidate, val_candidate = _beam_search_blocks(N, num_blocks, beam_width=20, max_time=deadline - time.time() - 5)
            if val_candidate < best_score:
                best_score = val_candidate
                best_h = h_candidate
        except Exception:
            pass
            
    # Final local refinement: flip individual bits?
    # Since we are in "block" land, flipping a bit breaks the block structure.
    # But the task allows ANY signed step function.
    # So we can take the best block solution and refine it with local search.
    
    if time.time() < deadline - 10:
        h_refined = best_h.copy()
        current_val = best_score
        improved = True
        passes = 0
        while improved and passes < 5 and time.time() < deadline - 5:
            improved = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                h_refined[i] = -h_refined[i]
                new_val = _score(h_refined)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    h_refined[i] = -h_refined[i] # revert
            if current_val < best_score:
                best_score = current_val
                best_h = h_refined.copy()
            else:
                # If no improvement in a full pass, break
                if not improved:
                    break
                    # Actually, if improved is False, we break the while loop anyway.
                    # But we need to revert the last change if it didn't improve?
                    # The loop reverts if not improved. So h_refined is at the best state of the previous pass.
                    # Wait, if improved is False in this pass, h_refined is unchanged from start of pass.
                    # So best_h is already correct.
                
    return best_h

# EVOLVE-BLOCK-END
