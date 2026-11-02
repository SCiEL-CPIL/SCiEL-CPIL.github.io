import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute the maximum overlap over all non-zero shifts."""
    h = np.asarray(h)
    max_ov = 0.0
    for k in range(1, N):
        ov = np.sum(h * (1 - np.roll(h, k)))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _gen_msequence(m):
    """Generate an m-sequence (maximum length LFSR sequence) of length 2^m - 1."""
    # Use a primitive polynomial for the LFSR
    # For m=7, use x^7 + x^3 + 1 (0b1001001)
    # Feedback taps at positions 0 and 2 (0-indexed from LSB)
    
    if m == 7:
        taps = [0, 2]  # feedback from bit 0 and bit 2
    elif m == 6:
        taps = [0, 1]  # x^6 + x + 1
    elif m == 5:
        taps = [0, 3]  # x^5 + x^3 + 1
    elif m == 4:
        taps = [0, 3]  # x^4 + x + 1
    elif m == 3:
        taps = [0, 1]  # x^3 + x + 1
    else:
        taps = [0, 1]
    
    length = (1 << m) - 1
    state = 1  # non-zero initial state
    seq = []
    
    for _ in range(length):
        seq.append(state & 1)
        # Compute feedback
        feedback = 0
        for tap in taps:
            feedback ^= (state >> tap) & 1
        # Shift right and add feedback at MSB
        state = (state >> 1) | (feedback << (m - 1))
    
    return np.array(seq, dtype=np.float64)

def _make_balanced_mseq(m, N):
    """Generate m-sequence and adjust to have exactly N/2 ones in length N."""
    mseq = _gen_msequence(m)
    # Take a window of length N
    seq = mseq[:N].copy()
    
    # Count ones
    ones = np.sum(seq)
    target = N // 2
    
    if ones < target:
        # Need to add (target - ones) ones: change some 0s to 1s
        zeros = np.where(seq == 0)[0]
        for i in range(int(target - ones)):
            seq[zeros[i]] = 1
    elif ones > target:
        # Need to remove (ones - target) ones: change some 1s to 0s
        ones_idx = np.where(seq == 1)[0]
        for i in range(int(ones - target)):
            seq[ones_idx[i]] = 0
    
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)  # should be 25.0
    
    # Try multiple m-sequence based initializations
    np.random.seed(42)
    
    # Try different m-sequence lengths and offsets
    candidates = []
    
    for m in [7, 6, 5, 4, 3]:
        mseq = _gen_msequence(m)
        mlen = len(mseq)
        # Try different offsets
        for offset in range(0, min(mlen, 20), 5):
            # Create a repeating pattern of length mlen, take N elements
            idx = (np.arange(N) + offset) % mlen
            seq = mseq[idx].copy()
            
            # Balance to exactly N/2 ones
            ones = np.sum(seq)
            target = N // 2
            
            if ones < target:
                zeros = np.where(seq == 0)[0]
                np.random.shuffle(zeros)
                for i in range(int(target - ones)):
                    seq[zeros[i]] = 1
            elif ones > target:
                ones_idx = np.where(seq == 1)[0]
                np.random.shuffle(ones_idx)
                for i in range(int(ones - target)):
                    seq[ones_idx[i]] = 0
            
            candidates.append(seq)
    
    # Also try the direct m-sequence approach
    candidates.append(_make_balanced_mseq(7, N))
    
    # Evaluate all candidates
    for h in candidates:
        assert np.sum(h) == N // 2, f"Sum is {np.sum(h)}, expected {N//2}"
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search from the best candidate
    h = best_h.copy()
    current_max_ov = best_score
    
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a new m-sequence based structure
            np.random.seed(int(time.time() * 1000) % (2**31))
            m = np.random.choice([7, 6, 5])
            mseq = _gen_msequence(m)
            mlen = len(mseq)
            offset = np.random.randint(0, mlen)
            idx = (np.arange(N) + offset) % mlen
            h = mseq[idx].copy()
            
            ones = np.sum(h)
            target = N // 2
            if ones < target:
                zeros_idx = np.where(h == 0)[0]
                np.random.shuffle(zeros_idx)
                for i in range(int(target - ones)):
                    h[zeros_idx[i]] = 1
            elif ones > target:
                ones_idx = np.where(h == 1)[0]
                np.random.shuffle(ones_idx)
                for i in range(int(ones - target)):
                    h[ones_idx[i]] = 0
            
            current_max_ov = _overlap_max(h)
            if current_max_ov < best_score:
                best_score = current_max_ov
                best_h = h.copy()
            no_improve_count = 0
            max_no_improve = 300
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            continue
        
        # Pick a random 1 and a random 0 to swap
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        if i == j:
            continue
        
        # Try the swap
        h[i] = 0
        h[j] = 1
        
        new_max_ov = _overlap_max(h)
        
        if new_max_ov < current_max_ov:
            # Accept
            current_max_ov = new_max_ov
            # Update ones and zeros
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            no_improve_count = 0
            if current_max_ov < best_score:
                best_score = current_max_ov
                best_h = h.copy()
        else:
            # Revert
            h[i] = 1
            h[j] = 0
            no_improve_count += 1
    
    # Return the best found
    return best_h
