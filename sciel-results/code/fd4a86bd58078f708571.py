"""Third autocorrelation inequality — cellular automaton + simulated annealing with block flips."""
import numpy as np
import time

N = 400
DEADLINE = 110.0  # seconds, safety margin below 150s evaluator limit


def _compute_autocorr(h):
    """
    Compute the third-order autocorrelation (autoconvolution of f with f*f).
    For a discrete signal h of length N, C3[k] = sum_i h[i] * h[i+k] * h[i+2k]
    over valid i for each lag k.
    
    The "upper bound on C3" is the maximum absolute value of this triple product sum.
    """
    h_float = h.astype(np.float64)
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    autoconv_fft = h_fft * h_fft
    c3_fft = autoconv_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    c3 = c3_full[:N]
    return c3


def _score(h):
    """The evaluator measures the raw value: max |C3| (lower is better)."""
    c3 = _compute_autocorr(h)
    return np.max(np.abs(c3))


def _ca_rule_110(seed_state, steps):
    """
    Generate a binary sequence from elementary cellular automaton rule 110.
    seed_state: initial random binary state of length N
    steps: number of CA steps to evolve
    Returns a binary array of length N from the final state.
    """
    state = seed_state.copy()
    for _ in range(steps):
        # Rule 110: 01101110 in binary (from 111 to 000)
        # new[i] = (left & 0) | (center & 0) | (right & 0) ... 
        # Rule 110 mapping: 111->0, 110->1, 101->1, 100->0, 011->1, 010->1, 001->1, 000->0
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        # Compute rule 110
        new_state = np.zeros_like(state)
        # 111 -> 0
        mask_111 = (left == 1) & (state == 1) & (right == 1)
        new_state[mask_111] = 0
        # 110 -> 1
        mask_110 = (left == 1) & (state == 1) & (right == 0)
        new_state[mask_110] = 1
        # 101 -> 1
        mask_101 = (left == 1) & (state == 0) & (right == 1)
        new_state[mask_101] = 1
        # 100 -> 0
        mask_100 = (left == 1) & (state == 0) & (right == 0)
        new_state[mask_100] = 0
        # 011 -> 1
        mask_011 = (left == 0) & (state == 1) & (right == 1)
        new_state[mask_011] = 1
        # 010 -> 1
        mask_010 = (left == 0) & (state == 1) & (right == 0)
        new_state[mask_010] = 1
        # 001 -> 1
        mask_001 = (left == 0) & (state == 0) & (right == 1)
        new_state[mask_001] = 1
        # 000 -> 0
        mask_000 = (left == 0) & (state == 0) & (right == 0)
        new_state[mask_000] = 0
        state = new_state
    return state


def _ca_rule_30(seed_state, steps):
    """
    Generate a binary sequence from elementary cellular automaton rule 30.
    """
    state = seed_state.copy()
    for _ in range(steps):
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        # Rule 30: 00011110
        # 111->0, 110->0, 101->0, 100->1, 011->1, 010->1, 001->1, 000->0
        new_state = np.zeros_like(state)
        mask_100 = (left == 1) & (state == 0) & (right == 0)
        new_state[mask_100] = 1
        mask_011 = (left == 0) & (state == 1) & (right == 1)
        new_state[mask_011] = 1
        mask_010 = (left == 0) & (state == 1) & (right == 0)
        new_state[mask_010] = 1
        mask_001 = (left == 0) & (state == 0) & (right == 1)
        new_state[mask_001] = 1
        state = new_state
    return state


def _ca_rule_45(seed_state, steps):
    """
    Generate a binary sequence from elementary cellular automaton rule 45.
    """
    state = seed_state.copy()
    for _ in range(steps):
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        # Rule 45: 00101101
        # 111->0, 110->0, 101->1, 100->0, 011->1, 010->1, 001->0, 000->1
        new_state = np.ones_like(state)
        mask_111 = (left == 1) & (state == 1) & (right == 1)
        new_state[mask_111] = 0
        mask_110 = (left == 1) & (state == 1) & (right == 0)
        new_state[mask_110] = 0
        mask_100 = (left == 1) & (state == 0) & (right == 0)
        new_state[mask_100] = 0
        mask_001 = (left == 0) & (state == 0) & (right == 1)
        new_state[mask_001] = 0
        state = new_state
    return state


def _ca_generate(rule_func, n_steps, seed_offset=0):
    """Generate a sign sequence from a CA with given rule and steps."""
    rng = np.random.default_rng(seed_offset)
    seed_state = rng.integers(0, 2, size=N).astype(np.int8)
    final_state = rule_func(seed_state, n_steps)
    # Convert binary to ±1
    h = 1.0 - 2.0 * final_state.astype(np.float64)
    return h


def _block_flip(h, block_size, rng):
    """Flip a contiguous block of signs in h."""
    h_new = h.copy()
    start = rng.integers(0, N - block_size)
    h_new[start:start + block_size] *= -1
    return h_new


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Returns a sequence of signed step heights.
    Score is the upper bound on C3 (max |C3|), lower is better.
    """
    start_time = time.time()
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Phase 1: Generate diverse initial candidates using cellular automata
    ca_rules = [_ca_rule_110, _ca_rule_30, _ca_rule_45]
    n_steps_range = range(1, 30)
    
    best_h = None
    best_score = np.inf
    
    # Try various CA configurations
    for rule_func in ca_rules:
        for n_steps in n_steps_range:
            if not elapsed_ok():
                break
            for seed_offset in range(10):
                if not elapsed_ok():
                    break
                h = _ca_generate(rule_func, n_steps, seed_offset)
                s = _score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()
        if not elapsed_ok():
            break
    
    # Also try random initializations as a baseline
    for _ in range(50):
        if not elapsed_ok():
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    # Phase 2: Simulated annealing with block flips
    h_work = best_h.copy()
    current_score = best_score
    T = 0.5  # initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    rng = np.random.default_rng(42)
    
    max_iter = 50000
    for iteration in range(max_iter):
        if not elapsed_ok():
            break
        
        # Choose block size: mix of small and large blocks
        if rng.random() < 0.3:
            block_size = rng.integers(1, 5)
        elif rng.random() < 0.5:
            block_size = rng.integers(5, 20)
        else:
            block_size = rng.integers(20, 50)
        
        h_try = _block_flip(h_work, block_size, rng)
        s_try = _score(h_try)
        
        delta = s_try - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            h_work = h_try
            current_score = s_try
            if current_score < best_score:
                best_score = current_score
                best_h = h_work.copy()
        
        # Cool down
        T *= cooling_rate
    
    # Phase 3: Fine-tuning with single-bit flips on the best solution
    if elapsed_ok():
        h_work = best_h.copy()
        current_score = best_score
        for _ in range(10000):
            if not elapsed_ok():
                break
            idx = rng.integers(0, N)
            h_try = h_work.copy()
            h_try[idx] *= -1
            s_try = _score(h_try)
            if s_try < current_score:
                h_work = h_try
                current_score = s_try
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_work.copy()
    
    # Final: ensure output is ±1
    best_h = np.sign(best_h).astype(np.float64)
    if np.all(best_h == 0):
        best_h = np.ones(N)
    return best_h


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
