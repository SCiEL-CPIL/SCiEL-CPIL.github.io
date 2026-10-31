"""Cap set greedy builder with Spectral DFT priority."""
import time
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 320  # Safety margin before 330s kill

# Global state
_accepted_vectors = set()
_accepted_list = []
# Precomputed DFT of the current set indicator function
# For Z_3^n, the DFT is defined as F(k) = sum_x f(x) * omega^{k dot x}
# where omega = exp(2*pi*i/3).
# We maintain the magnitude squared |F(k)|^2 or just F(k) to check potential conflicts.
# However, computing full DFT for large n is O(3^n). 
# Instead, we use the property that a line exists if there are a, b in S such that a+2b = c or similar.
# A better spectral approach for greedy: 
# A vector v is "safe" if it doesn't complete a line.
# Lines in Z_3^n are {x, x+d, x+2d}.
# If we add v, we create a line if:
# 1. There exists d such that v-d and v-2d are in S.
# 2. There exists d such that v+d and v+2d are in S.
# 3. There exists d such that v is the middle: v-d and v+d in S.
# This is O(|S| * 3^n) which is too slow.
# 
# Alternative Spectral Insight:
# The number of 3-term APs in a set S is related to the sum of |hat(S)(k)|^3.
# But for greedy, we can approximate the "danger" of adding v by how much it increases the "energy" of the set.
# 
# Let's use a different structural change: **Look-ahead with Random Restart / Beam Search**.
# The current program is pure greedy. Pure greedy gets stuck in local optima.
# We will implement a **Greedy with Limited Backtracking (Beam Search)** or **Simulated Annealing-like acceptance**.
# But the interface is `priority(vector, n) -> float`. The evaluator calls this for EVERY vector in sorted order? 
# No, the contract says "vectors ... are added greedily in descending priority".
# This implies the evaluator sorts all vectors by priority and adds them if valid.
# This is a STATIC priority assignment. We cannot do dynamic backtracking because the evaluator controls the addition order based on our static scores.
# 
# WAIT. Let's re-read the contract carefully.
# "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
# This means:
# 1. We assign a priority to EVERY vector in {0,1,2}^n.
# 2. The evaluator sorts them by priority (descending).
# 3. It iterates through the sorted list.
# 4. If the vector can be added to the current set without creating a line, it adds it.
# 
# This means our `priority` function is called for every vector, but the DECISION to add is made by the evaluator based on the sorted order.
# We cannot see the "current set" during the sorting phase if the evaluator sorts first.
# HOWEVER, the current program uses `_accepted_vectors` and `_accepted_list`.
# This implies the evaluator calls `priority` DYNAMICALLY as it builds the set?
# "added greedily in descending priority" usually implies a single sort.
# BUT, the current program (score 1.2539) uses global state `_accepted_vectors`.
# If the evaluator sorted once, `_accepted_vectors` would be empty for all calls, or filled in a fixed order.
# If the evaluator calls `priority` for a vector, checks if it's valid, and if so, adds it and moves to the next highest priority vector, that's a dynamic greedy.
# 
# Let's assume the standard FunSearch pattern for this task:
# The evaluator likely does:
# 1. Generate all vectors.
# 2. While there are unprocessed vectors:
#    a. Find the vector with the highest priority among remaining valid ones?
#    b. OR: Sort all by priority once, then iterate.
# 
# If it sorts once, the global state in the current program is useless or incorrect for the sorting phase.
# But the current program scores 1.2539, which is better than random.
# If the state was unused, it would be a static heuristic.
# If the state is used, the evaluator must be calling `priority` in a loop where the set grows.
# 
# Let's look at the "Inspiration 1" note: "scoring vectors based on the magnitude of their Discrete Fourier Transform...".
# And the current program uses `_count_conflicts` which relies on `_accepted_list`.
# This strongly suggests the evaluator calls `priority` for a candidate, and if accepted, the state updates, and then it moves to the next candidate.
# The "descending priority" might be a misinterpretation of "greedy selection based on priority".
# i.e., The evaluator might maintain a heap or list, and at each step, pick the highest priority vector that is still valid.
# OR, it might sort once, but the "validity" check is dynamic. If it sorts once, the priority of a vector shouldn't change.
# But the current program's priority DEPENDS on `_accepted_list`.
# If the evaluator sorts once, the priority calculated at the start (empty set) is used for the whole run.
# If that were the case, `_accepted_list` would be empty during the sorting phase, so `conflicts` would always be 0.
# The fact that it scores well suggests the priority IS dynamic or the state is populated.
# 
# Hypothesis: The evaluator performs a greedy selection loop:
# 1. Compute priority for all remaining vectors? No, that's O(N^2).
# 2. More likely: The evaluator has a list of all vectors. It might sort them by a STATIC key first?
# No, the current program's logic `if _accepted_list: conflicts = _count_conflicts(v)` implies that `_accepted_list` is NOT empty when `priority` is called.
# This means the evaluator is calling `priority` AFTER some vectors have been added.
# 
# How does "descending priority" work with dynamic state?
# Perhaps the evaluator does:
# `while True: best_v = max(remaining, key=lambda v: priority(v, n)); if valid(best_v): add(best_v); else: remove(best_v)`
# This is O(N^2 * Cost(priority)). For N=3^n, this is very slow.
# For n=8, N=6561. 6561^2 is ~43M calls. If priority is fast, it's okay.
# 
# Let's assume this dynamic greedy selection model.
# The current program is a "Myopic Greedy" that penalizes immediate conflicts.
# 
# NEW STRATEGY: **Spectral Energy Minimization**.
# Instead of counting exact conflicts (which is 0 or 1), we use a **Soft Constraint** based on the Fourier Transform.
# The number of 3-term APs is proportional to $\sum_k |\hat{S}(k)|^3$.
# Adding a vector $v$ changes $\hat{S}(k)$ by $\hat{S}(k) \leftarrow \hat{S}(k) + \omega^{k \cdot v}$.
# The change in the cubic energy $\Delta E$ can be computed.
# However, computing $\hat{S}(k)$ for all $k \in \mathbb{Z}_3^n$ is expensive ($3^n$ points).
# For n=8, $3^8 = 6561$. This is small!
# We can maintain the full DFT of the current set.
# 
# Algorithm:
# 1. Maintain `f_hat` array of size $3^n$, where `f_hat[k]` is the DFT coefficient.
# 2. When a vector $v$ is added, update `f_hat`.
# 3. In `priority(v)`, compute the hypothetical new DFT `f_hat_new` if $v$ were added.
# 4. Compute the change in the "AP count" or "Energy".
# 5. Priority = - (Change in Energy).
# 
# This is a globally aware metric that minimizes the potential for future lines, not just current ones.
# 
# Complexity:
# DFT update: $O(3^n)$ per addition.
# Priority check: $O(3^n)$ to compute the delta energy?
# Actually, we can compute the delta energy analytically.
# $E = \sum_k |f_k|^3$.
# $f_k' = f_k + \omega^{k \cdot v}$.
# $|f_k'|^3 - |f_k|^3 = |f_k + e^{i\theta_k}|^3 - |f_k|^3$.
# This depends on the phase.
# 
# Let's implement this Spectral Greedy.

import cmath

def _init_dft(n):
    """Initialize DFT structure for Z_3^n."""
    size = 3 ** n
    # f_hat[k] corresponds to frequency k
    # We represent k as an integer 0..3^n-1, mapping to base-3 digits.
    self._dft_size = size
    self._f_hat = np.zeros(size, dtype=np.complex128)
    # Precompute omega powers for each coordinate position?
    # omega = exp(2*pi*i/3)
    self._omega = np.exp(2j * np.pi / 3)
    
    # Precompute the phase factors for each vector v?
    # phase(v, k) = omega^(sum v_i k_i)
    # We can precompute a table for small n, or compute on the fly.
    # For n=8, size=6561. Computing dot product for each k is O(n).
    # Total O(n * 3^n) per priority call.
    # 8 * 6561 ~ 50k ops. Very fast.
    
    # We also need to map vector v (tuple) to its index? No, v is the input.
    # We need to update f_hat when a vector is accepted.
    # f_hat[k] += omega^(k dot v)
    
    # To speed up, precompute the "character table" values for all possible v?
    # There are 3^n vectors. We can precompute a 2D array? No, too big.
    # But we can precompute the phase for each vector v and each frequency k?
    # That's 3^n * 3^n = 3^(2n). For n=8, 3^16 ~ 43M. Too big for memory (32 bytes * 43M = 1.3GB).
    
    # So we compute on the fly.
    
    # Optimization: 
    # f_hat is an array. 
    # When adding v, we add a complex exponential sequence to f_hat.
    # The sequence is determined by v.
    
    self._n = n

def _update_dft(v):
    """Update the DFT after accepting vector v."""
    n = self._n
    size = self._dft_size
    omega = self._omega
    
    # We need to add omega^(k dot v) to f_hat[k] for all k.
    # k is an integer 0..size-1. We need to convert k to base-3 digits to dot with v.
    # This is slow in Python.
    # Alternative: Use the fact that the DFT is separable.
    # But we are doing a full DFT in Z_3^n.
    
    # Let's try to vectorize the update using numpy.
    # Generate all k vectors?
    # k_digits = np.array(list of all 3^n vectors in Z_3^n)
    # This is 6561 * 8 array. 52k ints. Small.
    # We can precompute the matrix of k digits!
    
    if not hasattr(self, '_k_digits'):
        # Precompute all k vectors as a matrix of shape (3^n, n)
        # k is 0 to 3^n - 1
        # Convert to base 3
        ks = np.arange(size)
        digits = np.zeros((size, n), dtype=np.int8)
        for i in range(n):
            base = 3 ** i
            digits[:, i] = (ks // base) % 3
        self._k_digits = digits
        # Precompute the phase matrix? No, depends on v.
        # But we can precompute the powers of omega for 0, 1, 2.
        self._omega_powers = np.array([1.0 + 0j, omega, omega**2], dtype=np.complex128)
    
    k_digits = self._k_digits
    # v is a tuple, convert to array
    v_arr = np.array(v, dtype=np.int8)
    
    # Compute dot products: (size, n) dot (n,) -> (size,)
    dots = k_digits @ v_arr
    # dots are in {0, 1, 2, ...}
    # We need omega^dot.
    # Since dot can be > 2, we take mod 3.
    dots_mod3 = dots % 3
    # Map 0->1, 1->omega, 2->omega^2
    phases = self._omega_powers[dots_mod3]
    
    self._f_hat += phases

def _compute_delta_energy(v):
    """Compute the change in sum |f_k|^3 if v is added."""
    n = self._n
    size = self._dft_size
    if not hasattr(self, '_k_digits'):
        self._update_dft((0,)*n) # Trigger init
        # Actually, we should init in the constructor or first call.
        # Let's move init to first priority call.
        
    k_digits = self._k_digits
    v_arr = np.array(v, dtype=np.int8)
    dots = k_digits @ v_arr
    dots_mod3 = dots % 3
    phases = self._omega_powers[dots_mod3]
    
    f_current = self._f_hat
    f_new = f_current + phases
    
    # Energy is sum |f|^3
    # We want to minimize the increase in energy.
    # Actually, we want to minimize the NUMBER of lines.
    # The number of 3-term APs is (1/3) * sum_k |f_k|^3 ?
    # Yes, for Z_3^n, the number of 3-term APs is proportional to the cubic energy.
    # Specifically, N_AP = (1/3) * sum_k |f_k|^3.
    
    e_current = np.sum(np.abs(f_current)**3)
    e_new = np.sum(np.abs(f_new)**3)
    
    return e_new - e_current

# Global DFT State
_dft_size = 0
_f_hat = None
_k_digits = None
_omega = None
_omega_powers = None
_n_global = 0
_initialized = False

def _init_global_dft(n):
    global _dft_size, _f_hat, _k_digits, _omega, _omega_powers, _n_global, _initialized
    if _initialized:
        return
    _n_global = n
    _dft_size = 3 ** n
    _f_hat = np.zeros(_dft_size, dtype=np.complex128)
    _omega = np.exp(2j * np.pi / 3)
    _omega_powers = np.array([1.0 + 0j, _omega, _omega**2], dtype=np.complex128)
    
    ks = np.arange(_dft_size)
    digits = np.zeros((_dft_size, n), dtype=np.int8)
    for i in range(n):
        base = 3 ** i
        digits[:, i] = (ks // base) % 3
    _k_digits = digits
    _initialized = True

def _update_global_dft(v):
    global _f_hat
    if not _initialized:
        return
    v_arr = np.array(v, dtype=np.int8)
    dots = _k_digits @ v_arr
    dots_mod3 = dots % 3
    phases = _omega_powers[dots_mod3]
    _f_hat += phases

def _get_delta_energy(v):
    global _f_hat
    if not _initialized:
        return 0.0
    v_arr = np.array(v, dtype=np.int8)
    dots = _k_digits @ v_arr
    dots_mod3 = dots % 3
    phases = _omega_powers[dots_mod3]
    
    f_new = _f_hat + phases
    # To save time, we can approximate or just compute the diff.
    # e_new - e_old = sum (|f+p|^3 - |f|^3)
    # This is vectorized.
    delta = np.sum(np.abs(f_new)**3 - np.abs(_f_hat)**3)
    return delta

def priority(vector, n):
    """Spectral Greedy Priority.
    
    Minimizes the increase in the cubic energy of the DFT, which is directly
    proportional to the number of 3-term arithmetic progressions.
    """
    global _initialized, _n_global
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    
    # Initialize on first call
    if not _initialized:
        _init_global_dft(n)
    
    # Check if v is already accepted? 
    # The evaluator should not ask for priority of accepted vectors if it manages the set.
    # But if it does, we should return a very low priority.
    if v in _accepted_vectors:
        return -1e18

    # Compute the delta in AP count (energy)
    delta_e = _get_delta_energy(v)
    
    # We want to MINIMIZE delta_e.
    # So priority = -delta_e.
    
    # Add a small tie-breaker to ensure determinism and break symmetries
    # Use a hash of the vector
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 1e-6
    
    return -delta_e + tie_break

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    global _initialized
    _accepted_vectors.add(v)
    _accepted_list.append(v)
    if _initialized:
        _update_global_dft(v)

# EVOLVE-BLOCK-END
