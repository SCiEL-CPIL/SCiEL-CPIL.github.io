"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit


def _build_dp_value_function(weights, values, capacity, max_cap_quantized):
    """
    Build a DP value function V(c) = max value achievable with capacity c
    using quantized capacity units. Returns array of size max_cap_quantized+1.
    """
    n = len(weights)
    if n == 0 or max_cap_quantized <= 0:
        return np.zeros(max_cap_quantized + 1)
    
    # Quantize weights
    scale = max_cap_quantized / max(capacity, 1e-9)
    quantized_weights = np.clip((weights * scale).astype(int), 0, max_cap_quantized)
    
    # DP: V[i][c] = max value using items 0..i-1 with capacity c
    # Use 1D rolling array for memory efficiency
    V = np.zeros(max_cap_quantized + 1)
    
    for i in range(n):
        w = quantized_weights[i]
        v = values[i]
        if w == 0:
            # Zero-weight item: always take it
            V[:] += v
            continue
        # Update in reverse to avoid using item twice
        for c in range(max_cap_quantized, w - 1, -1):
            candidate = V[c - w] + v
            if candidate > V[c]:
                V[c] = candidate
    
    return V


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item using a DP value function heuristic.
    The marginal gain of picking item i is approximated by:
    V(capacity) - V(capacity - w_i) - v_i  (opportunity cost)
    We want to minimize opportunity cost / maximize marginal benefit.
    Returns index into remaining arrays, or -1 to stop.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Check time
    if time.time() - _start_time > _TIME_LIMIT:
        return -1

    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1

    # If only one feasible item, take it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Check time again before heavier computation
    if time.time() - _start_time > _TIME_LIMIT - 5:
        # Fallback to density
        densities = remaining_values / remaining_weights
        feasible_densities = densities[feasible_indices]
        return int(feasible_indices[np.argmax(feasible_densities)])

    # Build DP value function with quantized capacity
    # Use a reasonable quantization level based on n
    # With n=100 and capacity ~12.5, we can use moderate quantization
    max_cap_quantized = min(200, max(int(remaining_capacity * 10), 50))
    
    V = _build_dp_value_function(remaining_weights, remaining_values, 
                                  remaining_capacity, max_cap_quantized)
    
    # Check time
    if time.time() - _start_time > _TIME_LIMIT - 2:
        densities = remaining_values / remaining_weights
        feasible_densities = densities[feasible_indices]
        return int(feasible_indices[np.argmax(feasible_densities)])

    scale = max_cap_quantized / max(remaining_capacity, 1e-9)
    cap_quantized = int(remaining_capacity * scale)
    cap_quantized = min(cap_quantized, max_cap_quantized)

    # For each feasible item, compute marginal gain score
    # Score = V(cap) - V(cap - w_i) - v_i
    # This is the "opportunity cost" of NOT being able to use that capacity for other items
    # We want to MINIMIZE this (or maximize negative)
    # Equivalently, maximize: v_i - (V(cap) - V(cap - w_i))
    
    best_score = -np.inf
    best_index = -1
    
    V_cap = V[cap_quantized]
    
    for idx in feasible_indices:
        wi = remaining_weights[idx]
        vi = remaining_values[idx]
        w_quantized = int(wi * scale)
        w_quantized = min(w_quantized, max_cap_quantized)
        
        if w_quantized > cap_quantized:
            continue
        
        # V(cap - w_i) is the max value achievable with remaining capacity after taking i
        V_remaining = V[cap_quantized - w_quantized]
        
        # Marginal benefit = v_i - (V_cap - V_remaining)
        # If this is high, taking i is good because we lose little from the optimal solution
        marginal = vi - (V_cap - V_remaining)
        
        # Add a small density tiebreaker to break ties
        density = vi / wi if wi > 0 else float('inf')
        score = marginal + 0.01 * density
        
        if score > best_score:
            best_score = score
            best_index = int(idx)

    if best_index == -1:
        # Fallback: best density
        densities = remaining_values / remaining_weights
        feasible_densities = densities[feasible_indices]
        return int(feasible_indices[np.argmax(feasible_densities)])

    return best_index
# EVOLVE-BLOCK-END
