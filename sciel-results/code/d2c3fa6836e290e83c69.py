"""Broadband antireflection coating design via characteristic-matrix model + timed search."""
import math
import random
import time

import numpy as np

# Fixed task parameters (from the contract)
N_AMB = 1.0
N_SUB = 1.52
N_L = 1.38
N_H = 2.35
WL0 = 400.0
WL1 = 800.0
N_WL = 401
MAX_LAYERS = 20
MAX_TOTAL_NM = 1000.0
MAX_LAYER_NM = 400.0

# Evaluation wavelengths (401 equally spaced in [400,800])
_WL = np.linspace(WL0, WL1, N_WL)
_K = 2.0 * math.pi / _WL          # 1/nm
# Precompute per-material complex refractive index arrays
_NL = np.full(N_WL, N_L)
_NH = np.full(N_WL, N_H)
_NS = np.full(N_WL, N_SUB)
_NA = np.full(N_WL, N_AMB)

# Deadline guard (wall-clock)
_DEADLINE = None


def _reflectance(labels, thicknesses):
    """Mean normal-incidence reflectance (%) over the band, characteristic-matrix method."""
    n = len(labels)
    if n == 0:
        # Bare substrate reflectance
        r0 = (N_SUB - N_AMB) / (N_SUB + N_AMB)
        return 100.0 * (r0 * np.conj(r0)).real
    
    R_sum = 0.0
    for i_wl in range(N_WL):
        M = np.eye(2, dtype=complex)
        for i in range(n):
            ni = N_L if labels[i] == 'L' else N_H
            t = thicknesses[i]
            delta = _K[i_wl] * ni * t
            c = np.cos(delta)
            s = np.sin(delta)
            Mi = np.array([[c, 1j * s / ni], [1j * ni * s, c]], dtype=complex)
            M = M @ Mi
        
        m11, m12 = M[0, 0], M[0, 1]
        m21, m22 = M[1, 0], M[1, 1]
        
        Y = (m11 + m12 * N_SUB) / (m21 + m22 * N_SUB)
        r = (Y - N_AMB) / (Y + N_AMB)
        R_sum += (r * np.conj(r)).real
        
    return 100.0 * (R_sum / N_WL)


def _valid(labels, thicknesses):
    if len(labels) != len(thicknesses):
        return False
    if len(labels) > MAX_LAYERS:
        return False
    total = 0.0
    for t in thicknesses:
        if t < -1e-6 or t > MAX_LAYER_NM + 1e-6:
            return False
        total += t
    if total > MAX_TOTAL_NM + 1e-6:
        return False
    return True


def _clamp_thicknesses(thicknesses):
    return [min(max(t, 0.0), MAX_LAYER_NM) for t in thicknesses]


def _random_stack(rng, n_layers):
    labels = ['L' if rng.random() < 0.5 else 'H' for _ in range(n_layers)]
    # Random thicknesses, scaled so total stays within budget
    raw = [rng.uniform(10.0, MAX_LAYER_NM) for _ in range(n_layers)]
    total = sum(raw)
    if total > MAX_TOTAL_NM:
        scale = MAX_TOTAL_NM / total
        raw = [t * scale for t in raw]
    return labels, raw


def design(problem):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s wall-clock

    # Strategy: Optimize a periodic alternating stack (L H L H ...)
    # Parameters:
    # 1. base_th: Base thickness for the first layer (and generally the L layers if gradient is 0)
    # 2. grad: Gradient factor. Thickness of layer i = base_th * (1 + grad * i)
    #    This allows for a tapered design which is good for broadband AR.
    # We also need to decide the number of layers.
    # Let's fix the number of layers to a reasonable even number, say 4, 6, or 8.
    # Or we can optimize over a few fixed structures.
    
    best_score = 100.0 # Worst case
    best_labels = ['L']
    best_th = [600.0 / (4.0 * N_L)] # Fallback

    # Define a function to evaluate a specific periodic structure
    def eval_periodic(n_layers, base_th, grad):
        # Generate labels and thicknesses
        labels = []
        thicknesses = []
        total_th = 0.0
        for i in range(n_layers):
            # Alternate L and H starting with L
            labels.append('L' if i % 2 == 0 else 'H')
            t = base_th * (1.0 + grad * i)
            if t < 0.0:
                t = 0.0
            if t > MAX_LAYER_NM:
                t = MAX_LAYER_NM
            thicknesses.append(t)
            total_th += t
        
        if total_th > MAX_TOTAL_NM:
            # Scale down to fit
            if total_th > 0:
                scale = MAX_TOTAL_NM / total_th
                thicknesses = [t * scale for t in thicknesses]
            else:
                return 100.0, labels, thicknesses
            
        # Check individual layer constraints after scaling (should be fine if scaled down)
        # But base_th * (1+grad*i) might have been clamped before scaling?
        # The scaling is linear, so if it was <= MAX_LAYER_NM before, it is after.
        # If it was > MAX_LAYER_NM, it was clamped.
        
        score = _reflectance(labels, thicknesses)
        return score, labels, thicknesses

    # We will use a simple random search + local refinement on the 2D parameter space (base_th, grad)
    # for several fixed n_layers values.
    
    n_layers_options = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for n_layers in n_layers_options:
        if time.time() > deadline - 10.0:
            break
            
        # Local search on (base_th, grad)
        # base_th range: [10, 200]
        # grad range: [-0.5, 0.5]
        
        best_local_score = 100.0
        best_local_params = (50.0, 0.0)
        
        # Initial random samples
        for _ in range(50):
            b = random.uniform(10, 200)
            g = random.uniform(-0.5, 0.5)
            s, l, t = eval_periodic(n_layers, b, g)
            if s < best_local_score:
                best_local_score = s
                best_local_params = (b, g)
        
        # Coordinate descent / Nelder-Mead style refinement
        b, g = best_local_params
        step_b = 20.0
        step_g = 0.1
        improved = True
        iter_count = 0
        max_iter = 50
        
        while improved and iter_count < max_iter and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            
            # Try perturbations in b
            for db in [step_b, -step_b, step_b*0.3, -step_b*0.3]:
                nb = b + db
                if nb < 5 or nb > 300: continue
                s, l, t = eval_periodic(n_layers, nb, g)
                if s < best_local_score:
                    best_local_score = s
                    b = nb
                    improved = True
            
            # Try perturbations in g
            for dg in [step_g, -step_g, step_g*0.3, -step_g*0.3]:
                ng = g + dg
                if ng < -1.0 or ng > 1.0: continue
                s, l, t = eval_periodic(n_layers, b, ng)
                if s < best_local_score:
                    best_local_score = s
                    g = ng
                    improved = True
            
            step_b *= 0.7
            step_g *= 0.7
            
            if best_local_score < best_score:
                best_score = best_local_score
                # Get the final labels and thicknesses for this best local solution
                _, best_labels, best_th = eval_periodic(n_layers, b, g)
                
    return best_labels, best_th
