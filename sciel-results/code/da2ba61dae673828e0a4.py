"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import linalg

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse for a 3-level transmon.

    Returns (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    dt = 0.5  # ns
    N = 80
    T = N * dt  # 40 ns

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    # Time guard
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin

    # Time grid for constant steps
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Hamiltonian / Liouvillian setup
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)

    eye3 = np.eye(3)
    L1c = L1.conj().T
    L2c = L2.conj().T

    # Lindblad contribution to Liouvillian (9x9)
    Lindblad_part = np.zeros((9, 9), dtype=complex)
    for L, Lc in [(L1, L1c), (L2, L2c)]:
        Lindblad_part += (np.kron(Lc, L) - 0.5 * np.kron(L @ Lc, eye3)
                          - 0.5 * np.kron(eye3, L @ Lc))

    def hamiltonian_part(detuning, scale, I, Q):
        H = (detuning * n
             + 0.5 * anharmonicity * (n @ n - n)
             + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        Hc = H.conj().T
        return -1j * (np.kron(eye3, H) - np.kron(Hc, eye3))

    # States to simulate
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]

    # Pre-compute initial density matrices
    rho0s = np.array([np.outer(st, st.conj()).reshape(9) for st in states])

    # Target states for fidelity
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    target_states = [X_gate @ st for st in states]

    def score_pulse(I, Q):
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]

        fids = []
        for d in detunings:
            for s in scales:
                rho_batch = rho0s.copy()

                for k in range(N):
                    H_part = hamiltonian_part(d, s, I[k], Q[k])
                    L_full = Lindblad_part + H_part
                    U = linalg.expm(L_full * dt)
                    rho_batch = (U @ rho_batch.T).T

                case_fids = []
                for i in range(6):
                    rho_final = rho_batch[i].reshape(3, 3)
                    psi_target = target_states[i]
                    fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
                    case_fids.append(fid)
                fids.append(np.mean(case_fids))

        fids = np.array(fids)
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def make_pulse(I, Q):
        """Apply amplitude and slew constraints."""
        # Enforce amplitude bound
        amp = np.sqrt(I ** 2 + Q ** 2)
        max_amp = np.max(amp) if amp.size else 0.0
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I = I * factor
            Q = Q * factor

        # Enforce slew bound
        def check_slew(I, Q):
            slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
            slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
            m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
            return m <= slew_bound

        if not check_slew(I, Q):
            for _ in range(200):
                I = I * 0.95
                Q = Q * 0.95
                if check_slew(I, Q):
                    break

        return I, Q

    # Generate candidate pulses using different strategies
    best_I, best_Q = None, None
    best_score = float('inf')

    # Strategy 1: DRAG-style Gaussian
    sigma = T / 6.0
    center = T / 2.0

    def gaussian_env(center_shift=0.0, width_scale=1.0):
        sigma_eff = sigma * width_scale
        g = np.exp(-0.5 * ((t - center - center_shift) / sigma_eff) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma_eff) ** 2)
        g = g - edge_val
        g = np.maximum(g, 0.0)
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        return g

    # Strategy 2: Cosine window
    def cosine_env():
        g = 0.5 * (1 - np.cos(2 * np.pi * t / T))
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        return g

    # Strategy 3: Boxcar with smooth edges
    def boxcar_env(edge_frac=0.1):
        g = np.ones(N)
        edge_steps = int(N * edge_frac)
        for i in range(edge_steps):
            frac = (i + 1) / (edge_steps + 1)
            g[i] = 0.5 * (1 - np.cos(np.pi * frac))
            g[N - 1 - i] = 0.5 * (1 - np.cos(np.pi * frac))
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        return g

    candidates = []

    # DRAG pulses with various parameters
    for width_scale in [0.7, 0.8, 0.9, 1.0, 1.1]:
        for center_shift in [-1.0, 0.0, 1.0]:
            for drag_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                g = gaussian_env(center_shift, width_scale)
                g_deriv = np.gradient(g, dt)
                I = g.copy()
                Q = drag_alpha * g_deriv.copy()
                I, Q = make_pulse(I, Q)
                candidates.append((I, Q, f'gauss_w{width_scale}_c{center_shift}_d{drag_alpha}'))

    # Cosine window with DRAG
    for drag_alpha in [0.0, 0.1, 0.2, 0.3]:
        g = cosine_env()
        g_deriv = np.gradient(g, dt)
        I = g.copy()
        Q = drag_alpha * g_deriv.copy()
        I, Q = make_pulse(I, Q)
        candidates.append((I, Q, f'cos_d{drag_alpha}'))

    # Boxcar with DRAG
    for drag_alpha in [0.0, 0.1, 0.2]:
        g = boxcar_env()
        g_deriv = np.gradient(g, dt)
        I = g.copy()
        Q = drag_alpha * g_deriv.copy()
        I, Q = make_pulse(I, Q)
        candidates.append((I, Q, f'box_d{drag_alpha}'))

    # Evaluate all candidates
    for I, Q, name in candidates:
        if time.time() > deadline:
            break
        sc = score_pulse(I, Q)
        if sc < best_score:
            best_score = sc
            best_I, best_Q = I, Q

    # If no valid candidate found, use a simple DRAG pulse
    if best_I is None:
        g = gaussian_env()
        g_deriv = np.gradient(g, dt)
        I = g.copy()
        Q = 0.2 * g_deriv.copy()
        best_I, best_Q = make_pulse(I, Q)

    I, Q = best_I, best_Q

    # Final validity check
    amp = np.sqrt(I ** 2 + Q ** 2)
    assert np.all(amp <= amp_bound + 1e-10), "Amplitude constraint violated"

    def check_slew(I, Q):
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
        return m <= slew_bound

    assert check_slew(I, Q), "Slew rate constraint violated"

    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END
