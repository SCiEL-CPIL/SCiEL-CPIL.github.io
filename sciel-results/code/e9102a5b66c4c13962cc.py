import numpy as np
import time

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _liouvillian(I_val, Q_val, detuning, scale, a, adag, n):
    I3 = np.eye(3, dtype=complex)
    H = detuning * n + 0.5 * (-2*np.pi*0.25) * (n @ (n - I3))
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    H_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    L1 = a / np.sqrt(25000.0)
    L2 = n * np.sqrt(2.0 / 40000.0)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    return H_super + L1_super + L2_super

def _simulate_pulse(IQ, detuning, scale, a, adag, n):
    I3 = np.eye(3, dtype=complex)
    X_gate = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
    states = [
        np.array([1,0,0], dtype=complex),
        np.array([0,1,0], dtype=complex),
        (np.array([1,1,0], dtype=complex))/np.sqrt(2),
        (np.array([1,-1,0], dtype=complex))/np.sqrt(2),
        (np.array([1,1j,0], dtype=complex))/np.sqrt(2),
        (np.array([1,-1j,0], dtype=complex))/np.sqrt(2)
    ]
    from scipy.linalg import expm
    supers = [expm(_liouvillian(IQ[i,0], IQ[i,1], detuning, scale, a, adag, n) * 0.5) for i in range(80)]
    fids = []
    for psi in states:
        rho = np.outer(psi, psi.conj()).flatten()
        for s in supers:
            rho = s @ rho
        rho = rho.reshape(3,3)
        psi_t = X_gate @ psi
        rho_t = np.outer(psi_t, psi_t.conj())
        fids.append(np.real(np.trace(rho_t @ rho)))
    return np.mean(fids)

def _score(IQ):
    a, adag, n = _build_matrices()
    infs = []
    for det in [-2*np.pi*0.002, 0, 2*np.pi*0.002]:
        for sc in [0.97, 1.0, 1.03]:
            fid = _simulate_pulse(IQ, det, sc, a, adag, n)
            infs.append(1 - fid)
    return np.mean(infs) + np.max(infs)

def _valid(IQ):
    amps = np.sqrt(IQ[:,0]**2 + IQ[:,1]**2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    for i in range(80):
        prev = IQ[i-1] if i > 0 else np.array([0,0])
        if np.linalg.norm(IQ[i] - prev) / 0.5 > 0.12 + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) / 0.5 > 0.12 + 1e-12:
        return False
    return True

def _make_pulse(sigma_f, drag, amp):
    t = np.arange(80) * 0.5
    sigma = (40/6) * sigma_f
    env = np.exp(-(t-20)**2 / (2*sigma**2))
    env = np.maximum(env - env[0] - env[-1], 0)
    area = np.sum(env) * 0.5
    if area > 0:
        env *= np.pi / area
    env *= amp
    I = env
    Q = drag * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    # Enforce constraints
    max_amp = np.max(np.sqrt(IQ[:,0]**2 + IQ[:,1]**2))
    if max_amp > 0.4:
        IQ *= 0.4 / max_amp * 0.99
    for i in range(80):
        prev = IQ[i-1] if i > 0 else np.array([0,0])
        diff = np.linalg.norm(IQ[i] - prev)
        max_d = 0.12 * 0.5 * 0.99
        if diff > max_d:
            IQ[i] = prev + (max_d/diff) * (IQ[i] - prev)
    if np.linalg.norm(IQ[-1]) > 0.12 * 0.5 * 0.99:
        IQ[-1] *= 0.12 * 0.5 * 0.99 / np.linalg.norm(IQ[-1])
    return IQ

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = 320
    best_IQ = _make_pulse(1.0, -0.3, 1.0)
    best_s = _score(best_IQ)
    # Coarse grid
    for sf in [0.9, 1.0, 1.1]:
        for dc in [-0.5, -0.3, -0.1, 0.1, 0.3]:
            if time.time() > deadline:
                break
            iq = _make_pulse(sf, dc, 1.0)
            if _valid(iq):
                s = _score(iq)
                if s < best_s:
                    best_s = s
                    best_IQ = iq
    # Fine grid
    for sf in np.linspace(0.8, 1.2, 5):
        for dc in np.linspace(-0.6, 0.4, 9):
            if time.time() > deadline:
                break
            iq = _make_pulse(sf, dc, 1.0)
            if _valid(iq):
                s = _score(iq)
                if s < best_s:
                    best_s = s
                    best_IQ = iq
    return best_IQ
