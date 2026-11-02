import numpy as np
import time
from itertools import combinations

N = 11

def get_optimal_center_and_S(points, theta):
    angles = [theta, theta + np.pi/3, theta + 2*np.pi/3]
    us = [np.array([np.cos(a), np.sin(a)]) for a in angles]
    
    A_mat = np.array(us)
    m_vec = np.zeros(3)
    W_max = 0
    for i, u in enumerate(us):
        projs = points @ u
        m_vec[i] = 0.5 * (np.max(projs) + np.min(projs))
        W_max = max(W_max, np.max(projs) - np.min(projs))
        
    # Solve for C
    C, residuals, rank, sv = np.linalg.lstsq(A_mat, m_vec, rcond=None)
    
    S = W_max / np.sqrt(3)
    return C, S

def min_enclosing_hexagon_side(points):
    min_S = float('inf')
    best_theta = 0.0
    best_C = np.zeros(2)
    
    thetas = np.linspace(0, np.pi/3, 180)
    
    for theta in thetas:
        C, S = get_optimal_center_and_S(points, theta)
        if S < min_S:
            min_S = S
            best_theta = theta
            best_C = C
            
    return min_S, best_theta, best_C

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    base = np.array([[0.0, 0.0]])
    for k in range(6):
        ang = k * np.pi / 3
        base = np.vstack([base, [np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang)]])
        
    gap_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]
    
    best_S = float('inf')
    best_points = None
    best_theta = 0.0
    best_C = np.zeros(2)
    
    for combo in combinations(range(6), 4):
        pts = base.copy()
        for idx in combo:
            ang = gap_angles[idx]
            pts = np.vstack([pts, [2.0*np.cos(ang), 2.0*np.sin(ang)]])
        
        S, theta, C = min_enclosing_hexagon_side(pts)
        if S < best_S:
            best_S = S
            best_points = pts
            best_theta = theta
            best_C = C
            
        if time.time() > deadline:
            break
            
    if best_points is None:
        best_points = base
        best_S, best_theta, best_C = min_enclosing_hexagon_side(base)
        
    inner_list = []
    for p in best_points:
        inner_list.append([float(p[0]), float(p[1]), 0.0])
        
    return inner_list, (float(best_C[0]), float(best_C[1])), float(best_S), float(np.degrees(best_theta))
