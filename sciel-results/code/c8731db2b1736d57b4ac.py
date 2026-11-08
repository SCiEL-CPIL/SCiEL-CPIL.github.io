"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen. U:4x7, V:4x7, W:4x7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        # M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        # M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        
        # C00 = M1+M4-M5+M7
        W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
        # C01 = M3+M5
        W[1,2]=1; W[1,4]=1
        # C10 = M2+M4
        W[2,1]=1; W[2,3]=1
        # C11 = M1-M2+M3+M6
        W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
        
        return U, V, W
    
    def schoolbook(n_, m_, p_):
        rank = n_*m_*p_
        U = np.zeros((n_*m_, rank))
        V = np.zeros((m_*p_, rank))
        W = np.zeros((p_*n_, rank))
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i*m_+j, col]=1
                    V[j*p_+k, col]=1
                    W[k*n_+i, col]=1
                    col+=1
        return U, V, W
    
    rank_total = 56
    Uf = np.zeros((12, rank_total))
    Vf = np.zeros((20, rank_total))
    Wf = np.zeros((15, rank_total))
    col = 0
    
    # 2x4x5 block (A rows 0-1)
    # 2x4x2: two 2x2x2 Strassen blocks
    Us1,Vs1,Ws1 = strassen_2x2x2()
    Uf[0:4,col:col+7]=Us1
    Vf[0:4,col:col+7]=Vs1
    Wf[0:4,col:col+7]=Ws1
    col+=7
    
    Us2,Vs2,Ws2 = strassen_2x2x2()
    Uf[2:4,col:col+7]=Us2[0:2,:]
    Uf[6:8,col:col+7]=Us2[2:4,:]
    Vf[10:12,col:col+7]=Vs2[0:2,:]
    Vf[15:17,col:col+7]=Vs2[2:4,:]
    Wf[0:4,col:col+7]=Ws2
    col+=7
    
    # 2x4x3: schoolbook 2x2x1 + Strassen 2x2x2 + schoolbook 2x2x1 + Strassen 2x2x2
    Usc1,Vsc1,Wsc1 = schoolbook(2,2,1)
    Uf[0:4,col:col+4]=Usc1
    Vf[2,col:col+4]=Vsc1[0,:]
    Vf[7,col:col+4]=Vsc1[1,:]
    Wf[6:8,col:col+4]=Wsc1
    col+=4
    
    Us3,Vs3,Ws3 = strassen_2x2x2()
    Uf[0:4,col:col+7]=Us3
    Vf[3:5,col:col+7]=Vs3[0:2,:]
    Vf[8:10,col:col+7]=Vs3[2:4,:]
    Wf[9:11,col:col+7]=Ws3[0:2,:]
    Wf[12:14,col:col+7]=Ws3[2:4,:]
    col+=7
    
    Usc2,Vsc2,Wsc2 = schoolbook(2,2,1)
    Uf[2:4,col:col+4]=Usc2[0:2,:]
    Uf[6:8,col:col+4]=Usc2[2:4,:]
    Vf[12,col:col+4]=Vsc2[0,:]
    Vf[17,col:col+4]=Vsc2[1,:]
    Wf[6:8,col:col+4]=Wsc2
    col+=4
    
    Us4,Vs4,Ws4 = strassen_2x2x2()
    Uf[2:4,col:col+7]=Us4[0:2,:]
    Uf[6:8,col:col+7]=Us4[2:4,:]
    Vf[13:15,col:col+7]=Vs4[0:2,:]
    Vf[18:20,col:col+7]=Vs4[2:4,:]
    Wf[9:11,col:col+7]=Ws4[0:2,:]
    Wf[12:14,col:col+7]=Ws4[2:4,:]
    col+=7
    
    # 1x4x5 block (A row 2)
    Usb,Vsb,Wsb = schoolbook(1,4,5)
    Uf[8:12,col:col+20]=Usb
    Vf[0:20,col:col+20]=Vsb
    for k in range(5):
        Wf[k*3+2,col:col+20]=Wsb[k,:]
    col+=20
    
    assert col == rank_total
    
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
