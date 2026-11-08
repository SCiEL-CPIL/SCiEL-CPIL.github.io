import numpy as np
# Test Strassen
a,b,c,d = 1,2,3,4
e,f,g,h = 5,6,7,8
# Expected C
C00 = a*e + b*g  # 5+14=19
C01 = a*f + b*h  # 6+16=22
C10 = c*e + d*g  # 15+28=43
C11 = c*f + d*h  # 18+32=50

P1 = (a+d)*(e+h)  # 5*13=65
P2 = (c+d)*e      # 7*5=35
P3 = a*(f-h)      # 1*(-2)=-2
P4 = d*(g-e)      # 4*2=8
P5 = (a+b)*h      # 3*8=24
P6 = (c-a)*(e+f)  # 2*11=22
P7 = (b-d)*(g+h)  # -2*15=-30

# Try: C00 = P1 - P3 - P4 + P7 = 65+2-8-30=29 (wrong, need 19)
# Try: C00 = P1 + P4 - P3 - P7 = 65+8+2+30=105 (wrong)
# Try: C00 = P1 - P4 + P3 - P7 = 65-8-2+30=85 (wrong)
# ...
