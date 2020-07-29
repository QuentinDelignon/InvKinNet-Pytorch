import os
import sys
import numpy as np
from numpy import pi,sin,cos
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

nb_points = 50 #nombre de points par dimension

def getPosition(q):
    """
    Retourne la position de la pointe avec les angles actuels
    """
    a = [0,0.5,0.5,0,0]
    d = [0.2,0,0,0,0.2]
    al = [-pi/2,0,0,pi/2,0]
    pos = np.array([cos(q[0])*(-d[4]*sin(q[1]+q[2]+q[3])+a[2]*cos(q[1]+q[2])+a[1]*cos(q[1])),
                    sin(q[0])*(-d[4]*sin(q[1]+q[2]+q[3])+a[2]*cos(q[1]+q[2])+a[1]*cos(q[1])),
                    d[0]-a[1]*sin(q[1])-a[2]*cos(q[1]+q[2])-d[4]*cos(q[1]+q[2]+q[3])])
    return pos

from numpy import cos,sin
from scipy.spatial.transform import Rotation as R
def getRot(q):
  c1 = cos(q[0]) ; c234 = cos(q[1]+q[2]+q[3]) ; c5 = cos(q[4])
  s1 = sin(q[0]) ; s234 = sin(q[1]+q[2]+q[3]) ; s5 = sin(q[4])
  mat = np.array([[c1*c234*c5+s1*s5, -c1*c234*s5+s1*c5, -c1*s234],
                  [c1*c234*c5-s1*s5, -s1*c234*s5-c1*c5, -s1*s234],
                  [-s234*c5,        s234*s5,            -c234]])
  rot = R.from_matrix(mat)
  quat = rot.as_quat()
  print(quat.shape)
  angle = 2*np.arccos((mat[0,0]+mat[1,1]+mat[2,2])/2)
  angle = (angle+2*np.pi)%(2*np.pi)
  norm = np.sqrt((mat[2,1]-mat[1,2])**2+(mat[0,2]-mat[2,0])**2+(mat[1,0]-mat[0,1])**2)
  ux = (mat[2,1]-mat[1,2])/norm
  uy = (mat[0,2]-mat[2,0])/norm
  uz = (mat[1,0]-mat[0,1])/norm
  return np.array([ux,uy,uz,angle])


def vec2string(mat):
    s = ''
    for i in mat:
        #print('value:',i)
        s += str(i)+','
    return s

values = np.linspace(-1,1,nb_points)
zvalues = np.linspace(0,1,nb_points)
f = open("C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\inverse_kin\\data_val.txt","w")
for x in tqdm(values):
    for y in values:
        for z in zvalues:
            f.write(vec2string([x,y,z])+'\n')
f.close()
