import numpy as np
from dataclasses import dataclass, field


#taken from class provided meshutils
def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    Parameters
    ----------
    X : 2D numpy.array (dtype=float)
    vertex coordinates shape (3,Nvert)
    color : 2D numpy.array (dtype=float)
    vertex colors shape (3,Nvert)
    should be float in range (0..1)
    tri : 2D numpy.array (dtype=float)
    triangular faces shape (Ntri,3)
    filename : string
    filename to save to
    """
    f = open(filename,"w");
    f.write('ply\n');
    f.write('format ascii 1.0\n');
    f.write('element vertex %i\n' % X.shape[1]);
    f.write('property float x\n');
    f.write('property float y\n');
    f.write('property float z\n');
    f.write('property uchar red\n');
    f.write('property uchar green\n');
    f.write('property uchar blue\n');
    f.write('element face %d\n' % tri.shape[0]);
    f.write('property list uchar int vertex_indices\n');
    f.write('end_header\n');
    C = (255*color).astype('uint8')
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' %
        (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]));
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))
    f.close();



#taken from 117
@dataclass
class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 
    
    """

    f:float
    c:np.ndarray
    R:np.ndarray
    t:np.ndarray

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        #inverse is transpose
        pts3_mod = self.R.T @ (pts3 - self.t)
        pts3_mod = (self.f * (pts3_mod/pts3_mod[2]))[:2]
        pts2 = pts3_mod + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2


 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array of shape (6,) (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[:3] are the rotation angles, params[3:] are the translation

        """
        rx, ry, rz = params[:3]
        self.R = makerotation(rx, ry, rz)
        self.t = params[3:].reshape((3, 1))



#helpers, from duke notes

#skew
def cross(t):
    return np.array(((0, -t[2], t[1]),
                     (t[2], 0, -t[0]),
                     (-t[1], t[0], 0)))


#not the triangulate I'm used to, certainly
def triangulate(p, q, t, R):
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns'
    P = np.zeros((3, n))
    
    i, j, k = R[0], R[1], R[2]
    kt = np.dot(k, t)
    proj = np.vstack((i, j))
    projt = np.dot(proj, t)
    
    C = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0))).astype(float)
    c = np.zeros(4)
    
    for m in range(n):
        C[:2, 2] = -p[:2, m]
        C[2:, :] = np.outer(q[:2, m], k) - proj
        c[2:] = kt * q[:2, m] - projt
        
        x = np.linalg.lstsq(C, c, rcond=None)
        P[:, m] = x[0]
        
    Q = np.dot(R, P - np.outer(t, np.ones((1, n))))
    return P, Q



#eight point madness, taken from the duke notes
def LH(p, q):# -> tuple[ndarray[_AnyShape, dtype[Any]], Any, _Array[tuple[i...:
    # Number of point pairs
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns'
    
    # Transform images from 2D to 3D in the standard reference frame
    o = np.ones((1, n)).astype(float)
    p, q = np.concatenate((p, o)), np.concatenate((q, o))

    # Set up matrix A such that A*E.flatten() = 0, where E is the essential
    # matrix.
    # This system encodes the epipolar constraint q' * E * p = 0 for each of
    # the points p and q
    A = np.zeros((n, 9))
    for k in range(n):
        A[k, :] = np.outer(q[:, k], p[:, k]).flatten()        
    assert np.linalg.matrix_rank(A) >= 8, 'Insufficient rank for A'
    
    # The singular vector corresponding to the smallest singular value of A
    # is the arg min_{norm(e) = 1} A * e, and is the LSE estimate of E.flatten()
    _, _, VT = np.linalg.svd(A)
    E = np.reshape(VT[-1, :], (3, 3))
    
    # The two possible translation vectors are t and -t, where t is a unit
    # vector in the null space of E. The vector t (or -t) is also the
    # second epipole of the camera pair
    _, _, VET = np.linalg.svd(E)
    t = VET[2, :]
    
    # The cross-product matrix for vector t
    tx = cross(t)
    
    # Two rotation matrix choices are found by solving the Procrustes problem
    # for the rows of E and tx, and allowing for the ambiguity resulting
    # from the sign of the null-space vectors (both E and tx are rank 2).
    # These two choices are independent of the sign of t, because both E
    # and -E are essential matrices
    UF, _, VFT = np.linalg.svd(np.dot(E, tx))
    R1 = np.dot(UF, VFT)
    R1 *= np.linalg.det(R1)
    UF[:, 2] = -UF[:, 2]
    R2 = np.dot(UF, VFT)
    R2 *= np.linalg.det(R2)
    
    # Combine the two sign options for t with the two choices for R
    tList = [t, t, -t, -t]
    RList = [R1, R2, R1, R2]
    
    # Pick the combination of t and R that yields the greatest number of
    # positive depth (Z) values in the structure results for the frames of
    # reference of both cameras. Ideally, all depth values should be positive
    P, Q, npdMax = [], [], -1
    for k in range(4):
        tt, RR = tList[k], RList[k]
        PP, QQ = triangulate(p, q, tt, RR)
        npd = np.sum(np.logical_and(PP[2, :] > 0, QQ[2, :] > 0))
        if npd > npdMax:
            t, R, P, Q, npdMax = tt, RR, PP, QQ, npd
            
    return t, R, P, Q

#given LH output, deriving my camL and camR
#i really should have tested this one thoroughly...
def cameras(t, R, f, c):
    #setting camL to the defaults
    R1 = np.eye(3) #eye-dentity (ba dum tss)
    t1 = np.zeros((3, 1))
    camL = Camera(f=f, c=c, R=R1, t=t1)

    #camR translated and rotated by what is specified in LH
    t = t.reshape(3, 1)
    R2 = R
    t2 = -R.T @ t #...most likely cause of my problems
    camR = Camera(f=f, c=c, R=R2, t=t2)

    return camL, camR


def actual_triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system
    
    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera
    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera
    camL : Camera
        The first "left" camera view
    camR : Camera
        The second "right" camera view
        
    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates
    """

    assert(pts2L.shape[0] == 2)
    assert(pts2R.shape[0] == 2)

    pts3 = np.zeros([3, pts2L.shape[1]])

    xL_cam = (pts2L[0, :] - camL.c[0])/camL.f
    yL_cam = (pts2L[1, :] - camL.c[1])/camL.f
    q_L = np.vstack([xL_cam, yL_cam, np.ones(xL_cam.shape)])

    xR_cam = (pts2R[0, :] - camR.c[0])/camR.f
    yR_cam = (pts2R[1, :] - camR.c[1])/camR.f
    q_R = np.vstack([xR_cam, yR_cam, np.ones(xR_cam.shape)])

    b = camR.t - camL.t
    for i in range(len(pts2L[0])):
        q_L_i = q_L[:, i].reshape((3,1))
        q_R_i = q_R[:, i].reshape((3,1))
        A_i = np.hstack([camL.R@q_L_i, -1*camR.R@q_R_i])
        u, *_ = np.linalg.lstsq(A_i, b)
        z_L, z_R = u[0], u[1]

        P_L = z_L * q_L_i 
        P_1 = (camL.R @ P_L) + camL.t
        
        P_R = z_R * q_R_i 
        P_2 = (camR.R @ P_R) + camR.t

        pts3[:, i] = ((P_1 + P_2)/2).T

    assert(pts3.shape[0] == 3)

    return pts3
