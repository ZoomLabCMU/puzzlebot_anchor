import math
import itertools
import casadi as ca
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def get_R(t):
    '''
    return SE2 rotation matrix from angle
    '''
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])

def get_g(x):
    '''
    x is a 1d vector
    '''
    g = np.eye(3, 3)
    g[0:2, 0:2] = get_R(x[2])
    g[0:2, 2] = x[0:2]
    return g

def wrap_pi(t):
    return ((t + np.pi) % (2*np.pi) - np.pi)

def dd_fx(t):
    '''
    differential drive dynamics update matrix
    '''
    N = len(t)
    F = np.zeros([3*N, 2*N])
    F[0::3, 0::2] = np.diag(np.cos(t))
    F[1::3, 0::2] = np.diag(np.sin(t))
    F[2::3, 1::2] = np.eye(N)
    return F

def get_cp_dis(x, cp):
    '''
    Distance between contact pairs
    x: 3*N vector,
    cp: dict {(id0, id1): 2-by-2 or 3-by-2 matrix}
    return: len(cp) vector of distance
    '''
    dis = []
    for (i0, i1) in cp:
        d0 = cp[(i0, i1)][0:2, 0]
        d1 = cp[(i0, i1)][0:2, 1]
        t0 = x[i0*3+2]
        t1 = x[i1*3+2]
        x_diff = x[i0*3:(i0*3+2)] - x[i1*3:(i1*3+2)]
        x_diff += get_R(t0).dot(d0) - get_R(t1).dot(d1)
        dis.append(np.linalg.norm(x_diff))
    return dis

def yaw_from_quaternion(quat):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = quat
    #  t0 = +2.0 * (w * x + y * z)
    #  t1 = +1.0 - 2.0 * (x * x + y * y)
    #  roll_x = math.atan2(t0, t1)

    #  t2 = +2.0 * (w * y - z * x)
    #  t2 = +1.0 if t2 > +1.0 else t2
    #  t2 = -1.0 if t2 < -1.0 else t2
    #  pitch_y = math.asin(t2)

    # simplified for speed
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    #  return roll_x, pitch_y, yaw_z
    return yaw_z

def body2world(x, pt):
    """
    x: size 3 vec of the current robot pose
    pt: 2(or 3)-by-M matrix, points on the body frame
    """
    assert(x.shape == (3,))
    assert(pt.shape[0] >= 2)
    pt = pt[0:2, :]
    R = get_R(x[2])
    cw = (R.dot(pt) + x[0:2, np.newaxis]).T
    return cw

def get_corners(x, L, margin=0):
    """
    x: 3 vec, L: length of robot body
    return: 4-by-2 matrix of the corner locations
    """
    assert(x.shape == (3,))
    l = [L/2 + margin, -L/2 - margin] 
    corners = list(itertools.product(l, l))
    c_arr = np.array(corners).T
    return body2world(x, c_arr)

def is_inside_poly(pt, poly):
    """
    pt: size 2 vec, the point to consider
    poly: the points of the polygon
    """
    assert(pt.size == 2)
    assert(poly.shape[1] == 2)
    if len(pt.shape) > 1:
        pt = pt.flatten()

    point = Point(pt[0], pt[1])
    polygon = Polygon(poly)
    return polygon.contains(point)

def is_inside_robot(pt, x, L, margin=0):
    """
    pt: size 2 vec, the point to consider
    x: 3 vec, L: length of robot body
    """
    cs = get_corners(x, L, margin=margin)
    return is_inside_poly(pt, cs)

def get_Ab_vwlim(vmax, wmax, N, double_int=True):
    """
    return the polygon from the 4 points
    [vmax, 0], [-vmax, 0], [0, wmax], [0, -wmax]
    """
    if not double_int:
        raise NotImplementedError("get_Ab is not implemented for single integrator yet")
    a = np.zeros([4, 5])
    vw_a = np.array(list(itertools.product([1/vmax, -1/vmax], 
                                            [1/wmax, -1/wmax])))
    a[:, 3:] = vw_a
    A = np.kron(np.eye(N), a)
    b = 1
    return A, b
    
def get_heading_err(x, cp):
    err = []
    for ids in cp:
        cp_d = cp[ids]
        if cp_d.shape[0] < 3: 
            err.append(0)
            continue
        angle_diff = (x[3*ids[0]+2] - x[3*ids[1]+2]) - (cp_d[2, 0] - cp_d[2, 1])
        err.append(angle_diff)
    if len(err) == 1: return err[0]
    return np.array(err)
