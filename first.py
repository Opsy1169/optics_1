import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse




def angle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2)/(np.dot(vec1, vec1)*np.dot(vec2, vec2)))

def intersectionPoint(n, r0, p0, e):
    t = np.dot(n, r0-p0)/np.dot(n, e)
    print(t)
    return t

def plane():
    nlong = np.array([1, -0.5])
    n = nlong / np.sqrt(np.dot(nlong, nlong))
    r0 = np.array([3, 3])
    p0 = np.array([1, 6])
    elong = np.array([1, -1])
    n1 = 1
    n2 = 1.5
    e = elong / np.sqrt(np.dot(elong, elong))
    #############
    t = intersectionPoint(n, r0, p0, e)
    r = p0 + e*t
    print(r)

    erefl = reflection(e, n)
    erefr = refraction(e, n, n1, n2)

    x1, y1 = [(r + 10*getOrthogonal(n))[0], (r - 10*getOrthogonal(n))[0]], [(r + 10 * getOrthogonal(n))[1], (r - 10 * getOrthogonal(n))[1]]
    rayx, rayy = [p0[0], r[0]], [p0[1], r[1]]
    rayreflx, rayrefly = [r[0], (r+erefl*5)[0]], [r[1], (r+erefl*5)[1]]
    rayrefrx, rayrefry =  [r[0], (r+erefr*5)[0]], [r[1], (r+erefr*5)[1]]
    normalx, normaly = [r[0], (r  - n)[0]], [r[1], (r - n)[1]]
    angleray = angle(-1*n, e)
    anglerefl = angle( erefl, -1*n)
    print("angle ", angleray, anglerefl)
    # print(np.sin(angleray)/np.sin(anglerefr))
    # x1, y1 = [3, 3], [-6, 14]
    plt.plot(x1, y1, rayx, rayy, rayreflx, rayrefly, rayrefrx, rayrefry, normalx, normaly)
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.legend(['плоскость', 'падающий', 'отраженный', 'преломленный', 'нормаль'])
    
    plt.show()

def sphere():
    p0 = np.array([5, 5])
    r0 = np.array([1, 2])
    elong = np.array([2, 1])
    e = elong / np.sqrt(np.dot(elong, elong))
    N1 = 1
    N2 = 1.5
    R = 2
    t1, t2 = intersectionPointSphere(r0, p0, e, R)
    print(t1)
    print(t2)
    r1 = r0 + e*-t1
    r2 = r0 + e*-t2
    ax = plt.gca()
    circle1 = plt.Circle(p0, R, fill=False)
    ax.add_patch(circle1)
    plt.gca().set_aspect("equal")
    # plt.axis('scaled')
    n1 = -(r1 - p0)/np.sqrt(np.dot(r1-p0, r1-p0))
    n2 = (r2 - p0)/np.sqrt(np.dot(r2-p0, r2-p0))
    erfl1 = -reflection(e, n1)
    erfl2 = -reflection(e, n2)
    erfr1 = -refraction(e, n1, N1, N2)
    erfr2 = -refraction(e, n2, N2, N1)
    print(n2)
    print(n1)
    plt.plot( [r0[0], r1[0]], [r0[1], r1[1]], [r1[0], r1[0] - n1[0]], [r1[1], r1[1] - n1[1]], [r0[0], r2[0]], [r0[1], r2[1]], [r2[0], r2[0]-n2[0]], [r2[1], r2[1]-n2[1]])
    plt.plot([r1[0], r1[0] - erfl1[0]], [r1[1], r1[1] - erfl1[1]], [r2[0], r2[0] - erfl2[0]], [r2[1], r2[1] - erfl2[1]])
    plt.plot([r1[0], r1[0] - erfr1[0]], [r1[1], r1[1] - erfr1[1]], [r2[0], r2[0] - erfr2[0]], [r2[1], r2[1] - erfr2[1]])
    plt.show()

    # plt.gcf().gca().add_artist(circle1)

def intersectionPointSphere(r0, p0, e, R):
    t1 = (np.dot(r0-p0, e)+np.sqrt((np.dot(r0 - p0, e))**2 - np.dot(e, e)*(np.dot(r0 - p0, r0 - p0) - R**2)))/np.dot(e, e)
    t2 = (np.dot(r0 - p0, e) - np.sqrt((np.dot(r0 - p0, e)) ** 2 - np.dot(e, e) * (np.dot(r0 - p0, r0 - p0) - R ** 2))) / np.dot(e, e)
    return t1, t2

def getOrthogonal(n):
    return np.array([n[1], -n[0]])

def reflection(e, n):
    erefl = e - 2*np.dot(e, n)*(n)
    print(erefl)
    return erefl

def refraction(e, n, n1, n2):
    erefr = (n1*e - n*(n1*np.dot(e, n) - n2*np.sqrt(1 - ((n1**2)/(n2**2))*(1 - np.dot(e, n)**2))))/n2
    return erefr

def ellipse():
    p0 = np.array([1, 2])
    r0 = np.array([-1, 0])
    elong = np.array([1, 2])
    e = elong / np.sqrt(np.dot(elong, elong))
    N1 = 1
    N2 = 1.5
    a2 = 3
    b2 = 2
    matrix = np.array([[b2, 0], [0, a2]])
    ax = plt.gca()
    circle1 = Ellipse((p0[0], p0[1]), 2*a2, 2*b2,  fill=False)
    ax.add_patch(circle1)
    plt.gca().set_aspect("equal")
    # plt.axis('scaled')
    plt.grid()
    t = intersectionPointEllipse(matrix, r0, p0, e)
    print(t)
    r1 = r0 + e * t[0]
    r2 = r0 + e*t[1]
    a = ellipsnorm(r2[0], r2[1], a2, b2)
    a = np.arctan(a)
    a = np.array([np.cos(a), np.sin(a)])
    erefl = -reflection(e, a)
    erefr = -refraction(e, a, N2, N1)
    print(erefl)
    plt.plot([r0[0], r2[0]], [r0[1], r2[1]], [r2[0], r2[0]-a[0]], [r2[1], r2[1]-a[1]], [r2[0], r2[0]-erefl[0]], [r2[1], r2[1]-erefl[1]])
    plt.plot([r2[0], r2[0]-erefr[0]], [r2[1], r2[1]-erefr[1]])
    plt.show()


def ellipsnorm(x0, y0, a, b):
    return -(x0*(b**2))/(y0*(a**2))


def intersectionPointEllipse(M, r, p, e):
    mr = np.dot(M, r - p)
    print(r)
    print(p)
    print(r-p)
    print(mr)
    me = np.dot(M, e)
    a = np.dot(np.dot(M, e), np.dot(M , e))
    b = 2*np.dot(np.dot(M , e), np.dot(M, r-p))
    c = np.dot(np.dot(M, r-p), np.dot(M, r-p)) - (M[0][0]*M[1][1])**2
    print(a, b, c)
    # print(np.roots([a, b, c]))
    # print(mr)
    # print(np.dot(me, me))
    # print(np.dot(me, mr))
    # print((np.dot(me, mr)**2 - (np.dot(me, me))*(np.dot(mr, mr)) - M[1][1]))
    # shit1 =( np.sqrt((np.dot(me, mr)**2 - (np.dot(me, me))*(np.dot(mr, mr)) - M[1][1]**2)))
    # print(shit1)
    # t1 = -((np.dot(np.dot(M, e), np.dot(M, r-p)))
    #     + np.sqrt((np.dot(np.dot(M, e), np.dot(M, r-p)))**2 - (np.dot(np.dot(M, e), np.dot(M, e)))*(np.dot(np.dot(M, r-p), np.dot(M, r-p))) - M[1][1]**2))\
    #     /np.dot(np.dot(M, e), np.dot(M, e))
    # return t1
    return np.roots([a, b, c])

# sphere()
# plane()
ellipse()