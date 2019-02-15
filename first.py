import numpy as np
import matplotlib.pyplot as plt


nlong = np.array([1, -0.5])
n = nlong/np.sqrt(np.dot(nlong, nlong))
r0 = np.array([3, 3])
p0 = np.array([1, 6])
elong = np.array([1, -1])
n1 = 1
n2 = 1.5
e = elong/np.sqrt(np.dot(elong, elong))

def angle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2)/(np.dot(vec1, vec1)*np.dot(vec2, vec2)))

def intersectionPoint():
    t = np.dot(n, r0-p0)/np.dot(n, e)
    print(t)
    return t

def main():
    t = intersectionPoint()
    r = p0 + e*t
    print(r)

    erefl = reflection()
    erefr = refraction()

    x1, y1 = [(r + 10*getOrthogonal())[0], (r - 10*getOrthogonal())[0]], [(r + 10 * getOrthogonal())[1], (r - 10 * getOrthogonal())[1]]
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


def getOrthogonal():
    return np.array([n[1], -n[0]])

def reflection():
    erefl = e - 2*np.dot(e, n)*(n)
    print(erefl)
    return erefl

def refraction():
    erefr = (n1*e - n*(n1*np.dot(e, n) - n2*np.sqrt(1 - ((n1**2)/(n2**2))*(1 - np.dot(e, n)**2))))/n2
    return erefr


main()