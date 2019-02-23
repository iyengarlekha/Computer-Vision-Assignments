import numpy as np
from scipy.linalg import null_space


world_file = "world.txt"
image_file = "image.txt"

x_world = np.loadtxt(world_file)
x_img = np.loadtxt(image_file)

augment_array = np.ones((1,10))

ax_world = np.append(x_world, augment_array, axis=0)
ax_img = np.append(x_img, augment_array, axis=0)

print("World: " , ax_world)
print("Image: " , ax_img)


zeros_4d = np.zeros((4, 1))
A = np.zeros((1,12))

for i in range(10):
    xi = ax_img[:, i].reshape(3,1)
    yi = xi[1]
    wi = xi[2]
    xi = xi[0]
    Xi = ax_world[:, i].reshape(4,1)

    A1 = np.concatenate((zeros_4d.T, -wi * Xi.T, yi * Xi.T), axis=1)
    A2 = np.concatenate((wi *  Xi.T, zeros_4d.T, -xi * Xi.T), axis=1)

    A = np.append(A, A1, axis=0)
    A = np.append(A, A2, axis=0)


A = A[1:,:]

# print(A.shape)
# print("A", A);

# P matrix
_, _, v = np.linalg.svd(A)
P = v[-1, :].reshape((3, 4))
print ("Camera Matrix P:")
print (P)

_, _, v = np.linalg.svd(P)
C = v[-1, :]
C = np.array([float(C[0] / C[-1]), float(C[1] / C[-1]), float(C[2] / C[-1])])
print ("Projection Center C:")
print (C)

r, k = np.linalg.qr(P.T)
R = -r[:-1,:].T
_rc = r[-1,:].T
c = np.linalg.solve(R, _rc)
print("K :\n", k)
print("R :\n", R)
print("C by QRDecomposition : ")
print(c)









