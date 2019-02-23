import numpy as np
import scipy.io as sio
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

def plot3(xs,ys,zs=0,mark="o",col="r"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=col, marker=mark)
    plt.show()

def centeroidnp(arr):
    length = arr.shape[1]
    sum_x = np.sum(arr[0, :])
    sum_y = np.sum(arr[1, :])
    return sum_x/length, sum_y/length

def main():
    input_file = "sfm_points.mat"

    sfm_data = sio.loadmat(input_file)

    centroid = np.zeros((2,1))

    for i in range(10):
        x,y = centeroidnp(sfm_data["image_points"][:, :, i])
        centroid = np.append(centroid, [[x],[y]], axis=1)

    centroid = centroid[:, 1:]

    W = np.zeros((20, 600))

    for i in range(10):
        temp = sfm_data["image_points"][:, :, i] - centroid[:, i].reshape(2,1)
        W[i] = temp[0, :]
        W[i+10] = temp[1, :]

    # print (W.shape)
    # print(W)
    U, D, Vt = np.linalg.svd(W)

    # print (U.shape)
    # print (D.shape)
    # print (Vt.shape)
    print ("t_i for first camera:", centroid[:,0])
    # print (U[:, :3])
    # print (D[:3])
    # Mi = U[:, :3].dot(np.diag(D[:3]))
    Mi = np.matmul(U[:, :3], np.diag(D[:3]))
    print ("Mi_shape :", Mi.shape)
    print ("Mi for first camera :", Mi[:2,:])

    points = Vt[:3]
    ax.plot3D(points[0], points[1], points[2], 'gray', marker="o")
    plt.show()

    print ("3d coordinates of first 10 world points")
    print (np.matrix.transpose(points[:, 0:10]))


    # print sfm_data["image_points"][:,:,1].shape

if __name__ == "__main__":
    main()

