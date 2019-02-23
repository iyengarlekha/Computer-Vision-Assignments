import glob
from PIL import Image
import random
import numpy as np

def rotate(image, rotation_angle = 25):
    image = image.rotate(random.uniform(-rotation_angle, rotation_angle))
    return image

def gaussian_noise(image, mean = 0., stddev = 1.):
    image = np.asarray(image)
    noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(image))
    image = np.add(image, noise)
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    image = Image.fromarray(image)
    return image

def find_coeffs(pa, pb):
	matrix = []
	for p1, p2 in zip(pa, pb):
		matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
		matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

	A = np.matrix(matrix, dtype=np.float)
	B = np.array(pb).reshape(8)

	res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
	return np.array(res).reshape(8)

def generate_random_shifts(img_size, factor = 17):
	w = img_size[0] / factor
	h = img_size[1] / factor
	shifts = []
	for s in range(0, 4):
		w_shift = (random.random() - 0.5) * w
		h_shift = (random.random() - 0.5) * h
		shifts.append((w_shift, h_shift))
	return shifts

def projection_tf(image):
	img_size = image.size
	w = img_size[0]
	h = img_size[1]
	shifts = generate_random_shifts(img_size)
	coeffs = find_coeffs(
		[(shifts[0][0], shifts[0][1]),
			(w + shifts[1][0], shifts[1][1]),
			(w + shifts[2][0], h + shifts[2][1]),
			(shifts[3][0], h + shifts[3][1])], [(0, 0), (w, 0), (w, h), (0, h)])
	return image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def create_new_image(image):
    image = projection_tf(image)
    image = gaussian_noise(image)
    image = rotate(image)
    return image

def main():

    classes = glob.glob("~/Downloads/all/train_images/*")

    # all the classses
    for class_id in classes:
        images = glob.glob(class_id + '/*')

        # all the images in a class
        no_of_images = len(images)

        # TODO: adjust the value of n according to no_of_images and total images required
        n = 100

        print("number of images in class " + repr(class_id) + " are " + repr(no_of_images))
        print("all the files will be augmented " + repr(n) + " times")

        for file_name  in images:
            # augment an image n times
            for i in range(n):
                if not "$" in file_name:
                    image = Image.open(file_name)
                    image = create_new_image(image)
                    image.save(file_name.replace('.ppm', '$' + repr(i) + '.ppm'))

if __name__ == '__main__':
    main()