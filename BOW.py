import numpy as np 
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import inv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# generate sift features for image
def generate_sift_features(data):
	sift = cv2.xfeatures2d.SIFT_create()
	sift_feature = []
	img_num = len(data)
	for i in range (img_num):
		kp, des = sift.detectAndCompute(data[i],None)
		sift_feature.append(des) 
	return sift_feature

# combine all features together
def combine_feature(data):
	img_num = len(data)
	for i in range (img_num):
		if i == 0:
			sift_feature_all = data[i]
		else:
			sift_feature_all = np.concatenate((sift_feature_all, data[i]), axis = 0)
	return sift_feature_all

# Using PCA to complete dimension reduction
def generate_pcaSift_feature(sift_feature_all):
	pca = PCA(n_components = dimension)
	pca.fit(sift_feature_all)
	pca_feature = pca.transform(sift_feature_all)
	return pca_feature

# Generate feature vector for every image
def generate_feature_vector(sift_feature, kmeans_labels):
	img_num = len(sift_feature)
	feature_vector = []
	cnt = 0
	for i in range (img_num):
		features_num_per_img = sift_feature[i].shape[0]
		temp = np.zeros(k_num)
		for k in range (features_num_per_img):

			label = kmeans_labels[cnt + k]
			temp[label] += 1
		feature_vector.append(temp)
		cnt += features_num_per_img
	return feature_vector

# calculate the closest center for every feature in testing images 
def calculate_label(input_feature, centroids):
	num = input_feature.shape[0]
	kmeans_label = np.zeros(num, dtype = np.uint)
	for i in range(num):
		dist = np.zeros(k_num) 
		for j in range(k_num):
			dist[j] = np.linalg.norm(input_feature[i] - centroids[j])
		min_idx = np.argmin(dist)
		kmeans_label[i] =int(min_idx)
	return kmeans_label


#------------------------Step 1: load train data an dtest data-----------------------------
n_class = 5
train_num = 20
test_num = 10
k_num = 80
dimension = 25

name = ['airplanes', 'car_side', 'electric_guitar', 'faces', 'Motorbikes']
train_images = []
test_images = []
train_label = []
test_label = []


for j in range (5):
	for i in range (20):
		train_img = cv2.imread('./HW5_Data/' + name[j] + '/train/' + str(i+1) + '.jpg',0)	
		train_images.append(train_img)
		train_label.append(j)
	for i in range (10):
		test_img = cv2.imread('./HW5_Data/' + name[j] + '/test/' + str(i+1) + '.jpg',0)
		test_images.append(test_img)
		test_label.append(j)

# #------------------------Step 2: Extract features -----------------------------

sift_feature_train = generate_sift_features(train_images)
sift_feature_test = generate_sift_features(test_images)
print 'length of sift_feature_train', len(sift_feature_train)
print 'length of sift_feature_test', len(sift_feature_test)


# # # #------------------------Step 3: PCA reduction -----------------------------
sift_feature_train_all = combine_feature(sift_feature_train)
sift_feature_test_all = combine_feature(sift_feature_test)
sift_feature_all = np.concatenate((sift_feature_train_all,sift_feature_test_all),axis = 0)
print 'sift_feature_train_all', sift_feature_train_all.shape
print 'sift_feature_test_all', sift_feature_test_all.shape
print 'sift_feature_all', sift_feature_all.shape

trian_feature_num = sift_feature_train_all.shape[0]

sift_pca = generate_pcaSift_feature(sift_feature_all)
sift_pca_train = sift_pca[:trian_feature_num,:] 
sift_pca_test = sift_pca[trian_feature_num:,:]

print 'sift_pca_train', sift_pca_train.shape
print 'sift_pca_test', sift_pca_test.shape


# # # # #------------------------Step 4: Kmeans -----------------------------
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_PP_CENTERS
# Apply KMeans
compactness,kmeans_labels_train,centers = cv2.kmeans(sift_pca_train,k_num,None,criteria,10,flags)

print 'centers', centers.shape
print 'labels', kmeans_labels_train.shape


# # # # #------------------------Step 5: feature vector -----------------------------

feature_vector_train = generate_feature_vector(sift_feature_train, kmeans_labels_train)
kmeans_labels_test = calculate_label(sift_pca_test, centers)
feature_vector_test = generate_feature_vector(sift_feature_test, kmeans_labels_test)

print 'len(feature_vector_train)', feature_vector_train[0]
# print 'len(feature_vector_test)',feature_vector_test

# # # # # #--------------------Step 6: Testing and error measure-----------------------------

neigh = KNeighborsClassifier(n_neighbors=13, weights = 'distance')
# neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(feature_vector_train, train_label) 
predict_label =  neigh.predict(feature_vector_test)
print 'k = ', k_num
print 'd = ', dimension
print predict_label

correct_num = np.count_nonzero(predict_label == test_label)
print 'correct_num', correct_num



















