import cv2
import numpy as np
from sklearn import cluster

class CloudMaskCreator:

    def __init__(self, image: str, num_clusters: int):

        self.clustered_image = self.image_clustering(image, num_clusters)
        self.mask = self.create_mask(self.clustered_image, image)

    @staticmethod
    def image_clustering(image: np.ndarray, num_clusters: int) -> np.ndarray:
        hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hue_channel = hls_image[:,:,0]

        h, w = hue_channel.shape

        image_cluster = hue_channel.reshape(-1, 1)
        image_cluster = np.nan_to_num(image_cluster, copy=True, nan=0)

        kmeans_cluster = cluster.KMeans(n_clusters=num_clusters, random_state=1).fit(image_cluster)
        clusters_centers = kmeans_cluster.cluster_centers_
        clusters_labels = kmeans_cluster.labels_

        clustered_image = clusters_centers[clusters_labels].reshape(h, w)

        return clustered_image

    @staticmethod
    def create_mask(clustered_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        original_image = np.uint8(original_image)

        np.place(original_image, original_image != 0, 255)

        np.place(original_image[:,:,0], clustered_image == np.unique(clustered_image)[0], 0)
        np.place(original_image[:,:,1], clustered_image == np.unique(clustered_image)[0], 0)
        np.place(original_image[:,:,2], clustered_image == np.unique(clustered_image)[0], 0)

        np.place(original_image[:,:,0], clustered_image != 0, 1)
        np.place(original_image[:,:,1], clustered_image != 0, 1)
        np.place(original_image[:,:,2], clustered_image != 0, 1)

        #original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        return original_image

    def get_mask(self):
        return self.mask