import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PCA():

    def __init__(self, x_data, k):
        self.x_data = x_data
        self.k = k

    def standartize(self):
        self.x_data_std = (self.x_data - np.mean(self.x_data, axis=0) ) / np.std(self.x_data, axis=0)
        return self.x_data_std
    
    def calc_covariance(self):
        self.covariance = 1/(self.x_data_std.shape[0]) * np.dot(np.transpose(self.x_data_std) , self.x_data_std)
        return self.covariance
    
    def calc_score(self):
        self.z = np.dot(self.x_data_std, np.transpose(self.pca))

    def calc_pc(self):
        self.standartize()
        self.calc_covariance()
        eigen_value, eigen_vector = np.linalg.eig(self.covariance) 
        eigen_value = eigen_value.real
        eigen_vector = eigen_vector.real
        self.pca = []
        for _ in range(self.k):
            idx = np.argmax(eigen_value)
            self.pca.append(eigen_vector[:, idx])
            eigen_value = np.delete(eigen_value, idx)
            eigen_vector= np.delete(eigen_vector, idx, axis=1)
        self.calc_score()
    
    def visualize(self, features):
        # Plot original data
        #plt.scatter(x_data[:, 0], x_data[:, 1], label='Original Data', color='blue')
        plt.scatter(self.z[:,0], self.z[:,1], label='PCA Data', color='blue')
        scale_factor = 8
        # Plot principal components as arrows
        for i in range(0, len(features)):
            plt.arrow(0, 0, self.pca[0][i] * scale_factor, self.pca[1][i] * scale_factor, color='red', width=0.01, head_width=0.05)
            plt.text(self.pca[0][i] * scale_factor, self.pca[1][i]* scale_factor, f'{features[i]}', fontsize=9, ha='center', va='center')


        plt.title('Biplot of PCA')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    ds = pd.read_csv('/home/bora/Desktop/Data/ProblemSet/Iris.csv')

    x_data = ds.iloc[:,1:-1].values
    x_data = x_data[~np.isnan(x_data).any(axis=1)]
    x_data = x_data[:, ~np.isnan(x_data).any(axis=0)]

    features = list(ds.columns)[1:-1]
    
    pca = PCA(x_data, 2)
    pca.calc_pc()
    pca.visualize(features)


            



