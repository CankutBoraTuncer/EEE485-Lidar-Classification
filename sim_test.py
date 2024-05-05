#!/usr/bin/env python3
import pickle
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np


class ModelTest:
    def __init__(self, model, scaler):
        with open(model, 'rb') as input_file:
            self.model = pickle.load(input_file)

        with open(scaler, 'rb') as input_file:
            self.scaler = pickle.load(input_file)

        self.classification = {0: "Room", 1: "Corridor", 2: "Doorway", 3: "Hall"}

    def classify_lidar_data_decision_tree(self, msg):
        data = np.multiply(np.array(msg.ranges), 1000)
        data[np.isinf(data)] = 16000
        msg_scaled = self.scaler.transform(data.reshape(1, 360))
        prediction = self.model.predict(msg_scaled.reshape(360, 1), self.model.root_branch)
        print("Prediction: ", self.classification[prediction])
    
    def classify_lidar_data_knn(self, msg):
        data = np.multiply(np.array(msg.ranges), 1000)
        data[np.isinf(data)] = 16000
        msg_scaled = self.scaler.transform(data.reshape(1, 360))
        prediction = self.model.predict(msg_scaled)
        print("Prediction: ", self.classification[prediction[0]])

    def classify_lidar_data_svm(self, msg):
        data = np.multiply(np.array(msg.ranges), 1000)
        data[np.isinf(data)] = 16000
        msg_scaled = self.scaler.transform(data.reshape(1, 360))
        prediction = self.model.predict(msg_scaled)
        print("Prediction: ", self.classification[prediction[0]])

if __name__ == '__main__':
    model_path = "knn_raw.pkl"
    scaler_path = "standart_scaler.pkl"
    model_test = ModelTest(model_path, scaler_path)
    rospy.init_node('simulation_test', anonymous=True)
    rospy.Subscriber("/scan", LaserScan, model_test.classify_lidar_data_knn)
    rospy.spin()
