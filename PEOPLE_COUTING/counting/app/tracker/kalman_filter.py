"""8-dimensional constant-velocity Kalman filter for xyah bounding boxes."""
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.motion = np.eye(8, dtype=np.float64)
        self.motion[:4, 4:] = np.eye(4)
        self.measurement = np.eye(4, 8, dtype=np.float64)

    @staticmethod
    def to_xyah(box):
        x, y, w, h = map(float, box); h = max(h, 1.0)
        return np.array([x+w/2, y+h/2, max(w, 1.0)/h, h], dtype=np.float64)

    @staticmethod
    def to_xywh(state):
        cx, cy, aspect, height = state[:4]; h=max(float(height),1.0); w=max(float(aspect)*h,1.0)
        return (float(cx-w/2), float(cy-h/2), w, h)

    def initiate(self, box):
        mean=np.r_[self.to_xyah(box), np.zeros(4, dtype=np.float64)]; h=max(mean[3],1.0)
        std=np.array([h*.05,h*.05,.01,h*.05,h*.00625,h*.00625,1e-4,h*.00625])
        return mean, np.diag(std*std)

    def predict(self, mean, covariance, dt=1.0):
        motion=self.motion.copy(); motion[:4,4:]*=max(float(dt),0.0); h=max(mean[3],1.0)
        std=np.array([h*.05,h*.05,.01,h*.05,h*.00625,h*.00625,1e-4,h*.00625])
        return motion@mean, motion@covariance@motion.T+np.diag(std*std)

    def update(self, mean, covariance, box):
        projected=self.measurement@mean; h=max(projected[3],1.0)
        std=np.array([h*.1,h*.1,.1,h*.1]); innovation_cov=self.measurement@covariance@self.measurement.T+np.diag(std*std)
        gain=covariance@self.measurement.T@np.linalg.inv(innovation_cov)
        new_mean=mean+gain@(self.to_xyah(box)-projected)
        new_cov=covariance-gain@self.measurement@covariance
        return new_mean,(new_cov+new_cov.T)*.5
