import imageio
import os
import numpy as np
import cv2

class VideoRecorder(object):
    def __init__(self, dir_name, height=448, width=448, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.height = 128
        self.width = 128
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled
    
    def modify_frame(self, frame, episode_reward):
        frame = frame[0][6:9]
        frame = frame.detach().cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.resize(frame, (self.width, self.height))
        frame = frame.astype('uint8')

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, self.height - 10)
        fontScale = 0.3
        fontColor = (255, 255, 255)
        lineType = 1
        cv2.putText(frame, f'Episode Reward: {int(episode_reward)}', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        return frame

    def record(self, frame, reward):
        frame = self.modify_frame(frame, reward)
        self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
