import numpy as np
import imageio


def load_video(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    nframes = vid.count_frames()
    return vid, fps, nframes


class SaveVideo:
    def __init__(self, outname, fps):
        self.outname = outname
        self.fps = fps

    def __enter__(self):
        self.writter = imageio.get_writer(self.outname,
                                          format='FFMPEG',
                                          fps=self.fps)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.writter.close()

    def __iadd__(self, data):
        if np.max(data) <= 1:
            data = np.array(255*data, dtype=np.uint8)
        else:
            data = np.array(data, dtype=np.uint8)
        self.writter.append_data(data)
        return self
