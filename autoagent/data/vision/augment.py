import cv2
import numpy as np


class Transform:
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Returns:
            - augmented image
            - mapping func from old x,y pixel position
              to new x,y position
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__


class Identity(Transform):
    def __call__(self, img):
        return img, lambda x, y: (x, y)


class Resize(Transform):
    """
    Resize img to keep aspect ratio.
    """

    def __init__(self, size, pad=True):
        super().__init__()
        self.size = size
        self.pad = pad
        self.fill_color = np.array([127, 127, 127], dtype=np.uint8)

    def __call__(self, img):
        h, w, _ = img.shape

        # Find target ratio
        ratio = min(self.size[1]/w, self.size[0]/h)

        # Target width and height
        t_w = int(w*ratio)
        t_h = int(h*ratio)

        if self.pad:
            # Compute required padding
            if (self.size[1] - t_w) % 2 == 0:
                r = int((self.size[1] - t_w) / 2)
                d_w1, d_w2 = r, r
            else:
                r = (self.size[1] - t_w)//2
                d_w1, d_w2 = r, r+1

            if (self.size[0] - t_h) % 2 == 0:
                r = int((self.size[0] - t_h) / 2)
                d_h1, d_h2 = r, r
            else:
                r = (self.size[0] - t_h)//2
                d_h1, d_h2 = r, r+1

            img = cv2.resize(img, (t_w, t_h))
            new_img = np.full(self.size + (3,), self.fill_color, dtype=img.dtype)

            # Insert image
            new_img[d_h1:self.size[0]-d_h2, d_w1:self.size[1]-d_w2, :] = img

            return new_img, lambda x, y: (int(x*t_w/w + (self.size[1]-t_w)/2),
                                          int(y*t_h/h + (self.size[0]-t_h)/2))
        else:
            new_img = cv2.resize(img, (t_w, t_h))
            return new_img, lambda x, y: (int(x*t_w/w), int(y*t_h/h))


class HFlip(Transform):
    """
    Img horizontal flip.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        w = img.shape[1]
        new_img = img[:, ::-1, :]
        return new_img, lambda x, y: ((w-1) - x, y)


class DeltaBright(Transform):
    """
    Change img brightness.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        dtype = img.dtype
        new_img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
        new_img[..., 2] = np.clip(
            new_img[..., 2] * np.random.uniform(low=0.5, high=1.5),
            0,
            255
        )
        new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR).astype(dtype)
        return new_img, lambda x, y: (x, y)


class DataAugmentSeq:
    """
    Applies a sequence of transforms.
    """
    def __init__(self, transforms, probs):
        self.transforms = transforms
        self.probs = probs

    def augment(self, x):
        transform_funcs = []
        for i in range(len(self.transforms)):
            if np.random.rand(1) < self.probs[i]:
                x, t_func = self.transforms[i](x)
                transform_funcs.append(t_func)

        return x, transform_funcs

    def __str__(self):
        return ", ".join(t.__class__.__name__
                         for t in self.transforms)
