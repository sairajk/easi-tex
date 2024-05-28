import cv2
import numpy as np
from scipy import ndimage


class Augmentations:
    @staticmethod
    def get_random_crop(img_1, img_2, dim):
        crop_height, crop_width = dim
        max_x = img_1.shape[1] - crop_width + 1
        max_y = img_1.shape[0] - crop_height + 1
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = img_1[y: y + crop_height, x: x + crop_width]
        if img_2 is not None:
            crop_2 = img_2[y: y + crop_height, x: x + crop_width]
        else:
            crop_2 = None
        return crop, crop_2

    @staticmethod
    def random_horizontal_flip(img_1, img_2=None, probability=0.5):
        if np.random.random() < probability:
            img_1 = img_1[:,::-1]
            if img_2 is not None:
                img_2 = img_2[:,::-1]
        
        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def center_crop(img, dim):
        """
        Returns center cropped image
        Args:
            img: image to be center cropped
            dim: dimensions (height, width) [(y, x)] to be cropped
        """
        height, width = img.shape[0], img.shape[1]
        # Process crop width and height for max available dimension
        crop_width = dim[1] if dim[1] < img.shape[1] else img.shape[1]
        crop_height = dim[0] if dim[0] < img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img

    @staticmethod
    def center_pad(img, dim, pad_color=[0, 0, 0]):
        """
        Args:
            img: image to be center padded
            dim: dimensions (height, width) [(y, x)] to be cropped
        """
        h, w = img.shape[:2]
        if dim == -1:  # Pad to square
            dim = (max(h, w), max(h, w))
        top = (dim[0] - h) // 2
        bottom = dim[0] - h - top
        left = (dim[1] - w) // 2
        right = dim[1] - w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        return img

    @staticmethod
    def random_rotate(img_1, img_2=None, angle_range=[-15, 15], probability=0.5, bg_color=[0, 0, 0]):
        assert angle_range[0] < angle_range[1]
        if np.random.random() < probability:
            angle = np.random.randint(angle_range[0], angle_range[1] + 1)
            
            shape = img_1.shape[:2]
            img_1 = ndimage.rotate(img_1, angle)
            img_1 = Augmentations.center_crop(img_1, shape)
            img_1 = Augmentations.center_pad(img_1, shape, pad_color=bg_color)
            if img_2 is not None:
                shape = img_2.shape[:2]
                img_2 = ndimage.rotate(img_2, angle)
                img_2 = Augmentations.center_crop(img_2, shape)
                img_2 = Augmentations.center_pad(img_2, shape, pad_color=bg_color)
        
        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def random_resize(img_1, img_2=None, resize_range=[0.9, 1.1], probability=0.5, same_fxfy=False, bg_color=[0, 0, 0]):
        assert resize_range[0] < resize_range[1]
        if np.random.random() < probability:
            fx = np.random.uniform(resize_range[0], resize_range[1])
            if same_fxfy:
                fy = fx
            else:
                fy = np.random.uniform(resize_range[0], resize_range[1])
            
            shape = img_1.shape[:2]
            img_1 = cv2.resize(img_1, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
            img_1 = Augmentations.center_crop(img_1, shape)
            img_1 = Augmentations.center_pad(img_1, shape, pad_color=bg_color)
            if img_2 is not None:
                shape = img_2.shape[:2]
                img_2 = cv2.resize(img_2, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
                img_2 = Augmentations.center_crop(img_2, shape)
                img_2 = Augmentations.center_pad(img_2, shape, pad_color=bg_color)
        
        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def random_shear(img_1, img_2=None, max_shear=0.2, probability=0.5, bg_color=[0, 0, 0]):
        if np.random.random() < probability:
            sx = np.random.uniform(0, max_shear)
            sy = np.random.uniform(0, max_shear)
            M = np.float32([[1, sx, 0], [sy, 1, 0], [0, 0, 1]])

            flip_x = np.random.random() < 0.5
            flip_y = np.random.random() < 0.5
            if flip_x:
                img_1 = img_1[:,::-1]
                if img_2 is not None: 
                    img_2 = img_2[:,::-1]
            if flip_y:
                img_1 = img_1[::-1,:]
                if img_2 is not None:
                    img_2 = img_2[::-1,:]
            
            shape = img_1.shape[:2]
            img_1 = cv2.warpPerspective(img_1, M, (int(shape[1] + sx * shape[0]), int(shape[0] + sy * shape[1])))
            img_1 = Augmentations.center_crop(img_1, shape)
            img_1 = Augmentations.center_pad(img_1, shape, pad_color=bg_color)
            if img_2 is not None:
                shape = img_2.shape[:2]
                img_2 = cv2.warpPerspective(img_2, M, (int(shape[1] + sx * shape[0]), int(shape[0] + sy * shape[1])))
                img_2 = Augmentations.center_crop(img_2, shape)
                img_2 = Augmentations.center_pad(img_2, shape, pad_color=bg_color)

            if flip_y:
                img_1 = img_1[::-1,:]
                if img_2 is not None:
                    img_2 = img_2[::-1,:]
            if flip_x:
                img_1 = img_1[:,::-1]
                if img_2 is not None:
                    img_2 = img_2[:,::-1]

        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def random_contrast(img_1, img_2=None, alpha_range=[0.75, 1.25], probability=0.5):
        assert alpha_range[0] < alpha_range[1]
        if np.random.random() < probability:
            alpha = np.random.uniform(alpha_range[0], alpha_range[1])

            img_1 = np.clip(cv2.convertScaleAbs(img_1, alpha=alpha, beta=0), 0, 255)
            if img_2 is not None:
                img_2 = np.clip(cv2.convertScaleAbs(img_2, alpha=alpha, beta=0), 0, 255)

        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def random_brightness(img_1, img_2=None, beta_range=[-25, 25], probability=0.5):
        assert beta_range[0] < beta_range[1]
        if np.random.random() < probability:
            beta = np.random.randint(beta_range[0], beta_range[1] + 1)

            img_1 = np.clip(cv2.convertScaleAbs(img_1, alpha=1, beta=beta), 0, 255)
            if img_2 is not None:
                img_2 = np.clip(cv2.convertScaleAbs(img_2, alpha=1, beta=beta), 0, 255)

        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def random_gamma_correction(img_1, img_2=None, gamma_range=[0.75, 1.33], probability=0.5):
        assert gamma_range[0] < gamma_range[1]
        if np.random.random() < probability:
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

            img_1 = cv2.LUT(img_1, lookUpTable)
            if img_2 is not None:
                img_2 = cv2.LUT(img_2, lookUpTable)
        
        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def random_color_temperature(img_1, img_2=None, probability=0.5):
        if np.random.random() < probability:
            kelvin_table = {  # http://www.vendian.org/mncharity/dir3/blackbody/
                # 1000: (255,56,0),
                # 1500: (255,109,0),
                # 2000: (255,137,18),
                # 2500: (255,161,72),
                # 3000: (255,180,107),
                # 3500: (255,196,137),
                # 4000: (255,209,163),
                # 4500: (255,219,186),
                # 5000: (255,228,206),
                # 5500: (255,236,224),
                # 6000: (255,243,239),
                # 6500: (255,249,253),
                # 7000: (245,243,255),
                # 7500: (235,238,255),
                # 8000: (227,233,255),
                # 8500: (220,229,255),
                # 9000: (214,225,255),
                # 9500: (208,222,255),
                # 10000: (204,219,255)

                6000: (255, 243, 239),
                6100: (255, 244, 242),
                6200: (255, 245, 245),
                6300: (255, 246, 248),
                6400: (255, 248, 251),
                6500: (255, 249, 253),
                6600: (254, 249, 255),
                6700: (252, 247, 255),
                6800: (249, 246, 255),
                6900: (247, 245, 255),
                7000: (245, 243, 255),
                7100: (243, 242, 255),
                7200: (240, 241, 255),
                7300: (239, 240, 255),
                7400: (237, 239, 255),
                7500: (235, 238, 255),
            }
            multiplier = kelvin_table[np.random.choice(list(kelvin_table.keys()))]
            multiplier = np.array(multiplier).reshape((1,1,3)) / 255.0

            img_1 = (img_1 * multiplier).astype(np.uint8)
            if img_2 is not None:
                img_2 = (img_2 * multiplier).astype(np.uint8)

        if img_2 is not None:
            return img_1, img_2
        else:
            return img_1

    @staticmethod
    def structural_augmentation(img_1, img_2, bg_color=[0, 0, 0]):
        img_1, img_2 = Augmentations.random_horizontal_flip(img_1, img_2, probability=0.5)
        img_1, img_2 = Augmentations.random_resize(img_1, img_2, resize_range=[0.9, 1.1], probability=1.0, bg_color=bg_color)
        img_1, img_2 = Augmentations.random_rotate(img_1, img_2, angle_range=[-10, 10], probability=1.0, bg_color=bg_color)
        img_1, img_2 = Augmentations.random_shear(img_1, img_2, max_shear=0.2, probability=1.0, bg_color=bg_color)
        return img_1, img_2

    @staticmethod
    def structural_augmentation_threestudio(img_1, img_2, bg_color=[0, 0, 0]):
        # img_1, img_2 = Augmentations.random_horizontal_flip(img_1, img_2, probability=0.5)
        img_1, img_2 = Augmentations.random_resize(img_1, img_2, resize_range=[0.9, 1.1], probability=1.0, same_fxfy=True, bg_color=bg_color)
        img_1, img_2 = Augmentations.random_rotate(img_1, img_2, angle_range=[-5, 5], probability=1.0, bg_color=bg_color)
        # img_1, img_2 = Augmentations.random_shear(img_1, img_2, max_shear=0.2, probability=1.0)
        return img_1, img_2
    
    @staticmethod
    def color_augmentation(img_1, img_2):
        img_1, img_2 = Augmentations.random_contrast(img_1, img_2, alpha_range=[0.75, 1.25], probability=1.0)
        img_1, img_2 = Augmentations.random_brightness(img_1, img_2, beta_range=[-25, 25], probability=1.0)
        img_1, img_2 = Augmentations.random_gamma_correction(img_1, img_2, gamma_range=[0.75, 1.33], probability=0.5)
        img_1, img_2 = Augmentations.random_color_temperature(img_1, img_2, probability=0.5)
        return img_1, img_2


class ImageUtils:
    @staticmethod
    def recenter_object(img, mask, resolution, max_obj_size_percent, bg_color=[0, 0, 0]):
        ys, xs = np.nonzero(mask[:, :, 0])
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        mask = mask[y_min:y_max, x_min:x_max, :]
        img = img[y_min:y_max, x_min:x_max, :]
        # Resize_image
        max_object_size = max_obj_size_percent * resolution
        if max(img.shape[:2]) > max_object_size:
            ratio = max_object_size / max(img.shape[:2])
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        img = Augmentations.center_pad(img, (resolution, resolution), pad_color=bg_color)
        mask = Augmentations.center_pad(mask, (resolution, resolution), pad_color=bg_color)
        return img, mask
    
    @staticmethod
    def canny_edge_detector(img, low=25, high=100):
        img = cv2.GaussianBlur(img, (5,5), sigmaX=0, sigmaY=0)
        all_edges = []
        for i in range(3):  # loop over the R, G, and B channels
            edges = cv2.Canny(img[:, :, i], low, high)
            all_edges.append(edges)
        all_edges = np.stack(all_edges, axis=-1)
        all_edges = np.mean(all_edges, axis=-1)
        return all_edges
    
    @staticmethod
    def laplacian_edge_detector(img, kernel_size=3):
        all_edges = []
        for i in range(3):  # loop over the R, G, and B channels
            edges = cv2.Laplacian(img[:, :, i], cv2.CV_32F, ksize=kernel_size)
            edges = cv2.convertScaleAbs(edges)
            all_edges.append(edges)
        all_edges = np.stack(all_edges, axis=-1)
        all_edges = np.mean(all_edges, axis=-1)
        return all_edges
    
    @staticmethod
    def input_feature_extractor(img, low=25, high=100):
        edge_img = ImageUtils.canny_edge_detector(img, low, high)
        # edge_img = CompCarsDataset.laplacian_edge_detector(img)
        out_img = np.stack([edge_img,] * 3, axis=-1)
        # Sanity check
        if len(out_img.shape) == 2:
            out_img = np.expand_dims(out_img, axis=-1)
        if out_img.shape[-1] == 1:
            out_img = np.concatenate([out_img] * 3, axis=-1)
        return out_img

    @staticmethod
    def swap_background(img, mask, bg_color):
        bg_image = np.ones_like(img) * np.array(bg_color, dtype=np.uint8).reshape((1,1,3))
        return cv2.bitwise_and(img, mask) + cv2.bitwise_and(bg_image, (255 - mask))

    @staticmethod
    def load_image(img_path, zipfile=None):
        if zipfile is None:
            image = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        else:
            image = zipfile.read(img_path)
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR2RGB
        return image
