import cv2


def png_to_jpg_mask():
    img_path = "data/texture_images/round_bird/round_bird_4.png"

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = img[:, :, -1:]
    img = img[:, :, :-1]

    cv2.imwrite(img_path.replace(".png", "_mask.jpg"), mask)
    cv2.imwrite(img_path.replace(".png", ".jpg"), img)


if __name__ == '__main__':
    png_to_jpg_mask()
