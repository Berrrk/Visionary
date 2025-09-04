import cv2 as cv
import numpy as np
from torchvision import datasets

class DatasetCaltech:
    training_data = datasets.Caltech256(
        root='data/Caltech256',
        download=True,
        transform=lambda x: np.array(x)
    )

    @staticmethod
    def opencv(img):
        if img.shape[-1] == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = gray.astype(np.uint8)
        orb = cv.ORB_create(nfeatures=2000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors

def create_feature_bank():
    ds = DatasetCaltech.training_data
    all_features = []
    for img, label in ds:
        kp, des = DatasetCaltech.opencv(img)
        if des is not None:
            all_features.append({"label": label, "descriptors": des})

    X_list = []
    Y_list = []
    for item in all_features:
        des = item["descriptors"]
        X_list.append(des)
        Y_list.append(np.full((des.shape[0],), item["label"]))

    X = np.vstack(X_list)
    Y = np.concatenate(Y_list)

    np.savez("caltech256_orb_features.npz", X=X, Y=Y)
    print("Feature bankası oluşturuldu ve kaydedildi.")

def predict_image(img_path, top_k=50):
    try:
        data = np.load("caltech256_orb_features.npz")
    except FileNotFoundError:
        print("Feature bankası bulunamadı. Önce create_feature_bank() fonksiyonunu çalıştırın.")
        return None

    X = data['X']
    Y = data['Y']

    img_new = cv.imread(img_path)
    if img_new is None:
        raise FileNotFoundError(f"{img_path} bulunamadı")
    kp_new, des_new = DatasetCaltech.opencv(img_new)
    if des_new is None or len(des_new) == 0:
        print("Görüntüden descriptor çıkarılamadı.")
        return None

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_new, X)
    if len(matches) == 0:
        print("Eşleşme bulunamadı.")
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    match_labels = [Y[m.trainIdx] for m in matches[:top_k]]
    predicted_label = np.bincount(match_labels).argmax()
    print(f"Gelen görselin tahmini label'ı: {predicted_label}")
    return predicted_label


create_feature_bank()
predict_image("img.png")