import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(42)

# Load images from files
def Load_img(dossier):
    images = []
    noms = []
    for filename in os.listdir(dossier):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dossier, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                noms.append(filename)
    return images, noms

# Finds matches between two sets of SIFT descriptors
def Find_Matche(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract keypoints for the good matches
    kp1 = [des1[m.queryIdx] for m in good_matches]
    kp2 = [des2[m.trainIdx] for m in good_matches]

    return good_matches, kp1, kp2

dossier_training = "./Training"
dossier_test = "./Test"

# Load training and test images
images_train, noms_train = Load_img(dossier_training)
images_test, noms_test = Load_img(dossier_test)

resultats = {}
sift = cv2.SIFT_create()

for i, img_test in enumerate(images_test):
    kp_test, des_test = sift.detectAndCompute(img_test, None)
    if des_test is not None:
        meilleurs_matchs_par_image = []
        for img_train in images_train:
            kp_train, des_train = sift.detectAndCompute(img_train, None)
            if des_train is not None:
                bonnes_correspondances, _, _ = Find_Matche(des_test, des_train)
                meilleurs_matchs_par_image.append(len(bonnes_correspondances))

        if meilleurs_matchs_par_image:
            idx_max = np.argmax(meilleurs_matchs_par_image)
            image_similaire = noms_train[idx_max]
        else:
            image_similaire = "Aucune correspondance trouvée"
        resultats[noms_test[i]] = image_similaire

for test_img, similar_img in resultats.items():
    print(f"Image de Test: {test_img} --> Image de Training la plus similaire: {similar_img}")
    img_test = cv2.imread(os.path.join("./Test", test_img))
    img_train = cv2.imread(os.path.join("./Training", similar_img))

    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)

    good_matches, kp1, kp2 = Find_Matche(des_test, des_train)

    # Draw matches
    img_matches = cv2.drawMatchesKnn(img_test, kp_test, img_train, kp_train, [good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 8))
    plt.imshow(img_matches)
    plt.title("Matches between Test and Training Images")
    plt.show()
