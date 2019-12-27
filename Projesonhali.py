#modül girişleri
import cv2
import numpy as np
#kameradan görüntü alımı
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    #BGR görüntüyü HSV(hue-saturation-value) ye dönüştürme
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # kırmızı, mavi, yeşil renk aralığı tanımı
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    blue_lower = np.array([99, 115, 150], np.uint8)
    blue_upper = np.array([110, 255, 255], np.uint8)
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    #Kırmızı, mavi ve yeşil renklerin görüntüde aralığının tanımlanması
    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    #Morfoloji dönüşümler
    #noise removal
    dizi = np.ones((5, 5), np.uint8)
    #Dilation
    red = cv2.dilate(red, dizi)
    res = cv2.bitwise_and(img, img, mask=red) #cv2.bitwise_or (kaynak1, kaynak2, hedef, maske)

    blue = cv2.dilate(blue, dizi)
    res1 = cv2.bitwise_and(img, img, mask=blue)

    green = cv2.dilate(green, dizi)
    res3 = cv2.bitwise_and(img, img, mask=green)

    # kırmızı renk takibi
    contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #kontur-aynı renkli nesnelerin sınır boyunca birleştirilmesi-hierarch
    for pic, contour in enumerate(contours): #numaralandırma
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Kirmizi renk", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
    # mavi renk takibi
    contours, hierarchy = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Mavi renk", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
    # yeşil renk takibi
    contours, hierarchy = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Yesil renk", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    cv2.imshow("Colour Tracking", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break