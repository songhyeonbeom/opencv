import sys
import cv2
import numpy as np


src = cv2.imread('아이유03.jpeg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()



dx = cv2.Sobel(src, cv2.CV_32F, 1, 0) # float 형태의 미분값을 저장
dy = cv2.Sobel(src, cv2.CV_32F, 0, 1)


mag = cv2.magnitude(dx, dy) # 그래디언트 크기
mag = np.clip(mag, 0, 255).astype(np.uint8) # 255보다 커질 수 있으므로 saturate 연산

# 흰색과 검은색으로만 나타내는 윤곽선 생성
dst = np.zeros(src.shape[:2], np.uint8) # 0(검은색)으로 채워져 있는 영상 생성
dst[mag > 120] = 255 # 120은 임계값, 값을 적절하게 설정하면 내가 원하는 부분만 나타낼 수 있음

# dst = cv2.threshold(mag, 120, 255, cv2.THRESH_BINARY) # cv2 함수로 임계값 설정하기

cv2.imshow('src', src)
cv2.imshow('mag', mag)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()