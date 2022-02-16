import numpy as np
import cv2

width,height=512,512
x,y,R=256,256,50
direction = 0 #right

while True:
    key=cv2.waitKeyEx() # 키보드에서 입력키받음
    if key == 0x1B: #ESC키
        break
    elif key==0x270000: # 방향키 방향 전환 0x270000==right
        direction=0
    elif key==0x280000: # 방향키 방향 전환 0x280000==down
        direction=1
    elif key==0x250000: # 방향키 방향 전환 0x250000==left
        direction=2
    elif key==0x260000: # 방향키 방향 전환 0x260000==up
        direction=3

    if direction ==0:
        x+=5
    elif direction==1:
        y+=5
    elif direction==2:
        x-=5
    elif direction==3:
        y-=5

    if x<R:
        x=R
        direction=0
    if x>width-R:
        x=width-R
        direction=2
    if y<R:
        y=R
        direction=1
    if y>height-R:
        y=height-R
        direction=3

    img = np.zeros((width,height,3),np.uint8)+255
    cv2.circle(img,(x,y),R,(0,0,255),-1)
    cv2.imshow('img',img)
cv2.destroyAllWindows()



# https://kali-live.tistory.com/16