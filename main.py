import cv2
import mediapipe as mp
import numpy as np


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

heart_img = cv2.imread('Scripts//black.png', cv2.IMREAD_UNCHANGED)
like_img=cv2.imread('Scripts//like.png', cv2.IMREAD_UNCHANGED)
dislike_img=cv2.imread('Scripts//dislike.png', cv2.IMREAD_UNCHANGED)
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        if len(lmList) == 21:
            thumb_tip_y = lmList[tipIds[0]][2]
            other_finger_tips_y = [lmList[tip_id][2] for tip_id in tipIds[1:]]
        like_sign = all(finger_tip_y > thumb_tip_y for finger_tip_y in other_finger_tips_y) and thumb_tip_y < \
                    lmList[tipIds[0] - 2][2]
        dislike_sign = all(finger_tip_y < thumb_tip_y for finger_tip_y in other_finger_tips_y) and thumb_tip_y > \
                       lmList[tipIds[0] - 2][2]

        peace_sign = (lmList[tipIds[1]][2] < lmList[tipIds[1] - 2][2] and
                      lmList[tipIds[2]][2] < lmList[tipIds[2] - 2][2])

        all_fingers_up = all(finger_tip_y < thumb_tip_y for finger_tip_y in other_finger_tips_y) and thumb_tip_y < \
                         lmList[tipIds[0] - 2][2]



        if like_sign:
              cv2.putText(img, 'Like Sign', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
              like_width = 20  # Adjust the size of the heart image as needed
              like_height = 20  # Adjust the size of the heart image as needed

              like_center_x = (lmList[9][1] + lmList[0][1]) // 2
              like_center_y = (lmList[9][2] + lmList[0][2]) // 2
              # Define heart positions in different areas of the frame
              like_positions = [
                  (like_center_x - 150, like_center_y - 150),  # Upper left
                  (like_center_x + 150, like_center_y - 150),  # Upper right
                  (like_center_x - 150, like_center_y + 150),  # Lower left
                  (like_center_x + 150, like_center_y + 150),  # Lower right
                  (like_center_x, like_center_y - 200),  # Upper center
                  (like_center_x, like_center_y + 200),  # Lower center
                  (like_center_x - 200, like_center_y),  # Left center
                  (like_center_x + 200, like_center_y)  # Right center
              ]


              if like_img is not None:
                  for like_position in like_positions:
                      top_left_x, top_left_y = like_position
                      bottom_right_x, bottom_right_y = top_left_x + like_width, top_left_y + like_height

                      # Ensure the heart image is within the frame
                      if bottom_right_x <= img.shape[1] and bottom_right_y <= img.shape[0]:
                          # Extract heart region from heart image
                          like_roi_resized = cv2.resize(like_img, (like_width, like_height))

                          # Blend the heart image onto the frame at the specified position
                          img_region = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                          for c in range(0, 3):
                              img_region[:, :, c] = like_roi_resized[:, :, c] * 0.5 + img_region[:, :, c] * 0.5
        elif dislike_sign:
               cv2.putText(img, 'Dislike Sign', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
               dislike_width = 40  # Adjust the size of the heart image as needed
               dislike_height = 40  # Adjust the size of the heart image as needed

               dislike_center_x = (lmList[9][1] + lmList[0][1]) // 2
               dislike_center_y = (lmList[9][2] + lmList[0][2]) // 2
               # Define heart positions in different areas of the frame
               dislike_positions = [
                   (dislike_center_x - 150, dislike_center_y - 150),  # Upper left
                   (dislike_center_x + 150, dislike_center_y - 150),  # Upper right
                   (dislike_center_x - 150, dislike_center_y + 150),  # Lower left
                   (dislike_center_x + 150, dislike_center_y + 150),  # Lower right
                   (dislike_center_x, dislike_center_y - 200),  # Upper center
                   (dislike_center_x, dislike_center_y + 200),  # Lower center
                   (dislike_center_x - 200, dislike_center_y),  # Left center
                   (dislike_center_x + 200,dislike_center_y)  # Right center
               ]

               if dislike_img is not None:
                   for dislike_position in dislike_positions:
                       top_left_x, top_left_y = dislike_position
                       bottom_right_x, bottom_right_y = top_left_x + dislike_width, top_left_y + dislike_height

                       # Ensure the heart image is within the frame
                       if bottom_right_x <= img.shape[1] and bottom_right_y <= img.shape[0]:
                           # Extract heart region from heart image
                           dislike_roi_resized = cv2.resize(dislike_img, (dislike_width, dislike_height))

                           # Blend the heart image onto the frame at the specified position
                           img_region = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                           for c in range(0, 3):
                               img_region[:, :, c] = dislike_roi_resized[:, :, c] * 0.5 + img_region[:, :, c] * 0.5

        elif peace_sign and not like_sign and not dislike_sign  and not all_fingers_up:
            heart_width = 10  # Adjust the size of the heart image as needed
            heart_height = 10  # Adjust the size of the heart image as needed

            hand_center_x = (lmList[9][1] + lmList[0][1]) // 2  # Example: Calculate hand center using two landmarks
            hand_center_y = (lmList[9][2] + lmList[0][2]) // 2

            # Define heart positions in different areas of the frame
            heart_positions = [
                (hand_center_x - 150, hand_center_y - 150),  # Upper left
                (hand_center_x + 150, hand_center_y - 150),  # Upper right
                (hand_center_x - 150, hand_center_y + 150),  # Lower left
                (hand_center_x + 150, hand_center_y + 150),  # Lower right
                (hand_center_x, hand_center_y - 200),  # Upper center
                (hand_center_x, hand_center_y + 200),  # Lower center
                (hand_center_x - 200, hand_center_y),  # Left center
                (hand_center_x + 200, hand_center_y)  # Right center
            ]

            if heart_img is not None:
                for heart_position in heart_positions:
                    top_left_x, top_left_y = heart_position
                    bottom_right_x, bottom_right_y = top_left_x + heart_width, top_left_y + heart_height

                    # Ensure the heart image is within the frame
                    if bottom_right_x <= img.shape[1] and bottom_right_y <= img.shape[0]:
                        # Extract heart region from heart image
                        heart_roi_resized = cv2.resize(heart_img, (heart_width, heart_height))

                        # Blend the heart image onto the frame at the specified position
                        img_region = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                        for c in range(0, 3):
                            img_region[:, :, c] = heart_roi_resized[:, :, c] * 0.5 + img_region[:, :, c] * 0.5
        if all_fingers_up:
          cv2.putText(img, '', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


    cv2.imshow('HandTracker', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
