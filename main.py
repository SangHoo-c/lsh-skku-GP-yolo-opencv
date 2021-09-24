"""
https://github.com/theAIGuysCode/yolov4-custom-functions
https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/license_plate_recognizer.py
각 과정 설명 : https://mickael-k.tistory.com/27
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import os


def img_text(img):
    if img == "":
        # default image
        img = cv2.imread('images/bus2.jpeg')

    # serving 할때 사용할 코드
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print('image shape:', img.shape)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img_rgb)
    # plt.show()

    # 가급적 절대 경로 사용.
    CUR_DIR = os.path.abspath('.')
    weights_path = os.path.join(CUR_DIR, './yolov4/backup/yolov4-obj_last.weights')
    config_path = os.path.join(CUR_DIR, './yolov4/yolov4-obj.cfg')
    # config 파일 인자가 먼저 옴.
    cv_net_yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    labels_to_names_seq = {0: 'license_plate'}

    # 전체 Darknet layer 에서 13x13 grid, 26x26, 52x52 grid 에서 detect 된 Output layer 만 filtering
    # getLayerNames () : 네트워크의 모든 레이어 이름을 가져옵니다.
    layer_names = cv_net_yolo.getLayerNames()
    print('layer_names : ', layer_names)
    # layer_names : ['conv_0', 'bn_0', 'relu_1', 'conv_1', 'bn_1', 'relu_2', 'conv_ ....
    outlayer_names = [layer_names[i[0] - 1] for i in cv_net_yolo.getUnconnectedOutLayers()]

    # 마지막 레이어로 인식되는, 연결되어 있지 않은 레이어 => 출력 레이어
    # getUnconnectedOutLayers : 출력 레이어 가져오기
    for i in cv_net_yolo.getUnconnectedOutLayers():
        print(i, layer_names[i[0] - 1])
    # cv_net_yolo.getUnconnectedOutLayers() : [[327], [353], [379]]

    print('output_layer name:', outlayer_names)

    # 입력받은 이미지를 blob 형태로 변환
    out = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    print('type : ', type(out))
    print('shape :', out.shape)
    print('size :', out.size)

    # 말그대로 신경망에 넣을 사진만 Setting 해준다.
    # 생성된 blob 객체, 네트워크에 입력해준다.
    cv_net_yolo.setInput(out)

    # Object Detection 수행하여 결과를 cv_outs 으로 반환
    # 출력 blob 가 입력으로 네트워크에 전달되고, 정방향 패스가 실행되어
    # 출력으로 예측된 경계 상자 목록을 얻습니다.
    # run the forward pass to get output of the output layers.
    # 순방향으로 네트워크를 실행한다는 의미 == 추론
    # 반환값 : 지정한 레이어의 출력 블롭 (blob for first output of specified layer.)
    cv_outs = cv_net_yolo.forward(outlayer_names)  # inference 를 돌려서 원하는 layer 의 Feature Map 정보만 뽑아 낸다.
    print('cv_outs type : list // cv_outs length :', len(cv_outs))
    # print("cv_outs[0] : 첫번째 FeatureMap 13 x 13 x 85, cv_outs[1] : 두번째 FeatureMap 26 x 26 x 85 ")
    # 52 x 52 / 26 x 26 / 13 x 13 /
    for cv_out in cv_outs:
        print(cv_out.shape)

    # bounding box의 테두리와 caption 글자색 지정
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)

    # 원본 이미지를 네트웍에 입력시에는 (416, 416)로 resize 함.
    # 이후 결과가 출력되면 resize된 이미지 기반으로 bounding box 위치가 예측 되므로 이를 다시 원복하기 위해 원본 이미지 shape정보 필요
    rows = img.shape[0]
    cols = img.shape[1]

    conf_threshold = 0.5
    nms_threshold = 0.4  # 이 값이 클 수록 box가 많이 사라짐. 조금만 겹쳐도 NMS로 둘 중 하나 삭제하므로

    class_ids = []
    confidences = []
    boxes = []

    # 3개의 개별 output layer 별로 Detect 된 Object 들에 대해서 Detection 정보 추출 및 시각화
    for ix, output in enumerate(cv_outs):
        print('output shape:', output.shape)
        # Detected 된 Object 별 iteration
        for jx, detection in enumerate(output):
            # class score 는 detection 배열에서 5번째 이후 위치에 있는 값. 즉 6번쨰~85번째 까지의 값
            scores = detection[5:]
            # scores 배열에서 가장 높은 값을 가지는 값이 class confidence, 그리고 그때의 위치 인덱스가 class id
            class_id = np.argmax(scores)
            confidence = scores[class_id]  # 5번째 값은 objectness-score 이다. 객체인지 아닌지의 확률이다. 6번째~85번째 까지의 값이 그 객체일 확률 값이다.

            # confidence 가 지정된 conf_threshold 보다 작은 값은 제외
            if confidence > conf_threshold:
                print('ix:', ix, 'jx:', jx, 'class_id', class_id, 'confidence:', confidence)
                # detection 은 scale 된 좌상단, 우하단 좌표를 반환하는 것이 아니라, detection object 의 중심좌표와 너비/높이를 반환
                # 원본 이미지에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산
                center_x = int(detection[0] * cols)
                center_y = int(detection[1] * rows)
                width = int(detection[2] * cols)
                height = int(detection[3] * rows)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # 3개의 개별 output layer 별로 Detect 된 Object 들에 대한 class id, confidence, 좌표정보를 모두 수집
                class_ids.append(class_id)
                confidences.append(float(confidence))

                boxes.append([left, top, width, height])

    conf_threshold = 0.5
    nms_threshold = 0.4
    # 노이즈 제거, 같은 물체에 대해 겹쳐있는 박스 제거
    # 동일한 object 에 여러개의 박스가 있다면, 가장 확률(score) 가 높은 박스만 남긴다.
    # NMS (Non - Maximum Suppression)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    print(idxs)  # 이 index 의 box 만 살아남았음을 의미한다.

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 이미지 crop & resize
    if len(idxs) > 0:
        for i in idxs.flatten():
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # print(int(left), int(top))
            # print(int(left + width), int(top + height))

            img_cropped = img_rgb[int(top):int(top + height), int(left):int(left + width)]
            img_cropped_resized = cv2.resize(img_cropped, (0, 0), fx=3, fy=3)
            # print(img_cropped.shape)
            # print(img_cropped_resized.shape)

            break

    # gray scale
    plate_gray = cv2.cvtColor(img_cropped_resized, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blur = cv2.GaussianBlur(plate_gray, (5, 5), 1)

    # Otsu method
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # 모폴로지 구조 요소(커널) 생성 함수
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply dilation 이미지 팽창
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    # 윤곽선 검출
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # boundingRect : 주어진 점을 감싸는 최소 크기 사각형(바운딩 박스)를 반환, (x, y, w, h)
    # (x, y) 좌표가 작은 순으로 정렬
    sorted_contours = sorted(contours, key=lambda ctr: [cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]])

    # im2 에 사각형을 그리기 위해 copy
    im2 = plate_gray.copy()

    plate_num = ""
    # 좌상단을 기준으로 정렬된 윤곽선들에 대해 loop
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im2.shape

        if not (100 / 440 > w / width > 60 / 440):
            continue
        if not (130 / 220 > h / height > 90 / 220):
            continue

        # draw the rectangle
        print(x, y)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
        roi = cv2.bitwise_not(roi)
        text = pytesseract.image_to_string(roi,
                                           config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        plate_num += text.strip()

    # plt.figure(figsize=(12, 12))
    # plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    return plate_num


if __name__ == '__main__':
    img_text("")
