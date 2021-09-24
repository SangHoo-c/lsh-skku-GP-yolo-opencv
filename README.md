# lsh-skku-GP-yolo-opencv

[Client](https://github.com/SangHoo-c/lsh-skku-GP-client)


[Data-Converter](https://github.com/theAIGuysCode/OIDv4_ToolKit)



---------





![image](https://user-images.githubusercontent.com/59246354/134623931-302d64d0-74b4-47f1-a845-92f6c723858c.png)


 
그림. 작품의 시스템 구성도 

본 작품의 시스템 구성은 이와 같다. 

총 3단계로 나누어 개발을 진행했다. 

1. Yolo 모델 학습 
2. Model Serving Server 구축 
3. Client 개발 


1. Yolo v4 모델 학습 
Google colab 에서 darknet Framework (참조 https://github.com/AlexeyAB/darknet) 를 이용하여 YOLO model 을 학습하였다. 

Custom data 로 Darknet 을 학습하기 위해선 config / data set 분류 / images 가 필요하다. (출처 . https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

![image](https://user-images.githubusercontent.com/59246354/134623861-be787233-9b8a-4e49-ba8e-f8aea0cf01f8.png)
 
그림. Darknet 학습 준비 



학습에 필요한 데이터는 OIDv4 ToolKit (참조.https://github.com/theAIGuysCode/OIDv4_ToolKit) 를 이용하여 Open Image Dataset (https://storage.googleapis.com/openimages/web/index.html) 에서 수집하였다. 

총 800 개의 train images, 100 개의 test images 를 사용하였다. 
Darknet 학습에 맞도록 (코드 참조, generate_test.py / generate_train.py ) 를 이용하여 meta data 를 설정해주었다. 

끝으로 미리 작성되어 있는 config file, obj.data 및 obj.names 을 나의 class 가 하나인 상황에 맞도록 조정해주었다. 

이후 darknet 의 train 명령어를 통해 학습을 수행한다. 

 



2. Model Serving Server 구축 


![image](https://user-images.githubusercontent.com/59246354/134623877-33a51371-f1ec-4fa9-83dc-1b8069626f39.png)


 
그림. OpenCV DNN(참조. DNN : deep neural network module, https://docs.opencv.org/3.4.14/d6/d0f/group__dnn.html) 프로세스 


readNetFromDarknet(config파일, weight파일)을 이용하여 yolo inference network 모델을 로딩한다. 3개의 scale Output layer (output_layer : ['yolo_139', 'yolo_150', 'yolo_161'] )에서 결과 데이터 추출한다. 139번 150번 161번 (52 x 52, 26 x 26, 13 x 13) Layer에 직접 접근해서 Feature Map정보를 직접 가져와야 한다. 

416 x 416 size 의 input 이미지를 받아 RGB 로 변환 후,  blobFromImage() 를 사용해서 CNN에 넣어야 할 이미지 전처리를 수행한다. 

함수 setInput() 을 사용하여 신경망에 넣을 사진만 Setting 한 후, forward() 를 통해 inference를 돌려서 원하는 layer의 Feature Map 정보만 뽑아 낸다. NMS (Non - Maximum Suppression) 를 통해 동일한 object 에 여러개의 박스가 있다면, 가장 확률(score) 가 높은 박스만 남긴다. 이후 이미지 crop 을 한다. 



 ![image](https://user-images.githubusercontent.com/59246354/134623974-39c93ab3-5377-42b9-9ca1-8b0228492ec1.png)


그림. 이미지 digit recognition 프로세스

이미지를 그레이스케일로 변환하고 작은 GaussianBlur 를 적용하여 부드럽게 처리한다. 이어서 cv2.thresold() 함수로 오츠의 이진화 알고리즘을 적용한다. 검은색 배경에 있는 이 흰색 텍스트는 이미지의 윤곽선을 찾는 데 도움이 된다. 끝으로 dilate() 함수를 적용하고 contour 를 찾아낸다. 

 
 ![image](https://user-images.githubusercontent.com/59246354/134623993-83cd2f0a-8959-4bb6-8c09-deebf7c6b464.png)


그림. 찾아낸 모든 countour 들.


자동차 등록번호판 등의 기준에 관한 고시([시행 2020. 4. 20.] [국토교통부고시 제2020-344호, 2020. 4. 20., 일부개정] ) 에 따르면, 
 
 
 ![image](https://user-images.githubusercontent.com/59246354/134624007-b64d4fe6-af8a-48c4-b8c9-3cf78f82a8dd.png)

그림. 자동차 운수 사업용 대형 등록 번호판, 별표 6

따라서 정해진 규격에 맞는 기준을 도입하여 번호판 인식에 필요한 countour 만 검출하도록 했다. 


 ![image](https://user-images.githubusercontent.com/59246354/134624017-b5e89734-f619-4c95-94b2-cda3d917dc68.png)

그림, 변수 설정 


기준은 이와 같다. 
1. 100 / 440 >   contour 넓이 / 이미지 전체 넓이 > 60 / 440
2. 130 / 220 > contour 높이 / 이미지 전체 높이 > 90 / 220



![image](https://user-images.githubusercontent.com/59246354/134624028-eb0138f1-7390-4fed-889e-77ec44670c37.png)

 
그림. 검출된 contour 들 

결과적으로, 검출된 contour 를 pytesseract (tesseract 참조. ) 를 이용해서 digit recognition 을 수행한다.

Flask 를 사용해 http server 를 개발했다. 
Base64 로 인코딩된 binary 형태의 이미지를 decode 하여 ml 모델에 입력값으로 제공했다. 이후, 검출된 text 를 반환값으로 client 에 response 값을 전달한다. 


```python
#app.py

# Binary 형태로 이미지 데이터 읽은 다음 decode 하는 방법
def stringToRGB(imgdata):
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def convert(image):
    imgdata_bytes = base64.b64decode(image)
    image = stringToRGB(imgdata_bytes)
    return image
```

```python

# app.py

@app.route('/v1/image/convert_text', methods=['POST'])
def convert_text():
    data = request.get_json()

    if 'image' not in data:
        return "", 400

    else:
        captured_img_data = convert(data['image'])
        returned_text = img_text(captured_img_data)
        return returned_text, 200


```



3. Client 개발 

Client 는 flutter 를 이용해서 개발했다. 
Dio 모듈(참조. https://pub.dev/packages/dio)을 사용,  http   client 를 개발했다. 

 
이미지를 Base64 로 encoding 하여 http 헤더에 넣어서 server 에 request 한다. 

```dart

// ml_service.dart

class MLService {
  Dio dio = Dio();

  // ml server
  Future<String> convertImageToText(Uint8List imageData) async {
    try {
      var encodedData = await compute(base64Encode, imageData);
      Response response = await dio.post(
          'http://localhost:5000/v1/image/convert_text',
          data: {'image': encodedData});
      print(response);
      return response.data;
    } catch (e) {
      print("i'm dead");
      return null;
    }
  }
}

```
