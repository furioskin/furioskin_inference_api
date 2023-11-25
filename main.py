from fastapi import FastAPI, UploadFile
import uvicorn

from furiosa.models.vision import SSDResNet34
from furiosa.runtime.sync import create_runner

import cv2
import numpy as np

def preprocess(img_path, new_shape=(640, 640)):
    img = cv2.imread(img_path)

    img, preproc_params = letterbox(img, new_shape, auto=False)

    img = img.transpose((2, 0, 1))[::-1]
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img, preproc_params

def letterbox(
    img, new_shape, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)

    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = int(round(ratio * w)), int(round(ratio * h))
    dw, dh = (new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1])

    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        img = cv2.resize(img, new_unpad, interpolation=interpolation)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, (ratio, (dw, dh))



app = FastAPI()

@app.post("/infer")
def infer(file: UploadFile):

    file_location = f"static/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    base_path = ''
    
    #images = [base_path+'images/coco/a10.png']
    images = [base_path+file_location]

    ssd_resnet34 = SSDResNet34()

    with create_runner(ssd_resnet34.model_source(num_pe=1), device='warboy(1)*1') as runner1:
        with create_runner("/home/elicer/dongwon/furioskin_inference_api/resnet_retrained_model_quantized_percentile.onnx", device='warboy(1)*1') as runner2:
            # object detection
            input1, contexts = ssd_resnet34.preprocess(images[0])
            output1 = runner1.run(input1)

            result = ssd_resnet34.postprocess(output1, contexts)

            # object classification
            if len(result) > 0:
                input2, contexts = preprocess(images[0])
                outputs = runner2.run(input2)
                max_index = np.argmax(outputs)
                #print('label', max_index)
                # max_index = label -> 0 : 여드름, 1 : 두드러기
                return {"labels": int(max_index)}
            
            # object detection이 되지 않았을 때
            else :
                print('object non detected!!')
                return {"labels": -1}  

    
    
#    return {"label": 1}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=65501)