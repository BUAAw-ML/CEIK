from fastapi import FastAPI, Depends, Request, Response, status, HTTPException, Body
import uvicorn
from typing import List
from pydantic import BaseModel
import json
import re
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from predictor import Predictor
from easydict import EasyDict

from fastapi.middleware.cors import CORSMiddleware

predictor = Predictor()
# f = open("test.jpg","rb")
# ls_f = str(base64.b64encode(f.read()))
# print(ls_f)

# a = cv2.imread('test2.jpg')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chatbot")
async def chatbot(request: Request):#
    raw_body = await request.body()
    raw_body = json.loads(raw_body.decode())

    res = raw_body['content'][-1][-1]
    res = re.split('[,"\<\>]',res)
    img = res[-3]
    text = res[-1]

    # 解码图片
    imgdata = base64.b64decode(img)
    encode_image = np.asarray(bytearray(imgdata), dtype="uint8")# 二进制转换为一维数组
    img = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)
    # print(img.shape == a.shape)
    
    # difference = cv2.subtract(a, img)
    # print(not np.any(difference))
    input = {'img': img, 'question': text}

    res = predictor.predict(input)
    answer = res[0]['predictions'][0]['answer']
    knowledge = ''

    # print(res[0]['retrieved_docs'][0])
    for i,item in enumerate(res[0]['retrieved_docs'][0]):
        knowledge += str(i+1) + '. '
        knowledge += item['content']
    
    output = answer + "—— (Reference knowledge: " + knowledge + ")"
    # print(output)

    return output
 
if __name__ == '__main__':
    
    uvicorn.run(app="visualization:app", host="0.0.0.0", port=6901)#, reload=True
