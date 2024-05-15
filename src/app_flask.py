from flask import Flask, jsonify, render_template, make_response, send_from_directory, request
from flask_cors import CORS
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
import cv2
# import sys
# sys.path.append(os.getcwd())
from predictor import Predictor
from easydict import EasyDict
import base64
import numpy as np
from PIL import Image
from io import BytesIO

executor = ThreadPoolExecutor(1)
app = Flask(__name__)
CORS(app)

# predictor = Predictor()

# img_path = 'test.jpg'

# input = EasyDict({})
# input['img'] = cv2.imread(img_path)
# input['question'] = 'This holder is also also called a shoe what?'

# result_text = predictor.predict(input)
# print(result_text[0]["predictions"][0]["answer"])
# print(result_text[0]["retrieved_docs"][0])

# 在main中实例化
sess = None
classifyApp = None 

# @app.route('/')
# def hello_world():
#     return  render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def getAnswer():

    # a = request.form.to_dict()


    # text = get_data.get('text', '') # 有“text”字段则返回value，否则返回 '
    # content = json.loads(list(request.form.to_dict().keys())[0], strict=False)

    get_data = str(request.form.to_dict())
    print(get_data)
    get_data = get_data.split('[')[-1][9:]
    print(get_data)
    get_data = re.split('[,\<\>"]',get_data)
    img = get_data[3][:-2]
    # 看看image_base64类型是不是正确的“bytes”类型
    img = bytes(img,encoding="utf-8")
    print(img)
    print(type(img))  
    # 解码图片

    byte_data = base64.b64decode(img)#将base64转换为二进制
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")# 二进制转换为一维数组
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)# 用cv2解码为三通道矩阵
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)# BGR2RGB


    # image = np.asarray(bytearray(imgdata), dtype="uint8")
    # target = cv2.imdecode(image,0)
    print(img_array)
    exit()
    #将图片保存为文件
    with open("temp.jpg",'wb') as f:
        f.write(imgdata)

    print(get_data[5])
    exit()
    get_data = get_data[-2]
    
   
    print(get_data.split('<')[0])
    exit()

    get_data = str(request.form.to_dict()).split(':')[1].split('}')[0]
    print(get_data)
    get_data = list(get_data)[-1][-1]
    print(get_data)

    # print(*request.form.lists())
    exit()
    print(content)
    content = content["content"][-1][-1]
    print(content)
    print(content.split('<')[0])
    # print(request.form[0]['content'])
    # print(request.form[0]['content'][0])
    exit()
    text = request.form["text"]
    img = request.form["img"]

    result_text = predictor.predict(text, img)

    # return_dict = json.dumps(return_dict, ensure_ascii=False)
    return result_text

# # @app.route('/getClassifyLabel',methods=['POST'])
# # def getClassifyLabel():
# #     # get_data = request.args.to_dict()
# #     # text = get_data.get('text', '') # 有“text”字段则返回value，否则返回 ''
# #     text = request.form["text"]

# #     global sess
# #     return classifyApp.getId2label()[ classifyApp.questionClassify(sess,text)[0]  ]

if __name__ == '__main__':
    

    app.run(host='0.0.0.0',port=6901)

    