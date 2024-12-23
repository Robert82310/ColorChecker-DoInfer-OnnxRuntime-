import numpy as np
import cv2
import onnxruntime as ort
from utils import get_classes

# 设置运行设备，列表中的顺序表示优先级
# providers = ['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider']
providers = [ 'CUDAExecutionProvider', 'CPUExecutionProvider']

class OnnxDoInforForYoloV5(object):
    def __init__(self, classes_path, onnx_model_path) -> None:
        self.class_names, self.num_classes = get_classes(classes_path)
        print("类别名称:", self.class_names)
        print("onnx_model_path:", onnx_model_path)
 
        # 配置一些环境,如日志，优化器，线程等等
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.intra_op_num_threads = 4
 
        # 将上述配置应用到ONNX Runtime的session中
        self.ort_session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
 
        # 获取输入名
        self.input_name = self.ort_session.get_inputs()[0].name
        print("输入名:",self.input_name) 
        self.input_shape = self.ort_session.get_inputs()[0].shape
        print("输入形状:",self.input_shape)
 
        # 获取输出名
        self.output_names = []
        self.output_shapes = []
        for i in self.ort_session.get_outputs():
            print("输出名:",i.name,i.shape)
            self.output_names.append(i.name)
            self.output_shapes.append(i.shape)

    #---------------------------------------------------#
    #   对输入图像进行resize
    #---------------------------------------------------#
    def preprocess_input(self, image, input_shape):
        ih, iw,_   = image.shape
        w, h    = input_shape
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        image = cv2.resize(image, (nw,nh))
        new_image = np.zeros((1, input_shape[1],input_shape[0],3), np.float32)+128
        new_image[0, (h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw, :] = image
        return new_image[...,::-1] / 255. , nw, nh
    

    def postprocess_output(self, boxes, box_confidence, box_class_probs,
            confidence      = 0.5,
            nms_iou         = 0.3,
            imageShapePar=[0,0,0,0], ):
        iw,ih,nw, nh = imageShapePar
        box_scores  = box_confidence * box_class_probs

        boxes_out   = []
        scores_out  = []
        classes_out = []
        for c in range(self.num_classes):
            #-----------------------------------------------------------#
            #   取出所有box_scores >= score_threshold的框，和成绩
            #-----------------------------------------------------------#
            class_boxes      = boxes
            class_box_scores = box_scores[...,c]
            #-----------------------------------------------------------#
            #   非极大抑制
            #   保留一定区域内得分最大的框
            #-----------------------------------------------------------#
            nms_index = cv2.dnn.NMSBoxes(class_boxes, class_box_scores, confidence, nms_iou)
            if len(nms_index) == 0:
                continue
            nms_index = nms_index[0]
            #-----------------------------------------------------------#
            #   获取非极大抑制后的结果
            #   下列三个分别是：框的位置，得分与种类
            #-----------------------------------------------------------#
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = c
            # 还原框:
            dlW, dlH = 0.5*(self.input_shape[2] - nw), 0.5*(self.input_shape[2] - nh)
            class_boxes = class_boxes * self.input_shape[2]
            class_boxes -= np.array([dlW, dlH, dlW, dlH])
            class_boxes = class_boxes / np.array([nw, nh, nw, nh])
            class_boxes = class_boxes * np.array([iw, ih, iw, ih])

            boxes_out.append(class_boxes)
            scores_out.append(class_box_scores)
            classes_out.append(classes)
        if len(boxes_out)==0: return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,))
        boxes_out      = np.array(  boxes_out, dtype=np.int32)
        scores_out     = np.array( scores_out, dtype=np.float32)
        classes_out    = np.array(classes_out, dtype=np.int32)

        return boxes_out, scores_out, classes_out


    def doInfer(self, image):
        ih,iw,_ = image.shape
        inputData, nw, nh = self.preprocess_input(image, self.input_shape[1:3])
        out_boxes, out_scores, out_classes = self.ort_session.run(output_names=self.output_names, input_feed={self.input_name: inputData})
        
        out_boxes, out_scores, out_classes = self.postprocess_output(out_boxes, out_scores, out_classes, imageShapePar=[iw,ih,nw, nh])
        return out_boxes, out_scores, out_classes

def draw_bbox_for_Yolo(image, bboxes, scores, classes, class_names):
    x1, y1, x2, y2 = 0,0,1,1
    for i in range(len(classes)):
        className = class_names[classes[i]]
        score = scores[i]
        x1, y1, x2, y2 = bboxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(image, className + ' ' + str(score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    cropimg = image[y1:y2,x1:x2]
    return image, cropimg

class OnnxDoInforForUNet(object):
    def __init__(self, onnx_model_path) -> None:
 
        # 配置一些环境,如日志，优化器，线程等等
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.intra_op_num_threads = 4
 
        # 将上述配置应用到ONNX Runtime的session中
        self.ort_session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
 
        # 获取输入名
        self.input_name = self.ort_session.get_inputs()[0].name
        print("输入名:",self.input_name) 
        self.input_shape = self.ort_session.get_inputs()[0].shape
        print("输入形状:",self.input_shape)
 
        # 获取输出名
        self.output_names = []
        self.output_shapes = []
        for i in self.ort_session.get_outputs():
            print("输出名:",i.name,i.shape)
            self.output_names.append(i.name)
            self.output_shapes.append(i.shape)

    #---------------------------------------------------#
    #   对输入图像进行resize
    #---------------------------------------------------#
    def preprocess_input(self, image, input_shape):
        ih, iw, _  = image.shape
        w, h    = input_shape

        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image = cv2.resize(image,(nw,nh))
        new_image = np.zeros((1 ,h,w,3), dtype=np.float32)+128
        startX, startY = (w-nw)//2, (h-nh)//2
        new_image[0, startY:startY+nh, startX:startX+nw] = image[...,::-1]
        new_image = new_image / 127.5 - 1
        return new_image, nw, nh
    

    def doInfer(self, image):
        inputData, nw, nh = self.preprocess_input(image, self.input_shape[1:3])
        pred = self.ort_session.run(output_names=self.output_names, input_feed={self.input_name: inputData})[0][0]
        pred = np.argmax(pred, axis=-1).astype(np.float32)
        ph, pw = pred.shape[:2]
        ih,iw,_ =  image.shape
        startx, starty = (pw - nw)//2 , (ph - nh)//2
        pred = pred[starty:starty+nh, startx:startx+nw]
        pred = cv2.resize(pred, (iw, ih), interpolation = cv2.INTER_LINEAR)
        pred = np.expand_dims(pred, -1)
        
        return pred

def draw_for_UNet(image, pred):
    return np.array(pred)*image


class OnnxDoInforForColorChecker(object):

    def __init__(self, classes_path,yolov5OnnxPath, uNetOnnxPath) -> None:
        self.yoloOnnxRuntime = OnnxDoInforForYoloV5(classes_path,yolov5OnnxPath)
        self.uNetOnnxRuntime = OnnxDoInforForUNet(uNetOnnxPath)

    
    def doInfer(self, image):
        # 1.  YoloV5 检测box:
        boxes, scores, classes = self.yoloOnnxRuntime.doInfer(image)
        xmin, ymin, xmax, ymax = 0,0,1,1
        for i in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[i]

            cropImg = image[ymin:ymax, xmin:xmax]
            if cropImg.shape[0]<1 or cropImg.shape[1]<1:
                continue
            mask = self.uNetOnnxRuntime.doInfer(cropImg)
            # image[ymin:ymax, xmin:xmax] = mask * image[ymin:ymax, xmin:xmax]
            image[ymin:ymax, xmin:xmax] = np.where(mask>0, mask*10, image[ymin:ymax, xmin:xmax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv2.putText(image, self.yoloOnnxRuntime.class_names[int(classes[i])]+"%2f"%scores[i], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        return image 
if __name__ == "__main__":
    import time
    colorCheckerOnnxRuntime = OnnxDoInforForColorChecker(r"OnnxModel\YoloV5Model\ColorChecker_classes0.txt", \
                                                         r"OnnxModel\YoloV5Model\yolo_colorChecker0.onnx", \
                                                         r"OnnxModel\UNetModel\unet_colorChecker24.onnx")
    
    image = cv2.imread(r"335.jpg")
    image = colorCheckerOnnxRuntime.doInfer(image)
    cv2.imwrite("335_colorChecker.jpg", image)
    cap = cv2.VideoCapture(0)
    fps = 0

    while 1:
        t1 = time.time()
        _,image = cap.read()
        image = colorCheckerOnnxRuntime.doInfer(image)
        t2 = time.time() - t1
        fps += 1./t2
        fps /= 2
        cv2.putText(image, "FPS: " + str(int(fps)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("",image)
        cv2.waitKey(1)