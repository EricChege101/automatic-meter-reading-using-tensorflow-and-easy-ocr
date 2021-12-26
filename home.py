from os import close, mkdir
import os
import os
from unicodedata import name
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
from cv2 import readOpticalFlow
import matplotlib
from tensorflow.python.framework.ops import RegisterStatistics
import shutil


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("home.ui",self)
        self.Browse.clicked.connect(self.browsefiles)
        self.algorithm.clicked.connect(self.algorithmr)
        self.path=None

    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self, 'open file')
        #self.filename.setText(fname[0])
        self.sphoto.setPixmap(QtGui.QPixmap(fname[0]))
        pathy = fname[0]
        print(fname[0])
        import cv2

        path= (f"{pathy}")
        Image_path=path
        # Importing the modules
        import os
        import shutil

        # create a dir where we want to copy and rename
        dest_dir =r"Tensorflow\workspace\images\Working folder\20.jpg" 
        #dest_dir = src_dir+"/subfolder"
        src_file = fname[0]
        shutil.copy(src_file,dest_dir) #copy the file to destination dir

        #for file in os.listdir(dest_dir):
	        #os.rename(file, r"C:\Users\bengi\tensor\Tensorflow\workspace\images\Working folder\1.jpg")

    
    
    def algorithmr(self):
        import os
        CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
        PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        paths = {
                 'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
                'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
                'APIMODEL_PATH': os.path.join('Tensorflow','models'),
                'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
                'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('Tensorflow','protoc')
                }
        
        files = {
                'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
                'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
                'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
                }
        for path in paths.values():
            if not os.path.exists(path):
                if os.name == 'posix':
                    mkdir -p (path)
                if os.name == 'nt':
                    mkdir (path)
        import os
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            print(detections)
            return detections
        import cv2 
        import numpy as np
        import sys
        from matplotlib import pyplot as plt
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'Working folder', '20.jpg')
        img = cv2.imread(IMAGE_PATH)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset+22,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            track_ids=None,
            line_thickness=22,
            min_score_thresh=.9,
            agnostic_mode=False)
        print(max(detections))   
        img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
     
        import easyocr
        detection_threshold=0.7
        image = image_np_with_detections
        region_threshold = 0.3
        def filter_text(region, ocr_result, region_threshold):
            rectangle_size = region.shape[0]*region.shape[1]
            
            meter = [] 
            for result in ocr_result:
                length = np.sum(np.subtract(result[0][1], result[0][0]))
                height = np.sum(np.subtract(result[0][2], result[0][1]))
                
                if length*height / rectangle_size > region_threshold:
                    meter.append(result[1])
            return meter
        region_threshold = 0.3
        def ocr_it(image, detections, detection_threshold, region_threshold):
            # Scores, boxes and classes above threhold
            scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
            boxes = detections['detection_boxes'][:len(scores)]
            classes = detections['detection_classes'][:len(scores)]
            # Full image dimensions
            width = image.shape[1]
            height = image.shape[0]
            # Apply ROI filtering and OCR
            for idx, box in enumerate(boxes):
                roi = box*[height, width, height, width]
                region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
                reader = easyocr.Reader(['en'])
                ocr_result = reader.readtext(region)
                
                text = filter_text(region, ocr_result, region_threshold)
                
                #plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                #plt.show()
                print(text)
                return text, region

        text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
        import csv
        import os 
        import uuid
        name=()
        '{}.jpg'.format(uuid.uuid1())
        def save_results(text, region, csv_filename, folder_path):
            img_name = '{}.jpg'.format(uuid.uuid1())
                            
            cv2.imwrite(os.path.join(folder_path, img_name), region)
            # create a dir where we want to copy and rename
            dest_dir =r"Tensorflow\workspace\images\Working folder\10.jpg" 
            #dest_dir = src_dir+"/subfolder"
            src_file = (os.path.join(folder_path, img_name))
            shutil.copy(src_file,dest_dir) #copy the file to destination dir
            with open('Tensorflow\workspace\images\Working folder\myfile.txt','r+') as myfile:
                myfile.seek(0)
                myfile.write(str(text[0]))
                myfile.truncate()
            #self.image = cv2.imread(os.path.join(folder_path, img_name))
            #self.image=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            #self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888)
            #summary=self.ephoto.setPixmap(QtGui.QPixmap.fromImage(self.image))
            #reading=text
                
            with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([img_name, text])
        save_results(text, region, 'detection_results.csv', 'Detected_Images')
        os.system('summary.py')

        #def show_image(self):
            #self.image = cv2.imread('placeholder4.PNG')
           #self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            #self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        #print(name)
        #self.ephoto.setPixmap(QtGui.QPixmap(os.path.join('Detected_Images', name)))
        #DETECT_PATH:os.path.join('Detected_Images')
        
        #self.ephoto.setPixmap(QtGui.QPixmap()

app=QApplication(sys.argv)
mainwindow=MainWindow()
Widget=QtWidgets.QStackedWidget()
Widget.addWidget(mainwindow)
Widget.setFixedWidth(951)
Widget.setFixedHeight(781)
Widget.show()
sys.exit(app.exec_())



