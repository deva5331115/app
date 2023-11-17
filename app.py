from flask import Flask, render_template, Response
from flask_ngrok import run_with_ngrok
import cv2
from imutils.video import VideoStream
import imutils
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
from tensorflow.python.framework import ops
import numpy as np
from PIL import Image
import cv2
import imutils
import os
from sklearn.preprocessing import LabelEncoder
from PalmTracker import *
from pubnub.callbacks import SubscribeCallback
from pubnub.enums import PNStatusCategory
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
from pyngrok import ngrok
ngrok.set_auth_token('2YGQh4G2CCy9YwtWFd4VeFGANZG_6Rf9XQmpGCAM25RdrupK9')
saved_model_path = 'C:/deva/Miniproject/automation-using-hand-gestures-master/TrainedModel/Gesture12RecognitionModelBest.keras'
loaded_model = load_model(saved_model_path)
#import tflearn
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression


ENTRY = "GestureControl"
CHANNEL = "Detect"
KILL_CONNECTION = "exit"
the_update = None

pnconfig = PNConfiguration()
pnconfig.publish_key = 'pub-c-75f1104f-f51c-4d31-9f65-a763a31e7dad'
pnconfig.subscribe_key = 'sub-c-11c98f38-0c15-45a2-949e-f881da6b6a1f'
pnconfig.uuid = "serverUUID-PUB"

pubnub = PubNub(pnconfig)
# PUBNUB


checkpoint_path = 'C:/deva/Miniproject/automation-using-hand-gestures-master/TrainedModel/Gesture12RecognitionModelBest.keras'
best_checkpoint_path = 'C:/deva/Miniproject/automation-using-hand-gestures-master/TrainedModel/Gesture12RecognitionModelBest.keras'
saved_model_path = 'C:/deva/Miniproject/automation-using-hand-gestures-master/TrainedModel/Gesture12RecognitionModelBest.keras'

n_classes = len(os.listdir('C:/deva/Miniproject/automation-using-hand-gestures-master/Dataset'))
dataset_dir = Path('C:/deva/Miniproject/automation-using-hand-gestures-master/Dataset')

                #class_names = train_ds.class_names
batch_size = 32
img_height = 100
img_width = 89
train_ds = tf.keras.utils.image_dataset_from_directory(

dataset_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
class_names = train_ds.class_names



def resizeImage(imageName):

    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(imageName)





func_map = {

    'Thumbs_up': 'Red ON',            # thumbs up
    'Thumbs_down': 'Red half ON',       # thumbs down
    'Fist': 'Red OFF',            # fist

    'Two': 'Green ON',          # two
    'Three': 'Green half ON',      # three
    'Four': 'Green OFF',          # four

    'OK': 'Get rain-value',        # ok

    'One': 'Fan ON',             # one
    'Stop': 'Fan OFF',            # stop
    'Direction_left':'TV channel change',
    'Direction_right': 'TV channel change',     # right

    'Five-palm': 'Clean floor'         # palm-five
}
def showStatistics(predictedClass, confidence):

    textImage = np.zeros((250, 512, 3), np.uint8)
    #className = labels.get(int(predictedClass), "Unknown")

    cv2.putText(textImage, "Predicted class: " + str(predictedClass),
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1)

    cv2.putText(textImage, "Confidence: " + str(confidence ) + '%',
                (5, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1)

    cv2.putText(textImage, "Actuate: " + func_map[str(predictedClass)],
                (5, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                1)

    cv2.imshow("Statistics", textImage)





    
# PUBLISHER SIDE

# PUBNUB


app = Flask(__name__)
run_with_ngrok(app)

def generate_frames():
    

        # Process the frame using your existing code
        # ... (Your existing code here)

        # Convert the frame to JPEG
        


        
        
            # initialize weight for running average
        aWeight = 0.5

        # get the reference to the webcam
        camera = cv2.VideoCapture(0)

        # region of interest (ROI) coordinates
        top, right, bottom, left = 10, 350, 225, 590

        # initialize num of frames
        num_frames = 0
        start_recording = False

        # keep looping, until interrupted
        while(True):
            # get the current frame
            (grabbed, frame) = camera.read()

            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if num_frames < 30:
                run_avg(gray, aWeight)
                print(num_frames)

            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))
                    if start_recording:
                        cv2.imwrite('Temp.png', thresholded)
                        resizeImage('Temp.png')
                        sunflower_url =  Path('C:/deva/Miniproject/automation-using-hand-gestures-master/Temp.png')
                        img_height = 100
                        img_width = 89


                        img = tf.keras.utils.load_img(
                            sunflower_url, target_size=(img_height, img_width)
                        )
                        img_array = tf.keras.utils.img_to_array(img)
                        img_array = tf.expand_dims(img_array, 0) # Create a batch

                        predictions = loaded_model.predict(img_array)
                        score = tf.nn.softmax(predictions[0])
                        
                        print(
                            "This image most likely belongs to {} with a {:.2f} percent confidence."
                            .format(class_names[np.argmax(score)], 100 * np.max(score))
                        )
                    
                        showStatistics(class_names[np.argmax(score)], 100 * np.max(score))

                        # PUBNUB integration
                        the_update = str(class_names[np.argmax(score)])
                        the_message = {"entry": ENTRY, "update": the_update}
                        envelope = pubnub.publish().channel(CHANNEL).message(the_message).sync()


                        if envelope.status.is_error():
                            print("[PUBLISH: fail]")
                            print("error: {}".format(status.error))
                        else:
                            print("[PUBLISH: sent]")
                            print(f"Sent: {the_update}")


                        

                        # PUBNUB integration

                    cv2.imshow("Thresholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                the_update = KILL_CONNECTION
                break

            if keypress == ord("s"):
                start_recording = True


        # Model defined




        

    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
