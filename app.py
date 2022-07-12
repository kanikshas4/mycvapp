import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase,VideoProcessorBase
from  PIL import Image
import cv2
from datetime import datetime, timedelta
import av
import numpy as np       
import cvzone
import mediapipe as mp
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from keras.utils import img_to_array     
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun4.l.google.com:19302"]}]})
idlist =[22,23,24,26,110,157,158,159,160,161,130,243]
ratioList = []

blinkCounter = 0
counter = 0
t = datetime.now()  
class VideoTransformer(VideoTransformerBase):
      

    
    def recv(self,image):
        
        
        
        frame1 = image.to_ndarray(format="bgr24")
        # frm = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        detector= FaceMeshDetector(maxFaces=1)
        
        
        
        color = (255, 0, 255)


        frame,faces= detector.findFaceMesh(frame1)
        
        #checking screen time 
        current_time = datetime.now()
        global t
        global blinkCounter
        global counter
        
        dif = current_time - t 
        mins, secs = divmod(dif.total_seconds(), 60)
        min = int(mins)
        sec = int(secs)
        timer = '{:02d}:{:02d}'.format(min, sec)
        cvzone.putTextRect(frame, timer, (390, 100), scale=1.5,
                        colorR=color)

        if dif.total_seconds() > 1200:# Here 10 means 10 seconds. Put the required time in seconds, for eg:- 5 mins = 300s
            for sec in range(15):
                cvzone.putTextRect(frame, 'Take a break', (50, 150), scale=1.5,
                        colorR=color)
        
        if faces:
            face= faces[0]
            #for id in idlist:
        #cv2.circle(frame,face[id],5,(255,0,255),cv2.FILLED)
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
    #cv2.line(frame, leftUp, leftDown, (0, 200, 0), 3)
    #cv2.line(frame, leftLeft, leftRight, (0, 200, 0), 3)
            ratio = int((lenghtVer / lenghtHor) * 100)
            ratioList.append(ratio)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)
            if ratioAvg < 35 and counter == 0:
                
                blinkCounter += 1
                color = (0,200,0)
                
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255,0, 255)
            cvzone.putTextRect(frame, f'Blink Count: {blinkCounter}', (50, 100), scale=1.5,
                        colorR=color)




            if blinkCounter <12 and dif.total_seconds() > 60:
                cvzone.putTextRect(frame, 'blink your eyes more often', (50, 300), scale=1.5,
                        colorR=color)
                
            
            pointLeft=face[145]
            pointRight=face[374]
    #cv2.line(frame,pointLeft,pointRight,(0,200,0),3)
    #cv2.circle(frame,pointRight,5,(255,0,255),cv2.FILLED)
    #cv2.circle(frame,pointLeft,5,(255,0,255),cv2.FILLED)

            w, _= detector.findDistance(pointLeft,pointRight)
            W=6.3
    #focal length

    #d=50
    #f=(w*d)/W
    #print(f)
            f=1120
            d=(W*f)/w


            cvzone.putTextRect(frame,f'Depth: {int(d)}cm',
                    (face[10][0]-100,face[10][1]-50),
                    scale=2)
            if d<50:
                cvzone.putTextRect(frame, 'stay at a minimum distance of 51 cm from the screen', (350, 500), scale=1.5,
                        colorR=color)
        
                    
    # When the person is not on the screen then the timer will not start
        else:
            t= datetime.now()
            blinkCounter = 0
    #brightness in room check
        thresh=0.5

        frame= frame/np.max(frame)

        def isbrightF():
            if np.mean(frame) > thresh:
                text = "bright"
                cv2.putText(frame, "{}".format(text), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
                color=(0,128,0)
                stroke=2
                start_point = (10, 10)
                end_point = (1250, 650)
        
                cv2.rectangle(frame,start_point,end_point,color,stroke)
        
            else:
                text= "dark,increase the light of the room"
                cv2.putText(frame, "{}".format(text), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        isbrightF()
        

        # cv2.imshow('frame',frame)       
        

        return av.VideoFrame.from_ndarray(frame1, format='bgr24')
        

def app_loopback():
    """Simple video loopback"""
    webrtc_streamer(key="loopback")


def app_video_filters():
    """Video transforms with OpenCV"""

    _type = st.radio("Select transform type", ("noop", "cartoon", "edges", "rotate"))

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if _type == "noop":
            pass
        elif _type == "cartoon":
            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)
        elif _type == "edges":
            # perform edge detection
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif _type == "rotate":
            # rotate image
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
def about():
	st.write(
		'''
This app is to detect ****Computer Vision Syndrome (CVS)**** in a person by examining various causes that lead to vision syndrome such as eye blinking rate, distance from the screen, the intensity of light in the room, and the total continuous time spend in front of the screen.

**CVS**- Digital eye strain is a group of related eye and vision problems caused by the extended computer or digital device use. Symptoms include eye discomfort and fatigue, dry eye, blurry vision, and headaches

This app is based on computer vision and will show:
1. 468 Facial Landmarks
2. Eye Blinking Rate (Blink Counter)
3. Depth Recognition ( Distance of the person from the screen)
4. Intensity of the Surrounding lighting- whether it is dark or bright.
        ''')

def app():
    st.write('''*This app is based on computer vision with following Functionalities:*''')
        
    st.write('''

        1. 468 Facial Landmarks

        2. Eye Blinking Rate (Blink Counter)

        3. Depth Recognition ( Distance of the person from the screen)

        4. Intensity of the Surrounding lighting- whether it is dark or bright.
        
        ''')
   



    webrtc_streamer(key="demo1",video_processor_factory=VideoTransformer,media_stream_constraints={"video": True, "audio": False})

def main():
   

    st.title("Computer Vision Syndrome Detector and Alert App :sunglasses: ")
    st.write("**Using the MediaPipe, OpenCv**")
    st.sidebar.markdown(
        """ 
        Developed by Kaniksha Sharma 

        Email : workkanikshasharma.com  

        [LinkedIn] (https://www.linkedin.com/in/kaniksha-sharma-4117761a7)
            
        [GitHub] (https://github.com/kanikshas4)
            
        """)

    activities = ["Home", "About","App"]
    
    choice = st.sidebar.selectbox("know more", activities)
    
   
    
    if choice == "Home":
        st.write(
		'''
This app is to detect ****Computer Vision Syndrome (CVS)**** in a person by examining various causes that lead to vision syndrome such as eye blinking rate, distance from the screen, the intensity of light in the room, and the total continuous time spend in front of the screen.

**CVS**- Digital eye strain is a group of related eye and vision problems caused by the extended computer or digital device use. Symptoms include eye discomfort and fatigue, dry eye, blurry vision, and headaches


        ''')

        st.write('''*This app is based on computer vision with following Functionalities:*''')
        
        st.write('''

        1. 468 Facial Landmarks

        2. Eye Blinking Rate (Blink Counter)

        3. Depth Recognition ( Distance of the person from the screen)

        4. Intensity of the Surrounding lighting- whether it is dark or bright.
        
        ''')
        st.write('''****Go to App section from the Slidebar to test the application**** ''')
        st.write("******Go to the About section from the sidebar to learn more about the app.******")
        
        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:5px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Kaniksha Sharma using Streamlit Framework, OpenCV, MediaPipe library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and send me a request.  </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
        webrtc_streamer(key="demo",video_processor_factory=VideoTransformer,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True, "audio": False})


    elif choice == "About":
        about()

    elif choice == 'App':
        app()

        
if __name__ == "__main__":
    main()