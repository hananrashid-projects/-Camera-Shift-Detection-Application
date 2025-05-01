import json
from datetime import datetime
import subprocess
import os
import shutil
import uuid
import time
from datetime import datetime, timedelta, time as dt_time
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

TIME_UNIT="seconds"
INTERVAL=30#in time unit
SLEEP_TIME=1#in time unit

TIME_EXECUTE_WHOLE_APPLICATION=2 #in minutes
class BackgroundComparator(nn.Module):
    def __init__(self):
        super(BackgroundComparator, self).__init__()
        self.conv1=nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(2, 2)
        self.fc1=nn.Linear(64 * 64 * 64, 128)
        self.fc2=nn.Linear(128, 1)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(-1,64*64*64)
        x=torch.relu(self.fc1(x))
        x=torch.sigmoid(self.fc2(x))
        return x
def preprocess_image(image):
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)  # Add unsqueeze to add batch dimension
def compare_frames1(existing_img_path, recapture_filename):
    original_image = None
    next_image = None
    try:
        original_image = cv2.imread(existing_img_path)
        next_image=recapture_filename
        
        if original_image is None:
            print(f"Error: Could not read existing image from {existing_img_path}")
            return
        
        if next_image is None:
            print(f"Error: Could not read recaptured image from {recapture_filename}")
            return
        resized_original=cv2.resize(original_image,(500, 600))
        resized_next=cv2.resize(next_image, (500,600))
        cv2.imshow('Existing Image', resized_original)
        cv2.imshow('Recaptured Image', resized_next)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        print()
        start_time = time.time()
        original_background=preprocess_image(original_image)
        next_background=preprocess_image(next_image)
        
        original_background_gray=cv2.cvtColor(original_background[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
        next_background_gray=cv2.cvtColor(next_background[0].permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)
        original_background_tensor=torch.tensor(original_background_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        next_background_tensor=torch.tensor(next_background_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        model=BackgroundComparator()
        criterion=nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer=optim.Adam(model.parameters(), lr=0.001)


        #training loop
        epochs=16#epox
        for epoch in range(epochs):
            optimizer.zero_grad()
            output1=model(original_background_tensor)
            output2=model(next_background_tensor)
            target=torch.tensor([[0.0],[1.0]], dtype=torch.float32)  # Target labels: 0 for original background, 1 for next background
            loss=criterion(torch.cat((output1, output2)),target)
            loss.backward()#update weights backward
            optimizer.step()

        difference=torch.abs(output1-output2)
        threshold=0.5
        print(f'difference: {difference}')
        if difference.item()<=threshold:
            print("No Shift")
            shift_type="No shift"
        elif difference.item()<1:
            shift_type="Slight shift"
            print("Slight Shift")
        else:
            shift_type="Complete shift"
            print("Complete Shift")

        
        
        end_time=time.time()
        execution_time=end_time-start_time
        print(f"Execution time:{execution_time} seconds")

        return shift_type

    except Exception as e:
        print(f"Exception occurred in compare_frames: {str(e)}")
        return None
def array_initialize_cameras(config_data, start_time, org_path, report_path):
    #generate report
    generate_report(config_data, report_path)
    report=[]
    if os.path.exists(report_path):
        with open(report_path, 'r') as report_file:
            report = json.load(report_file)
    else:
        print(f"No existing report found at {report_path}. Creating new report.")

    #create the array of camera object. Update pings in report.json
    cameras=[]
    for c in config_data:
        camera={
            "camera_ip": c["camera_ip"],
            "url": c["url"],
            "stream_name": c["stream_name"],
            "stream_id": c["stream_id"],
            "start_time": start_time.strftime('%H:%M:%S'),
        }
        output_dir="C:/Users/Intership Student/Desktop/application/"+c["stream_name"]
        ping_result=ping(c["camera_ip"])
        if ping_result:
            camera["ping_status"]=True
            if os.path.exists(output_dir):
                all_frames=[f for f in os.listdir(output_dir) if f.endswith('.jpg')]
                print(f"All Frames in {c['stream_name']} is: {len(all_frames)}")
                camera["video_images"]=len(all_frames)
            for entry in report:
                if entry['stream_id']==camera["stream_id"]:
                    entry['ip_access']="accessed"

            stream_folder_path=os.path.join(org_path, c["stream_name"])
            if os.path.exists(stream_folder_path):
                print(f"Folder Exists for {c['stream_name']} at {stream_folder_path}")
            else:
                os.makedirs(stream_folder_path, exist_ok=True)
                print(f"Created Folder for {c['stream_name']} at {stream_folder_path}")

            if os.path.exists(stream_folder_path):
                image_files = [f for f in os.listdir(stream_folder_path) if f.endswith('.jpg')]
            if len(image_files) > 0:
                    camera["images"] = "exists"
                    camera["image_files"] = image_files  # Store the list of image files
            else:
                camera["images"] = "does not exist"
            cameras.append(camera)

        else:
            for entry in report:
                if entry['stream_id']==camera["stream_id"]:
                    entry['ip_access']="unreachable"
                    break
    save_report(report_path,report)
            
    return cameras
def get_total_frames(video_path):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Total Frames: {total_frames} ")
    return total_frames 
def sleep_for_minutes(minutes):
    seconds = minutes * 60
    time.sleep(seconds)
def ping(ip):
    try:
        result = subprocess.run(['ping', '-n', '4', ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        if 'Lost = 0' in output:
            print("Pinging Status: Successful Ping")
            return True
        else:
            print("Pinging Status: Ping failed")
            return False
    except subprocess.CalledProcessError:
        print("Ping failed with exception")
        return False
def create_cache_file(cache_file):
    default_cache = {
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data": {}
    }
    with open(cache_file, 'w') as file:
        json.dump(default_cache, file, indent=4)
    print(f"Created new cache file: {cache_file}")
def delete_all_folders(path):
    print("Folders Are Being Deleted")
    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
def load_file(upload_file):
    if os.path.exists(upload_file):
        with open(upload_file, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
            return data
    else:
        print(f" File '{upload_file}' not found. Creating new File.")
        create_cache_file(upload_file)
        return {}
def load_configuration(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def notify_capture_time(current_time):
    print(f"Capture time has arrived: {current_time.strftime('%H:%M:%S')}.")
def divide_24_hours(interval_min_or_secs):
    # total_minutes = 24*60
    if TIME_UNIT.lower()=="minutes":
        total_minutes=TIME_EXECUTE_WHOLE_APPLICATION
        num_intervals=total_minutes//interval_min_or_secs
    if TIME_UNIT.lower()=="seconds":
        total_seconds=TIME_EXECUTE_WHOLE_APPLICATION*60
        num_intervals=total_seconds//interval_min_or_secs
    return num_intervals
def capture_frames(cameras,org_path,report_path, cache_filename, duration_minutes):
    start_time=datetime.now()
    end_time=start_time+timedelta(minutes=duration_minutes)
    cache_data=load_file(cache_filename)
    report=load_file(report_path)
    while datetime.now()<end_time:
        current_time=datetime.now()
        notify_capture_time(current_time)
        for camera in cameras:
            if "ping_status" in camera and camera["ping_status"]:
                try:
                    total_frames=get_total_frames(camera["url"])
                    random_number=random.randint(0,total_frames)
                    print(f"Random number between 0 and {total_frames} is: {random_number}. Video:{camera['stream_name']}")
                                        
                    start_time_str=camera.get("start_time", "00:00:00")
                    start_time=datetime.strptime(start_time_str, '%H:%M:%S').time()
                    #checking if time interval have arrived
                    capture_interval=INTERVAL
                    elapsed_duration=0
                    
                    if TIME_UNIT.lower()=="minutes":
                        current_minutes=current_time.hour*60+current_time.minute
                        start_minutes=start_time.hour*60+start_time.minute
                        elapsed_duration=current_minutes-start_minutes

                    elif TIME_UNIT.lower()=="seconds":
                        current_seconds=current_time.hour*60*60+current_time.minute*60+current_time.second
                        start_seconds=start_time.hour*60*60+start_time.minute*60+start_time.second
                        elapsed_duration=current_seconds-start_seconds

                    if elapsed_duration >=0 and elapsed_duration%capture_interval==0:
                        try:
                            total_frames=get_total_frames(camera["url"])
                            random_number=random.randint(0,total_frames)
                            print(f"Random number between 0 and {total_frames} is: {random_number}. Video:{camera['stream_name']}")
                            cap=cv2.VideoCapture(camera["url"])
                            if not cap.isOpened():
                                raise IOError(f"Could not open video file at {camera['url']}")
                            cap.set(cv2.CAP_PROP_POS_FRAMES,random_number)
                            ret, frame=cap.read()
                            camera["start_time"]=current_time.strftime('%H:%M:%S')
                            cap.release()
                            folder_path=os.path.join(org_path, camera["stream_name"])
                            if TIME_UNIT.lower()=="minutes":
                                if INTERVAL%5==0:
                                    timestamp=round_minutes_in_timestamp(current_time, INTERVAL)
                                else:
                                    timestamp=current_time.strftime('%H-%M')
                            if TIME_UNIT.lower()=="seconds":
                                timestamp=current_time.strftime('%H-%M-%S')


                            filename=os.path.join(folder_path, f'{timestamp}.jpg')
                            #images already exists, find closest referance frame and pass it to the compare method
                            if "images" in camera and camera["images"] == "exists":
                                image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                                image_files.sort()
                                closest_image_path = None
                                closest_time_diff = float('inf')

                                # Find closest image by timestamp
                                for image_file in image_files:
                                    image_timestamp_str=os.path.splitext(image_file)[0]
                                    image_timestamp = datetime.strptime(image_timestamp_str, '%H-%M-%S')  # Adjust format as per your timestamps

                                    time_diff = abs((current_time - image_timestamp).total_seconds())
                                    if time_diff < closest_time_diff:
                                        closest_time_diff = time_diff
                                        closest_image_path = os.path.join(folder_path, image_file)

                                if closest_image_path and os.path.exists(closest_image_path):
                                    print(f"CLOSEST IMAGE IS {closest_image_path} for frame at {timestamp}")
                                    shift_type = compare_frames1(closest_image_path, frame)

                                    ip_access = "accessed" if camera.get("ping_status") else "not accessed"
                                    status = 0 if shift_type == "No shift" and ip_access == "accessed" else 1
                                    no_shifts_pixels = 0
                                    angle=0
                                    if shift_type == "Slight shift":
                                        no_shifts_pixels,angle = detect_pixel_difference(closest_image_path, frame)

                                    # Update report entry
                                    for entry in report:
                                        if entry['stream_id']==camera["stream_id"]:
                                            entry["shift_type"]=shift_type
                                            entry["Pixel Difference"]=str(no_shifts_pixels)
                                            entry["status"]=status
                                            entry["frames"]=ret
                                            entry["Angle Shifted"]=angle
                                    print(f"Updated report entry added for {camera['stream_name']}")
                                else:
                                    print(f"No suitable image found in {folder_path} for {camera['stream_name']}. Capturing new frame.")
                            else:
                                if cv2.imwrite(filename, frame):
                                    for entry in cache_data[camera["stream_name"]]:
                                        entry["no_image_exists"]+=1
                                        entry["image_exists"]="exists"

                                    for entry in report:
                                        if entry['stream_id']==camera["stream_id"]:
                                            entry["frames"]=ret

                                    print(f"Frame Captured & Saved at {filename} For Stream: {camera['stream_name']}.")
                                    camera["start_time"] = current_time.strftime('%H:%M:%S')
                                else:
                                    print(f"Error: Failed to save frame at {filename}")

                        except IOError as ioe:
                            print(f"IOError occurred while capturing frames for {camera['stream_name']}: {str(ioe)}")
                except Exception as e:
                    print(f"Exception occurred in processing for {camera['stream_name']}: {str(e)}")
            else:
                print(f"Camera {camera['stream_name']} is not reachable. Skipping frame capture.")

        save_report(report_path,report)
        save_cache(cache_filename, cache_data)
        print()
        if TIME_UNIT.lower()=="minutes":
            print(f"sleep for {SLEEP_TIME} minutes")
            sleep_for_minutes(SLEEP_TIME)

        if TIME_UNIT.lower()=="seconds":
            print(f"sleep for {SLEEP_TIME} seconds")
            time.sleep(SLEEP_TIME)
def detect_pixel_difference(capture_frame, frame):
    try:
        img1=cv2.imread(capture_frame)
        img2=frame

        if img1 is None:
            print(f"Error: Could not read capture frame from {capture_frame}")
            return []
        if img2 is None:
            print(f"Error: Could not read frame from input")
            return []
        img1=cv2.resize(img1,(600,500))
        img2=cv2.resize(img2,(600,500))
        height1, width1, _=img1.shape
        height2, width2, _=img2.shape
        if (height1, width1)!=(height2, width2):
            print(f"Error: Image dimensions do not match. Dimensions of img1: ({height1}, {width1}), img2: ({height2}, {width2})")
            return []
        gray_img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        fast=cv2.FastFeatureDetector_create()
        kp1=fast.detect(gray_img1, None)
        kp2=fast.detect(gray_img2, None)
        brief=cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1,des1=brief.compute(gray_img1,kp1)
        kp2,des2=brief.compute(gray_img2,kp2)
        bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        matches=bf.match(des1,des2)
        matches=sorted(matches,key=lambda x:x.distance)
        img_matches=cv2.drawMatches(img1, kp1, img2, kp2, matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(9000)
        cv2.destroyAllWindows()
        difference_list = []
        for match in matches[:1]:
            pt1 = np.int32(kp1[match.queryIdx].pt)
            pt2 = np.int32(kp2[match.trainIdx].pt)
            difference=(pt2[0]-pt1[0], pt2[1] - pt1[1])
            difference_list.append(difference)
            angle=np.degrees(np.arctan2(pt2[1]-pt1[1],pt2[0]-pt1[0]))
        print("Differences (x, y):")
        print(difference_list)
        print("Angle: ")
        print(angle)

        return difference_list,angle

    except Exception as e:
        print(f"Exception occurred in detect_pixel_difference: {str(e)}")
        return []
def generate_cache(cameras,cache_data,cache_filename,current_date):
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
    for camera in cameras:
        cache_data.setdefault(camera["stream_name"], [])
        data_entry = {
                    "stream_id": camera["stream_id"],
                    "stream_name": camera["stream_name"],
                    "image_exists": "does not exist",
                    "no_image_exists": 0,
                    "no_frames_compared": 0,
                    "date": current_date
                }
        cache_data[camera["stream_name"]].append(data_entry)
        print(f"Cache entry added for {camera['stream_name']}: {data_entry}")
        save_cache(cache_filename,cache_data)
def save_cache(cache_file, cache_data):
    with open(cache_file, 'w') as file:
        json.dump(cache_data, file, indent=4)
    print(f"Cache saved to {cache_file}")
def generate_report(cameras, report_path):
    if os.path.exists(report_path):
        return 
    report=[]
    for camera in cameras:
        current_date = datetime.now().strftime('%Y-%m-%d')
        report_entry = next((entry for entry in report if entry["stream_id"] == camera["stream_id"]), None)
        no_shifts_pixels=0
        if report_entry is None:
            report_entry = {
                "id": str(uuid.uuid4()),
                "stream_id": camera.get("stream_id", ""),
                "frames": False,
                "shift_type": "",
                "Pixel Difference":no_shifts_pixels,
                "Angle Shifted":0,
                "ip_access": "",
                "status": 1,
                "date": current_date
            }
            report.append(report_entry)
    save_report(report_path,report)
def save_report(report_path, report):
    with open(report_path, 'w') as report_file:
        json.dump(report, report_file, indent=4)
        print(f"Report saved to {report_path}")
def round_minutes_in_timestamp(current_time, interval_minutes):
    rounded_minute=round_to_nearest_interval(current_time.minute, interval_minutes)
    nearest_time=current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    
    if rounded_minute >= 60:
        additional_hours=rounded_minute // 60
        remaining_minutes=rounded_minute % 60
        
        nearest_time+=timedelta(hours=additional_hours)
        nearest_time=nearest_time.replace(minute=remaining_minutes)
    
    timestamp = nearest_time.strftime('%H-%M')
    return timestamp
def round_to_nearest_interval(minutes, interval):
        return round(minutes/interval)*interval
def main_application():
    RESET=False
    org_path='C:\\Users\\Admin\\Desktop\\application\\frame_captures'
    report_path=os.path.join(org_path,'report.json')
    cache_filename=os.path.join(org_path,'cache.json')
    config_file_path='cameras.json'

    if RESET:
        delete_all_folders(org_path)
        os.remove(report_path)
        os.remove(cache_filename)
    
    while True:    
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        start_time=datetime.now()
        print("")
        print(f"Application Restarted Start Time: {start_time.strftime('%H:%M:%S')}")
        config_data=load_configuration(config_file_path)
        cameras=array_initialize_cameras(config_data, start_time, org_path, report_path)
        terminate_application_interrupt=False  # terminate flag
        start_first_captures=False

        cache_data={}

        if os.path.exists(cache_filename):
            print(f"Cache file '{cache_filename}' exists.")
            cache_data=load_file(cache_filename)
            images_exists_recapture_allowance=False
            for camera in cameras:
                stream_name=camera["stream_name"]
                if stream_name in cache_data:
                    for entry in cache_data[stream_name]:
                        if entry.get('no_image_exists') is not None:
                            expected_images=divide_24_hours(INTERVAL)
                            if (expected_images != entry['no_image_exists'] and entry['no_image_exists'] != 0):
                                terminate_application_interrupt=True
                            elif expected_images==entry['no_image_exists']:
                                print(f'All Images Were Captured For that Day. Now Recapture Frames for {entry["stream_name"]}.')
                                images_exists_recapture_allowance=True
                else:
                    print(f"No data found for stream '{stream_name}' in cache.")
                    try:
                        os.remove(cache_filename)
                        print(f"Deleted cache file: {cache_filename}")
                    except FileNotFoundError:
                        print(f"Cache file '{cache_filename}' not found.")

            if terminate_application_interrupt:
                print("The application had stopped working in the middle.")
                delete_all_folders(org_path)
                os.remove(report_path)
                os.remove(cache_filename)
                break

        else:
            start_first_captures=True
            print(f"Cache file '{cache_filename}' does not exist.")
            start_time=datetime.now()
            current_date=start_time.strftime('%Y-%m-%d')
            generate_cache(cameras,cache_data,cache_filename,current_date)
        
        if start_first_captures or images_exists_recapture_allowance:
            capture_frames(cameras, org_path, report_path, cache_filename, duration_minutes=TIME_EXECUTE_WHOLE_APPLICATION)
        for camera in cameras:
            print(camera)
def main():
     while True:
        main_application()
        print("Restarting the loop...")
main()

