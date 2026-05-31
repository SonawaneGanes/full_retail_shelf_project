import cv2

def draw_boxes(image_path, detections, output_path):
    img = cv2.imread(image_path)
    for d in detections:
        x1,y1,x2,y2 = map(int,d['bbox'])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,d['brand'],(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imwrite(output_path,img)
