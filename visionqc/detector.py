from pathlib import Path
import cv2, numpy as np

CLASSES = ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"]

def inspect_image(image, model_path="models/best.onnx"):
    if Path(model_path).exists():
        try:
            return _onnx(image, model_path)
        except Exception as exc:
            return {"annotated":image.copy(),"detections":[{"class":"model_error","confidence":0.0,"details":str(exc)}],"model_used":False,"mode":"model_error"}
    return _heuristic(image)

def _heuristic(image):
    # Illustrative contour heuristic only; it is not a trained defect classifier.
    out=image.copy()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),60,150)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    h,w=gray.shape[:2]
    for c in contours:
        area=cv2.contourArea(c)
        if 120 < area < h*w*0.025:
            x,y,cw,ch=cv2.boundingRect(c)
            if cw>10 and ch>10:
                found.append({"class":"candidate_region","confidence":round(min(.65,.35+area/(h*w)),3),"x":x,"y":y,"width":cw,"height":ch})
                cv2.rectangle(out,(x,y),(x+cw,y+ch),(0,0,255),2)
    return {"annotated":out,"detections":found[:100],"model_used":False,"mode":"demo_heuristic"}

def _onnx(image, model_path):
    import onnxruntime as ort
    session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"])
    meta=session.get_inputs()[0]
    size=meta.shape[2] if len(meta.shape)==4 and isinstance(meta.shape[2],int) else 640
    h,w=image.shape[:2]
    rgb=cv2.cvtColor(cv2.resize(image,(size,size)),cv2.COLOR_BGR2RGB)
    tensor=np.transpose(rgb.astype(np.float32)/255.0,(2,0,1))[None,...]
    pred=np.asarray(session.run(None,{meta.name:tensor})[0])
    if pred.ndim==3: pred=pred[0]
    if pred.ndim!=2: raise ValueError(f"Unsupported output shape: {pred.shape}")
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256: pred=pred.T
    out=image.copy(); found=[]
    for row in pred:
        if len(row)<6: continue
        scores=row[4:]; cls=int(np.argmax(scores)); conf=float(scores[cls])
        if conf<.25: continue
        cx,cy,bw,bh=map(float,row[:4])
        x1=max(0,int((cx-bw/2)*w/size)); y1=max(0,int((cy-bh/2)*h/size))
        x2=min(w-1,int((cx+bw/2)*w/size)); y2=min(h-1,int((cy+bh/2)*h/size))
        name=CLASSES[cls] if cls<len(CLASSES) else f"class_{cls}"
        found.append({"class":name,"confidence":round(conf,3),"x":x1,"y":y1,"width":x2-x1,"height":y2-y1})
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(out,f"{name} {conf:.2f}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
    return {"annotated":out,"detections":found,"model_used":True,"mode":"onnx"}
