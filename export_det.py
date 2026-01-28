from ultralytics import YOLOE

model = YOLOE("trt_engines/temp/yoloe-v8l-seg-det.pt", task="segment")

with open("en_364.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f.readlines()]

model.set_classes(names, model.get_text_pe(names))
export_path = model.export(format='engine', half=True, device="0", imgsz=640)

