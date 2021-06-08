import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from flask import Flask, request
import time

app = Flask(__name__)


model = None
imgsz = None
out, source, weights, half, view_img, save_txt = None, None, None, None, None, None
webcam = None
device = 'cpu'
names = None
colors = None
class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        nI = len(images)

        self.img_size = img_size
        self.files = images
        self.nF = nI
        self.mode = 'images'
        self.cap = None
        assert self.nF > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (path, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]
            # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def __len__(self):
        return self.nF  # number of files


def load_model():
    global model, imgsz, out, source, weights, half, view_img, save_txt, webcam, device
    
    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt

    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    global names, colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

def detect(save_img = True):
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    global names, colors

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    res = []
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i

            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)
            cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
    print('Done. (%.3fs)' % (time.time() - t0))


def detect_v2(path, img):

    # Get names and colors
    global names, colors

    # Runlllll
    t0 = time.time()
    im0s = np.ascontiguousarray(img[:,:,::-1])
    # im0s = img[:,:,::-1]## RGB -> BGR
    img = letterbox(img, new_shape=416)[0]
    img = np.ascontiguousarray(img.transpose(2, 0, 1))

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    # Inference
    # t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    # t2 = torch_utils.time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    res = []
    for i, det in enumerate(pred):  # detections for image i

        im0 = im0s
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #with open(path + '.txt', 'a') as file:
                #    file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                
                _xyxy = list(map(int, xyxy))
                res.append([_xyxy, names[int(cls)], conf.item()])

            # Save results (image with detections)
            # cv2.imwrite(path + ".jpg", im0)

    print('Results saved to %s' % os.getcwd() + os.sep + out)
    print('Done. (%.3fs)' % (time.time() - t0))

    return res

@app.route('/detect', methods= ['POST'])
def detect():
    params = dict(request.form.items())
    im_file = request.files.get("img")
    file_name = get_file_name(params)
    im = plt.imread(im_file)
    # plt.imsave("images/store/" + file_name + ".jpg", im)
    result_list = detect_v2("images/output/" + file_name, im)
    

    res = {}
    res["resultList"] = result_list
    res["score"] = max(50, 100 - 10 * len(result_list))
    res["count"] = len(result_list)
    res["resultImg"] = None
    print(res)
    return res

def get_file_name(params:dict): # 根据userId 和时间生成文件名
    if params["deviceType"] == "machine": #
        file_name = params["userId"] + "-" + params["deviceType"] + ":" + params["deviceId"] + "-"\
            + time.strftime("%Y%m%d%H%M%S", time.localtime()) 
    else:
        file_name = params["userId"] + "-" + params["deviceType"] + "-" +\
            + time.strftime("%Y%m%d%H%M%S", time.localtime()) 
    params["fileName"] = file_name
    return file_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-waste.cfg', help='*.cfg path')

    # parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/waste.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')

    parser.add_argument('--source', type=str, default='waste_detect/input', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='waste_detect/output', help='output folder')  # output folder

    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)
    with torch.no_grad():
        load_model()

    app.run(host="0.0.0.0", port=5000,
        debug=False)