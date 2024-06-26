import json
import logging
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import detect
import detect2

logging.getLogger().setLevel(logging.INFO)


class MySimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        info = detect.main("yolo/best.pt", "yolo/")
        # self.wfile.write("GET request for {} and content is :{}".format(self.path, info).encode('utf-8'))
        self.wfile.write(str(info).encode('utf-8'))

    def do_POST(self):
        loglevel()
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
        #              str(self.path), str(self.headers), post_data.decode('utf-8'))

        # info = detect2.main1(weights=["runs/train/exp11/weights/best.pt", "yolo/models/gate-model.pt"],
        #                      source=post_data)
        start = time.time()
        # //老版本的all_in_one，模型修改版，小地图识别不准确
        # info1 = detect2.main1(weights=["runs/train/exp11/weights/best.pt"],
        #                       source=post_data, save_txt=False)
        # print("时间消耗1: " + str(time.time() - start))
        # info = detect2.main1(weights=["yolo/models/minimap-best.pt"],
        #                      source=post_data, save_txt=True)
        # minimap_info = detect2.main1(weights=["yolo/models/minimap-card-role2-best.pt"],
        #                              source=post_data, save_txt=False)
        # minimap_info = detect2.main1(weights=["yolo/models/best1.pt"],
        #                              source=post_data, save_txt=False)
        # info.extend(minimap_info)
        # predictionsInfo = json.loads("{}")
        # predictionsInfo["predictions"] = info
        # print(predictionsInfo)
        # info = str(predictionsInfo).replace('\'', '\"')

        info = detect2.main1(weights=["/Users/kuntang/PycharmProjects/yolov5/runs/train/exp8/weights/best.pt"],
                             source=post_data, save_txt=False)

        logging.info("time cost 2: " + str(time.time() - start))
        # info.extend(info1)
        predictionsInfo = json.loads("{}")
        predictionsInfo["predictions"] = info
        # logging.info(predictionsInfo)
        info = str(predictionsInfo).replace('\'', '\"')
        # info = detect1.main1(weights="yolo/best.pt", source="yolo/")
        self._set_response()
        self.wfile.write("{}".format(info).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=MySimpleHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


def loglevel():
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)


if __name__ == "__main__":
    run()
