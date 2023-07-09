import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import subprocess
import threading
from queue import Queue
import time
from collections import OrderedDict

class Eyez:
    def __init__(self, frame_rate=4, timeout=3):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        self.labels = {}
        self.spoken_labels = {}
        self.audio_queue = Queue()
        self.exit_event = threading.Event()
        self.frame_rate = frame_rate
        self.frame_count = 0
        self.timeout = timeout

    def speak_thread(self):
        def run():
            while not self.exit_event.is_set():
                text = self.audio_queue.get().strip()
                if any(char.isalnum() for char in text):
                    subprocess.run(['spd-say', '-p', '-20', '-l', 'us', '-w', text])
                self.audio_queue.task_done()
        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def start(self):
        speak_thread_obj = self.speak_thread()
        self.audio_queue.put("Welcome to the EYEZ.")
        while True:
            ret, frame = self.video.read()
            self.frame_count += 1
            if ret:
                if self.frame_count % self.frame_rate == 0:
                    bbox, label, conf = cv.detect_common_objects(frame)
                    output_image = draw_bbox(frame, bbox, label, conf)

                    now = time.time()
                    for item in label:
                        self.labels[item] = now

                    labels_to_remove = [k for k, v in self.labels.items() if now - v > self.timeout]
                    for l in labels_to_remove:
                        del self.labels[l]
                        if l in self.spoken_labels:
                            del self.spoken_labels[l]

                    for item in label:
                        if item not in self.spoken_labels:
                            self.spoken_labels[item] = now
                            self.audio_queue.put(item)

                    labels_sorted = OrderedDict(sorted(self.labels.items(), key=lambda t: t[1]))
                    text = ', '.join(labels_sorted.keys())

                    if text != '':
                        cv2.putText(output_image, "I see a " + text + ".", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                         cv2.putText(output_image, "I see a nothing...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.imshow("EYEZ", output_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.audio_queue.put('quit')
        self.exit_event.set()

        speak_thread_obj.join()

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    eyez = Eyez()
    eyez.start()
